import sentencepiece as spm
from datasets import load_dataset
import numpy as np
import os
from transformers.utils import cached_file
from huggingface_hub import HfApi
from tqdm import tqdm
import multiprocess as mp
from nbtr.data.data_common import write_datafile

TOK_FILE_NAME="tokenizer.model"
VOC_FILE_NAME="tokenizer.vocab"
class Tokenizer:
    def __init__(self, tokenizer_model_file) -> None:
        self.sp = spm.SentencePieceProcessor(model_file=tokenizer_model_file)
    
    @staticmethod
    def from_pretrained(repo_id):
        tokenizer_model_file = cached_file(repo_id, TOK_FILE_NAME, _raise_exceptions_for_missing_entries=True)
        return Tokenizer(tokenizer_model_file)
    
    @staticmethod
    def copy_pretrained(from_repo_id, to_repo_id):
        hfApi = HfApi()

        if not hfApi.repo_exists(repo_id=to_repo_id):
            repo = hfApi.create_repo(repo_id=to_repo_id)
            print("Repo created:", repo)

        tokenizer_model_file = cached_file(from_repo_id, TOK_FILE_NAME, _raise_exceptions_for_missing_entries=True)
        hfApi.upload_file(path_or_fileobj=tokenizer_model_file, path_in_repo=TOK_FILE_NAME, repo_id=to_repo_id)

        tokenizer_vocab_file = cached_file(from_repo_id, VOC_FILE_NAME, _raise_exceptions_for_missing_entries=True)
        hfApi.upload_file(path_or_fileobj=tokenizer_vocab_file, path_in_repo=VOC_FILE_NAME, repo_id=to_repo_id )
    
    def decode(self, ids):
        if isinstance(ids, dict) and 'input_ids' in ids:
            ids = ids['input_ids']
        return self.sp.Decode(ids)
    
    @property
    def eos_id(self):
        return self.sp.eos_id()
    
    @property
    def bos_id(self):
        return self.sp.bos_id()
        
    def encode(self, text:str, bos=True, eos=True):
        input_ids = self.sp.Encode(text,add_bos=bos, add_eos=eos)
        return {"input_ids": input_ids, 'len': [len(tokens) for tokens in input_ids] if isinstance(input_ids[0], list) else len(input_ids)}
    
    def encode_all(self, dataset, value_key):
        columns = dataset['train'].column_names
        assert value_key in columns, f"Column {value_key} not found in column list: [{columns}]"
        print("Generating tokens!")
        return dataset.map(lambda example: self.encode(example[value_key]), batched=True, batch_size=500, remove_columns=columns, desc="tokenizing the splits")
    
    def encode_ds_from_hub(self, dataset_repo_id, data_dir, filename, shard_size=None, value_key="text"):
        ds = load_dataset(dataset_repo_id)
        
        if 'validation' in ds:
            print("Renaming column 'validation' to 'val'!")
            ds['val']=ds['validation']
            del ds['validation']
        
        if "val" not in ds:
            print("No validation split is found. Creating a validation split.")
            ds = ds["train"].train_test_split(test_size=0.001, seed=2357, shuffle=True)
            ds['val'] = ds.pop('test') # rename the test split to val
            print("Validation split created.")
        try:
            if shard_size:
                self.encode_training_data_shards(ds, data_dir, filename, shard_size, value_key)
            else:
                self.encode_training_data(ds, data_dir, filename, value_key)
        except Exception as e:
            print("Error:"+str(e))

    def encode_training_data(self, dataset, data_dir, name="", value_key="text"):
        assert "train" in dataset, "Dataset does not contain split 'train'!"
        assert "val" in dataset, "Dataset does not contain split 'val'!"

        dataset = self.encode_all(dataset, value_key=value_key)

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        if name: name=name+"_"
        
        for split, dset in dataset.items():
            arr_len = sum(dset['len'])

            print(f"Split {split}, token count: {arr_len}")
            
            filename = os.path.join(data_dir, f'{name}{split}.bin')
            arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(arr_len,))
            
            total_batches = 1024*1024 if len(dset) > 1024*1024 else (1024 if len(dset) > 1024 else 1)
            print(f"Saving split '{split}', docs: {len(dset)}, tokens: {arr_len} in {total_batches} batches.")
            
            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                # Batch together samples for faster write
                batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                arr_batch = np.concatenate(batch['input_ids'])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()
            
            print(f"Saved '{split}' tokens.")
            
    def encode_training_data_shards(self, dataset, data_dir, name="", shard_size=10**8, value_key="text"):
        assert "train" in dataset, "Dataset does not contain split 'train'!"
        assert "val" in dataset, "Dataset does not contain split 'val'!"
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        def encode_text_key(row):
            text = row[value_key]
            return self.encode(text)
        
        if name: name=name+"_"
        
        for split, dset in dataset.items():
            # tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
            nprocs = max(1, os.cpu_count() - 2) # don't hog the entire system
            with mp.Pool(nprocs) as pool:
                shard_index = 0
                # preallocate buffer to hold current shard
                all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
                token_count = 0
                progress_bar = None

                for encoded in pool.imap(encode_text_key, dset, chunksize=16):
                    tokens = encoded['input_ids']
                    # is there enough space in the current shard for the new tokens?
                    if token_count + len(tokens) < shard_size:
                        # simply append tokens to current shard
                        all_tokens_np[token_count:token_count+len(tokens)] = tokens
                        token_count += len(tokens)
                        # update progress bar
                        if progress_bar is None:
                            progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                        progress_bar.update(len(tokens))
                    else:
                        # write the current shard and start a new one
                        filename = os.path.join(data_dir, f"{name}{split}_{shard_index:06d}.bin")
                        # split the document into whatever fits in this shard; the remainder goes to next one
                        remainder = shard_size - token_count
                        progress_bar.update(remainder)
                        all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                        write_datafile(filename, all_tokens_np.tolist())
                        shard_index += 1
                        progress_bar = None
                        # populate the next shard with the leftovers of the current doc
                        all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                        token_count = len(tokens)-remainder

                # write any remaining tokens as the last shard
                if token_count != 0:
                    filename = os.path.join(data_dir, f"{split}_{shard_index:06d}.bin")
                    write_datafile(filename, (all_tokens_np[:token_count]).tolist())

    def _save_ids(self, dataset, data_dir, name="train"):
        dataset = dataset[name]
        flat_ids = [id for sample in dataset for id in sample['input_ids']]

        print(f"Number of '{name}' tokens: {len(flat_ids)}")

        ids = np.array(flat_ids, dtype=np.uint16)
        assert len(ids.shape)==1 # this must be a flat array
        ids.tofile(os.path.join(data_dir, f'{name}.bin'))