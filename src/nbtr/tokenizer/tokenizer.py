import sentencepiece as spm
from datasets import load_dataset
import numpy as np
import os
from transformers.utils import cached_file
from huggingface_hub import HfApi
from tqdm import tqdm

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
        if 'input_ids' in ids:
            ids = ids['input_ids']
        return self.sp.Decode(ids)
    
    @property
    def eos_id(self):
        self.sp.eos_id
    
    @property
    def bos_id(self):
        self.sp.bos_id
        
    def encode(self, text:str):
        input_ids = self.sp.Encode(text,add_bos=True, add_eos=True)
        return {"input_ids": input_ids}
    
    def encode_all(self, dataset, value_key):
        columns = dataset['train'].column_names
        assert value_key in columns, f"Column {value_key} not found in column list: [{columns}]"

        return dataset.map(lambda example: self.encode(example[value_key]), batched=True, batch_size=500, remove_columns=columns, desc="tokenizing the splits", num_proc=1)
    
    def encode_ds_from_hub(self, dataset_repo_id, data_dir, value_key="text"):
        ds = load_dataset(dataset_repo_id)
        
        if "val" not in ds:
            print("No validation split is found. Creating a validation split.")
            ds = ds["train"].train_test_split(test_size=0.055, seed=2357, shuffle=True)
            ds['val'] = ds.pop('test') # rename the test split to val
            print("Validation split created.")
        try:
            self.encode_training_data(ds, data_dir, value_key)
        except Exception as e:
            print(e)

    def encode_training_data(self, dataset, data_dir, value_key="text"):
        assert "train" in dataset, "Dataset does not contain split 'train'!"
        assert "val" in dataset, "Dataset does not contain split 'val'!"

        dataset = self.encode_all(dataset, value_key=value_key)

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        for split, dset in dataset.items():
            arr_len = sum(len(ids) for ids in dset["input_ids"])
            
            filename = os.path.join(data_dir, f'{split}.bin')
            dtype = np.uint16
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
            
            total_batches = 1024 if len(dset) > 1024 else 1
            
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
            
        # self._save_ids(dataset,data_dir,"train")
        # self._save_ids(dataset,data_dir,"val")

    def _save_ids(self, dataset, data_dir, name="train"):
        dataset = dataset[name]
        flat_ids = [id for sample in dataset for id in sample['input_ids']]

        print(f"Number of '{name}' tokens: {len(flat_ids)}")

        ids = np.array(flat_ids, dtype=np.uint16)
        assert len(ids.shape)==1 # this must be a flat array
        ids.tofile(os.path.join(data_dir, f'{name}.bin'))