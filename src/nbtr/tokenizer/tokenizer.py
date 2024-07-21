import sentencepiece as spm
from datasets import load_dataset
import numpy as np
import os
from transformers.utils import cached_file
from huggingface_hub import HfApi

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

        return dataset.map(lambda example: self.encode(example[value_key]), batched=True, remove_columns=columns)
    
    def encode_ds_from_hub(self, dataset_repo_id, data_dir, value_key="text"):
        ds = load_dataset(dataset_repo_id)
        self.encode_training_data(ds, data_dir, value_key)

    def encode_training_data(self, dataset, data_dir, value_key="text"):
        assert "train" in dataset, "Dataset does not contain split 'train'!"
        assert "validation" in dataset, "Dataset does not contain split 'validation'!"

        dataset = self.encode_all(dataset, value_key=value_key)

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        self._save_ids(dataset,data_dir,"train")
        self._save_ids(dataset,data_dir,"validation")

    def _save_ids(self, dataset, data_dir, name="train"):
        dataset = dataset[name]
        flat_ids = [id for sample in dataset for id in sample['input_ids']]

        print(f"Number of '{name}' tokens: {len(flat_ids)}")

        ids = np.array(flat_ids, dtype=np.uint16)
        assert len(ids.shape)==1 # this must be a flat array
        ids.tofile(os.path.join(data_dir, f'{name}.bin'))