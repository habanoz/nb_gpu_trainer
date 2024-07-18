import sentencepiece as spm
from datasets import load_dataset
import numpy as np
import os
from transformers.utils import cached_file

class Tokenizer:
    def __init__(self, tokenizer_model_file) -> None:
        self.sp = spm.SentencePieceProcessor(model_file=tokenizer_model_file)
    
    @staticmethod
    def from_pretrained(repo_id):
        tokenizer_model_file = cached_file(repo_id, "tokenizer.model", _raise_exceptions_for_missing_entries=True)
        return Tokenizer(tokenizer_model_file)
    
    def decode(self, ids):
        if 'input_ids' in ids:
            ids = ids['input_ids']
        return self.sp.Decode(ids)

    def encode(self, text:str):
        input_ids = [self.sp.bos_id()]+self.sp.encode(text)+[self.sp.eos_id()]
        return {"input_ids": input_ids}
    
    def encode_all(self, dataset, value_key):
        columns = dataset.column_names
        assert value_key in columns, f"Column {value_key} not found in column list: [{columns}]"

        return dataset.map(lambda example: self.encode(example[value_key]), batched=True, remove_columns=columns)
    
    def encode_ds_from_hub(self, dataset_repo_id, data_dir, value_key="text"):
        ds = load_dataset(dataset_repo_id)
        self.encode_training_data(ds, data_dir, value_key)

    def encode_training_data(self, dataset, data_dir, value_key="text"):
        assert "train" in dataset, "Dataset does not contain split 'train'!"
        assert "validation" in dataset, "Dataset does not contain split 'validation'!"

        dataset = self.encode_all(dataset, value_key=value_key)

        self._save_ids(dataset,data_dir,"train")
        self._save_ids(dataset,data_dir,"validation")

    def _save_ids(self, dataset, data_dir, name="train"):
        dataset = dataset[name]
        flat_ids = [id for sample in dataset for id in sample['input_ids']]

        print(f"Number of '{name}' tokens: {len(flat_ids)}")

        ids = np.array(flat_ids, dtype=np.uint16)
        assert len(ids.shape)==1 # this must be a flat array
        ids.tofile(os.path.join(data_dir, f'{name}.bin'))