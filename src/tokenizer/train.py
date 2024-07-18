from datasets import load_dataset
import sentencepiece as spm
import tempfile
from huggingface_hub import HfApi
import os

class Trainer:
    def __init__(self, repo_id, out_dir) -> None:
        self.api = HfApi()
        self.repo_id = repo_id
        self.out_dir = out_dir

    def train(self):
        ds = load_dataset(self.repo_id)
        
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        
        # create a temporary long text file and remove it after training
        with tempfile.NamedTemporaryFile(mode='w', delete_on_close=False) as fp:
            for doc in ds['train']:
                fp.write(doc['text'])
            for doc in ds['validation']:
                fp.write(doc['text'])
            fp.close()
            
            spm.SentencePieceTrainer.Train(input=fp.name, model_prefix=f"{self.out_dir}/tokenizer", vocab_size=2**13, model_type="bpe", split_digits=True)

        if not self.api.repo_exists(repo_id=self.repo_id):
            repo = self.api.create_repo(repo_id=self.repo_id, private=True)
            print(f"Created repo {repo}")
            
        self.api.upload_file(path_or_fileobj=f"{self.out_dir}/tokenizer.model", path_in_repo="tokenizer.model", repo_id=self.repo_id)
        self.api.upload_file(path_or_fileobj=f"{self.out_dir}/tokenizer.vocab", path_in_repo="tokenizer.vocab", repo_id=self.repo_id)

        print(f"Files uploaded to the repo at location {self.repo_id}")