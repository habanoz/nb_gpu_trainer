from nbtr.model.gpt2 import GPTConfig
from huggingface_hub import HfApi
from dataclasses import asdict
from transformers.utils import cached_file
import json 
import os 

FILE_NAME="config.json"

class HfModelConfig():
    def __init__(self, gpt_config:GPTConfig):
        assert gpt_config is not None
        self._gpt_config = gpt_config
        super().__init__()

    @property
    def gpt_config(self) -> GPTConfig:
        return self._gpt_config

    def save_pretrained(
        self,
        out_dir,
        repo_id,
        push_to_hub: bool = True
    ):
        self.save(out_dir)

        if push_to_hub:
            self.upload_saved(out_dir, repo_id)

    @staticmethod
    def from_pretrained(repo_id):
        config_file = cached_file(repo_id, FILE_NAME, _raise_exceptions_for_missing_entries=True)
        with open(config_file) as f:
            doc = json.load(f)

        return HfModelConfig(gpt_config=GPTConfig(**doc['gpt_config']))

    def save(self, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        with open(self._get_path(out_dir), "w") as f:
            json.dump(self._to_dict(), f, indent=2)

    def upload_saved(self, out_dir, repo_id):
        assert repo_id is not None
        HfApi().upload_file(path_or_fileobj=self._get_path(out_dir), path_in_repo=FILE_NAME, repo_id=repo_id)

    def _get_path(self, out_dir):
        return os.path.join(out_dir, FILE_NAME)

    def _to_dict(self):
        return {"gpt_config": asdict(self.gpt_config)}