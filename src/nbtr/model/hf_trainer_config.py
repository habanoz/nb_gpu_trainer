from dataclasses import replace, asdict
from nbtr.train.trainer import TrainerConfig
from huggingface_hub import HfApi
from transformers.utils import cached_file
import os
import json

FILE_NAME = "trainer.json"

class HfTrainerConfig():
    def __init__(self, repo_id: str, trainer_config: TrainerConfig, init_repo_id=None):
        assert "/" in repo_id, f"Repo ID '{repo_id}' is must be in the form `user/repo_name`"
        
        self.repo_id = repo_id
        self.init_repo_id = init_repo_id
        self._trainer_config = trainer_config

        assert trainer_config.data_dir is not None, "Data dir cannot be None"

        super().__init__()

    @property
    def trainer_config(self) -> TrainerConfig:
        return self._trainer_config

    def save_pretrained(
        self,
        push_to_hub: bool = True
    ):
        self.save()

        if push_to_hub:
            HfApi().create_repo(repo_id=self.repo_id, private=True, exist_ok=True)
            self.upload_saved()

    @staticmethod
    def from_pretrained(repo_id):
        config_file = cached_file(repo_id, FILE_NAME, _raise_exceptions_for_missing_entries=True)
        with open(config_file) as f:
            doc = json.load(f)

        return HfTrainerConfig(repo_id=repo_id, trainer_config=TrainerConfig(**doc['trainer_config']))

    def save(self):
        if not os.path.exists(self.trainer_config.out_dir):
            os.makedirs(self.trainer_config.out_dir)
            
        with open(self._get_path(), "w") as f:
            json.dump(self._to_dict(), f, indent=2)

    def upload_saved(self):
        HfApi().upload_file(path_or_fileobj=self._get_path(), path_in_repo=FILE_NAME, repo_id=self.repo_id)

    def _get_path(self):
        return os.path.join(self.trainer_config.out_dir, FILE_NAME)

    def _to_dict(self):
        return {"repo_id": self.repo_id, "trainer_config": asdict(self.trainer_config)}
