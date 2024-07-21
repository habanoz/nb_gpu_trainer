from dataclasses import replace, asdict
from nbtr.train.trainer import TrainerConfig
from huggingface_hub import HfApi
from transformers.utils import cached_file
import os
import json

FILE_NAME = "trainer.json"


class HfTrainerConfig():
    def __init__(self, repo_id: str, ds_repo_id: str, trainer_config: TrainerConfig):
        self.repo_id = repo_id
        self.ds_repo_id = ds_repo_id
        self._trainer_config = trainer_config

        if self._trainer_config.out_dir is None:
            out_dir = repo_id.split("/")[-1]
            self._trainer_config = replace(self._trainer_config, out_dir=out_dir)

        if self._trainer_config.data_dir is None:
            data_dir = ds_repo_id.split("/")[-1]
            self._trainer_config = replace(self._trainer_config, data_dir=data_dir)

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
            HfTrainerConfig.upload_saved()

    @staticmethod
    def from_pretrained(repo_id):
        config_file = cached_file(
            repo_id, FILE_NAME, _raise_exceptions_for_missing_entries=True)
        with open(config_file) as f:
            doc = json.load(f)

        return HfTrainerConfig(trainer_config=TrainerConfig(**doc))

    def save(self):
        with open(HfTrainerConfig._get_path(), "w") as f:
            json.dump(self._to_dict(), f)

    def upload_saved(self):
        repo_id = self.trainer_config.repo_id
        HfApi().upload_file(path_or_fileobj=self._get_path(), path_in_repo=FILE_NAME, repo_id=repo_id)

    def _get_path(self):
        return os.path.join(self.trainer_config.out_dir, FILE_NAME)

    def _to_dict(self):
        return {"repo_id": self.repo_id, "ds_repo_id": self.ds_repo_id, "trainer_config": asdict(self.trainer_config)}
