from huggingface_hub import HfApi
from transformers.utils import cached_file
from nbtr.model.hf_trainer_config import HfTrainerConfig
from nbtr.train.trainer import TrainingState
import torch
import os

FILE_NAME = "trainer_state.bin"

class HFTrainerState:
    def __init__(self, state:TrainingState, config:HfTrainerConfig):
        assert isinstance(config,HfTrainerConfig )
        self.state = state if state is not None else TrainingState()
        self.config = config
    
    @staticmethod
    def from_pretrained_or_none(config:HfTrainerConfig):
        trainer_state_file = cached_file(config.repo_id, FILE_NAME, _raise_exceptions_for_missing_entries=False)

        if trainer_state_file is None:
            return None
            
        trainer_state = torch.load(trainer_state_file)

        return HFTrainerState(trainer_state, config)

    def save(self):
        torch.save(self.state, self._get_path())
    
    def upload_saved(self):
        try:
            HfApi().upload_file(path_or_fileobj=self._get_path(), path_in_repo=FILE_NAME, repo_id=self.config.repo_id)
        except Exception as e:
            print("Uploading state failed:"+str(e))
        
    def _get_path(self):
        return os.path.join(self.config.trainer_config.out_dir, FILE_NAME)