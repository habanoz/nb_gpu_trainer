from transformers import PreTrainedModel
from huggingface_hub import HfApi
from .gpt2 import GPT
from nbtr.trainer import TrainerConfig
from .hf_model_config import HfModelConfig
from transformers.utils import cached_file
import torch
import torch.nn as nn
import os

FILE_NAME = 'pytorch_model.bin'

class HfModel(PreTrainedModel):
    config_class = HfModelConfig
    def __init__(self, config:HfModelConfig):
        super().__init__(config)
        self._model = GPT(config=config.gpt_config)

    def forward(self, x, y):
        logits, loss =  self._model(x,y)
        return {'logits':logits, 'loss':loss}
    
    @property
    def model(self):
        return self._model
    
    @staticmethod
    def from_pretrained(repo_id):
        hf_cfg = HfModelConfig.from_pretrained(repo_id)
        hf_model = HfModel(hf_cfg)
        
        # update model state with the latest state from the repo
        model_file = cached_file(repo_id, FILE_NAME, _raise_exceptions_for_missing_entries=True)
        model_state = torch.load(model_file, torch.device(hf_model.device))
        hf_model._model.load_state_dict(model_state)
            
        return hf_model
    
    def save(self, model:nn.Module, trainer_config:TrainerConfig):
        torch.save(model.state_dict(), self._get_path(trainer_config))
    
    def upload_saved(self, trainer_config:TrainerConfig):
        HfApi().upload_file(path_or_fileobj=self._get_path(trainer_config), path_in_repo=FILE_NAME, repo_id=trainer_config.repo_id)
        
    def _get_path(self, trainer_config:TrainerConfig):
        return os.path.join(trainer_config.out_dir, FILE_NAME)