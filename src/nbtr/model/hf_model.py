from transformers import PreTrainedModel
from huggingface_hub import HfApi
from .gpt2 import GPT
from nbtr.trainer import TrainerConfig
from .hf_model_config import HfModelConfig
from transformers.utils import cached_file
import torch
import torch.nn as nn
import os
from typing import Optional

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
    
    def save_pretrained(
        self,
        save_directory: str,
        repo_id: Optional[str],
        push_to_hub: bool = False
    ):
        HfModel.save(self._model, save_directory)

        if push_to_hub:
            ## TODO: fix repo id
            HfModel.upload_saved(save_directory, repo_id)
        
    @staticmethod
    def from_pretrained(repo_id):
        hf_cfg = HfModelConfig.from_pretrained(repo_id)
        hf_model = HfModel(hf_cfg)

        # update model state with the latest state from the repo
        model_file = cached_file(repo_id, FILE_NAME, _raise_exceptions_for_missing_entries=True)
        model_state = torch.load(model_file, torch.device(hf_model.device))
        hf_model._model.load_state_dict(model_state)
            
        return hf_model
    
    @staticmethod
    def save(model:nn.Module, save_directory:str):
        torch.save(model.state_dict(), HfModel._get_path(save_directory))
    
    @staticmethod
    def upload_saved(save_directory:str, repo_id:str):
        HfApi().upload_file(path_or_fileobj=HfModel._get_path(save_directory), path_in_repo=FILE_NAME, repo_id=repo_id)
    
    @staticmethod
    def _get_path(save_directory:str):
        return os.path.join(save_directory, FILE_NAME)