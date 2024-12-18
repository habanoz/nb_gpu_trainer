from huggingface_hub import HfApi
from .gpt2_light_rope_flex_attention import GPT
from .hf_model_config import HfModelConfig
from transformers.utils import cached_file
import torch
import torch.nn as nn
import os
from typing import Optional
from .trainer_model import TrainerModel

FILE_NAME = 'pytorch_model.bin'

class HfModel(TrainerModel):
    config_class = HfModelConfig
    def __init__(self, hf_model_config:HfModelConfig):
        _model = GPT(config=hf_model_config.gpt_config)
        self._config = hf_model_config
        
        super().__init__(_model)
        
    def forward(self, x, y=None):
        logits, loss =  self.model(x,y)
        return {'logits':logits, 'loss':loss}
    
    def to(self, device:str):
        self.model.to(device)
    
    @property
    def config(self):
        return self._config
    
    def save_pretrained(
        self,
        save_directory: str,
        repo_id: Optional[str],
        push_to_hub: bool = True
    ):  
        self._config.save(save_directory)
        HfModel.save(self.model, save_directory)

        if push_to_hub:
            self._config.upload_saved(save_directory, repo_id=repo_id)
            HfModel.upload_saved(save_directory, repo_id)
        
    @staticmethod
    def from_pretrained(repo_id, device:str='cuda'):
        # update model state with the latest state from the repo
        model_file = cached_file(repo_id, FILE_NAME, _raise_exceptions_for_missing_entries=True)
        model_state = torch.load(model_file, torch.device(device))

        hf_cfg = HfModelConfig.from_pretrained(repo_id)
        hf_model = HfModel(hf_cfg)
        hf_model.to(device=device)
        model_state = { (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k):v for k,v in model_state.items()}
        hf_model.model.load_state_dict(model_state)
        print("Restored model state from repository!")

        return hf_model
    
    @staticmethod
    def from_pretrained_or_config(repo_id:str, hf_model_config:HfModelConfig, device:str='cuda'):
        try:
            return HfModel.from_pretrained(repo_id=repo_id, device=device)
        except Exception as e:
            print("Unable to get model state from repo! Training from scratch!!!"+str(e))
        
        model = HfModel(hf_model_config=hf_model_config)
        model.to(device)

        return model
    
    @staticmethod
    def save(model:nn.Module, save_directory:str):
        torch.save(model.state_dict(), HfModel._get_path(save_directory))
    
    @staticmethod
    def upload_saved(save_directory:str, repo_id:str):
        try:
            HfApi().upload_file(path_or_fileobj=HfModel._get_path(save_directory), path_in_repo=FILE_NAME, repo_id=repo_id)
        except Exception as e:
            print("Uploading model failed! "+str(e))
    
    @staticmethod
    def _get_path(save_directory:str):
        return os.path.join(save_directory, FILE_NAME)