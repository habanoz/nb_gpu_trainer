from transformers import PretrainedConfig
from nbtr.model.gpt2 import GPTConfig
from dataclasses import asdict

class HfModelConfig(PretrainedConfig):
    def __init__(self, gpt_config:GPTConfig=None, **kwargs):
        gpt_config = GPTConfig()
        self.gpt_config_map = asdict(gpt_config) ## cannot serialize GPTConfig object into JSON
        super().__init__(**kwargs)
    
    @property
    def gpt_config(self)->GPTConfig:
        return GPTConfig(**self.gpt_config_map)