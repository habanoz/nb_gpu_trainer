from transformers import PreTrainedModel
from .gpt2 import GPT, GPTConfig
from .cfg_mygpt2 import MyGPT2Config

class MyGTP2(PreTrainedModel):
    config_class = MyGPT2Config
    def __init__(self, config):
        super().__init__(config)
        self.model = GPT(config=GPTConfig(**config.gpt_config))

    def forward(self, tensor):
        return self.model(tensor)