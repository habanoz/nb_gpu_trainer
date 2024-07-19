from transformers import PretrainedConfig

class MyGPT2Config(PretrainedConfig):
    def __init__(self, gpt_config:dict=None, **kwargs):
        self.gpt_config = gpt_config
        super().__init__(**kwargs)
        