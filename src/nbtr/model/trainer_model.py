import torch.nn as nn

class TrainerModel():
    def __init__(self, model: nn.Module, raw_model: nn.Module=None):
        self._model = model
        self._raw_model = raw_model if raw_model else model
    
    @property
    def model(self):
        return self._model
    
    @property
    def raw_model(self):
        return self._raw_model