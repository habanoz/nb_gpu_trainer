import torch
from collections import defaultdict
from functools import partial
import numpy as np

class MupIteration:
    def __init__(self, model, config):
        self.config = config
        self.model = model
        self.enabled = config.mup_enable_coord_check_logging
        self._coord_check_dict = {key + '_act_abs_mean':0.0 for key in ['token_embedding', 'attn', 'mlp', 'lm_head']}
        self.coord_check_handles = []
            
    def __enter__(self):
        if not self.enabled:
            return self
        
        self._coord_check_dict = defaultdict(list)

        def hook(module, input, output, key):
            with torch.inference_mode():
                self._coord_check_dict[key+ '_act_abs_mean'].append(output.abs().mean().item())

        coord_check_handles = []
        for module_name, module in self.model.named_modules():
            if module_name == 'transformer.wte':
                coord_check_handles.append(module.register_forward_hook(partial(hook, key='token_embedding')))
            elif module_name.endswith('.attn'):
                coord_check_handles.append(module.register_forward_hook(partial(hook, key='attn')))
            elif module_name.endswith('.mlp'):
                coord_check_handles.append(module.register_forward_hook(partial(hook, key='mlp')))
            elif module_name == 'lm_head':
                coord_check_handles.append(module.register_forward_hook(partial(hook, key='lm_head')))

        self.coord_check_handles = coord_check_handles

        return self

    def __exit__(self, exc_type, exc_value, tb):
        for handle in self.coord_check_handles:
            handle.remove()
            
        if exc_type:
            raise
        
        return True
    
    @property
    def coord_check_dict(self):
        return  {key:np.mean(self._coord_check_dict[key]) for key in self._coord_check_dict}
    
class MupSession:
    def __init__(self, model, config):
        self.config = config
        self.model = model
        self.enabled = config.mup_enable_coord_check_logging
        self._iteration = MupIteration(model=model, config=config)

    def __iter__(self):
        return self

    def __next__(self):
        if not self.enabled:
            return self._iteration
        
        if True:
            self._iteration = MupIteration(model=self.model, config=self.config)
            return self._iteration
        raise StopIteration
    
    @property
    def last_coord_check_dict(self):
        if not self.enabled:
            return {}
        return self._iteration.coord_check_dict