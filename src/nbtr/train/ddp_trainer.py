from nbtr.model.hf_trainer_config import HfTrainerConfig
from .trainer import Trainer, TrainerEvent, TrainingState
from nbtr.model.hf_trainer_state import HFTrainerState
from nbtr.model.hf_model import HfModel
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import logging

logger = logging.getLogger(__name__)

class DDPTrainer():
    def __init__(self, trainer:Trainer) -> None:
        self.trainer = trainer
        self.add_callback(TrainerEvent.ON_LAST_MICRO_BATCH, lambda trainer, model : self.do_on_last_micro(model))
    
    def train(self, model:Module):
        if self.model.config.compile:
            logger.info("compiling the model...")
            model = torch.compile(model)
            logger.info("compiling the model done!")
            
        ddp_model = DDP(model)
        
        self.trainer.train(model=ddp_model, raw_model=model, compile=False)

    def do_on_last_micro(self, model:DDP):
        model.require_backward_grad_sync = True
    
    def set_state(self, state: TrainingState):
        self.trainer.set_state(state)
    
    def add_callback(self, event: TrainerEvent, callback):
        self.trainer.add_callback(event, callback)
        