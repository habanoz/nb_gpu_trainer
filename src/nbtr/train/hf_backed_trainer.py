from nbtr.model.hf_trainer_config import HfTrainerConfig
from .trainer import Trainer, TrainerEvent
from nbtr.model.hf_trainer_state import HFTrainerState
from nbtr.model.hf_model import HfModel
import os
from torch.nn import Module
from huggingface_hub import HfApi
from concurrent.futures import ThreadPoolExecutor

import logging

logger = logging.getLogger(__name__)

class HFBackedTrainer:
    num_workers = 2
    def __init__(self, hf_trainer_config:HfTrainerConfig, trainer, rank:int = 0) -> None:
        assert hf_trainer_config.repo_id is not None, "Repo id cannot be None"

        self.hf_trainer_config = hf_trainer_config
        self.rank = rank
        self.api = HfApi()

        trainer_state = None

        if self.api.repo_exists(repo_id=hf_trainer_config.repo_id):
            logger.info(f"Found existing repo!")
            hf_trainer_state = HFTrainerState.from_pretrained_or_none(hf_trainer_config)
            if hf_trainer_state is not None:
                trainer_state = hf_trainer_state.state
                logger.info("Resume training...")
        elif rank==0:
            repo = self.api.create_repo(repo_id=hf_trainer_config.repo_id, private=True, exist_ok=True)
            logger.info(f"New training. Created repo {repo}")

        self.trainer = trainer
        self.trainer.set_state(trainer_state)
        
        if rank==0:
            self.add_callback(TrainerEvent.ON_NEW_BEST_VAL_LOSS, lambda trainer, model : self.do_on_eval(model))
            
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
    
    def add_callback(self, event: TrainerEvent, callback):
        self.trainer.add_callback(event, callback)
        
    def train(self, hf_model:HfModel):
        assert isinstance(hf_model, HfModel)
        model = hf_model.model
        
        if self.rank == 0:
            self.hf_trainer_config.save_pretrained(push_to_hub=True)
            hf_model.config.save_pretrained(self.config.out_dir, repo_id=self.hf_trainer_config.repo_id, push_to_hub=True)

        super().train(model=model)

        # wait until jobs are complete
        self.executor.shutdown()

    def do_on_eval(self, model:Module):
        if not os.path.exists(self.config.out_dir):
            os.makedirs(self.config.out_dir)

        # cancel previous tasks
        self.executor.shutdown(cancel_futures=True)
        
        # save trainer state
        hf_state = HFTrainerState(self.state, self.hf_trainer_config)
        hf_state.save()
        # save model state
        HfModel.save(model, self.config.out_dir)

        # upload saved files at the background
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        self.executor.submit(hf_state.upload_saved)
        self.executor.submit(HfModel.upload_saved, self.config.out_dir, self.hf_trainer_config.repo_id)