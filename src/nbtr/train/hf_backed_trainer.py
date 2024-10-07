from nbtr.model.hf_trainer_config import HfTrainerConfig
from .trainer import Trainer, TrainerEvent
from nbtr.model.hf_trainer_state import HFTrainerState
from nbtr.model.hf_model import HfModel
import os
from torch.nn import Module
from huggingface_hub import HfApi
from concurrent.futures import ThreadPoolExecutor

class HFBackedTrainer:
    num_workers = 2
    def __init__(self, hf_trainer_config:HfTrainerConfig, trainer, rank:int = 0) -> None:
        assert hf_trainer_config.repo_id is not None, "Repo id cannot be None"

        self.hf_trainer_config = hf_trainer_config
        self.rank = rank
        self.api = HfApi()

        trainer_state = None

        if self.api.repo_exists(repo_id=hf_trainer_config.repo_id):
            print(f"Found existing repo!")
            hf_trainer_state = HFTrainerState.from_pretrained_or_none(hf_trainer_config)
            if hf_trainer_state is not None:
                trainer_state = hf_trainer_state.state
                print("Resume training...")
        elif rank==0:
            repo = self.api.create_repo(repo_id=hf_trainer_config.repo_id, private=True, exist_ok=True)
            print(f"New training. Created repo {repo}")

        self.trainer = trainer
        if trainer_state is not None:
            self.trainer.set_state(trainer_state)
        
        if rank==0:
            self.add_callback(TrainerEvent.ON_NEW_BEST_VAL_LOSS, lambda trainer, model : self.do_on_eval(trainer, model))
            
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
    
    def add_callback(self, event: TrainerEvent, callback):
        self.trainer.add_callback(event, callback)
        
    def train(self, hf_model:HfModel):
        assert isinstance(hf_model, HfModel)
        model = hf_model.model
        
        try:
            self.trainer.train(model=model, raw_model=model)
        except Exception as e:
            print("Error: "+str(e))

        # wait until jobs are complete
        self.executor.shutdown()

    def do_on_eval(self, trainer:Trainer, model:Module):
        if not os.path.exists(trainer.config.out_dir):
            os.makedirs(trainer.config.out_dir)

        # cancel previous tasks
        self.executor.shutdown(cancel_futures=True)
        
        # save trainer state
        hf_state = HFTrainerState(trainer.state, self.hf_trainer_config)
        hf_state.save()
        # save model state
        HfModel.save(model, trainer.config.out_dir)

        # upload saved files at the background
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        self.executor.submit(hf_state.upload_saved)
        self.executor.submit(HfModel.upload_saved, trainer.config.out_dir, self.hf_trainer_config.repo_id)