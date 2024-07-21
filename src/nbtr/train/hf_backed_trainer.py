from nbtr.model.hf_trainer_config import HfTrainerConfig
from .trainer import Trainer, TrainerConfig
from nbtr.model.hf_trainer_state import HFTrainerState
from nbtr.model.hf_model import HfModel
import os
from torch.nn import Module
import torch.nn as nn
from huggingface_hub import HfApi
from concurrent.futures import ThreadPoolExecutor

class HFBackedTrainer(Trainer):
    num_workers = 2
    def __init__(self, hf_trainer_config:HfTrainerConfig) -> None:
        assert hf_trainer_config.repo_id is not None, "Repo id cannot be None"

        self.hf_trainer_config = hf_trainer_config

        self.api = HfApi()

        trainer_state = None

        if self.api.repo_exists(repo_id=hf_trainer_config.repo_id):
            print(f"Found existing repo!")
            hf_trainer_state = HFTrainerState.from_pretrained_or_none(hf_trainer_config)
            if hf_trainer_state is not None:
                trainer_state = hf_trainer_state.state
                print("Resume training...")
        else:
            repo = self.api.create_repo(repo_id=hf_trainer_config.repo_id, private=True, exist_ok=True)
            print(f"New training. Created repo {repo}")

        super().__init__(config=hf_trainer_config.trainer_config, state=trainer_state)
        
        self.add_callback("on_new_best_val_loss", lambda trainer, model : self.do_on_eval(model))
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)

    """@staticmethod
    def from_config(config_file):
        import yaml

        with open(config_file) as f:
            try:
                doc = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)
                exit(1)
        
        trainer_config = TrainerConfig(**doc)
        hf_trainer_config = HfTrainerConfig(,trainer_config=trainer_config)
        return HFBackedTrainer(hf_trainer_config=hf_trainer_config)"""
    
    def train(self, hf_model:HfModel):
        assert isinstance(hf_model, HfModel)
        model = hf_model.model
        
        self.hf_trainer_config.save_pretrained(push_to_hub=True)
        hf_model.config.save_pretrained(self.config.out_dir, push_to_hub=True)

        super().train(model=model)

        # wait until jobs are complete
        self.executor.shutdown()

    def _upload_file(self, file_to_upload, target_path):
        self.api.upload_file(path_or_fileobj=file_to_upload, path_in_repo=target_path, repo_id=self.hf_trainer_config.repo_id)
        print(f"Completed {file_to_upload}")

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