from .trainer import Trainer, TrainerConfig, TrainingState
from nbtr.model.hf_trainer_state import HFTrainerState
from nbtr.model.hf_model import HfModel
import os
from torch.nn import Module
import torch.nn as nn
from huggingface_hub import HfApi
from concurrent.futures import ThreadPoolExecutor

class HFBackedTrainer(Trainer):
    num_workers = 2
    def __init__(self, config:TrainerConfig) -> None:
        assert config.repo_id is not None, "Repo id cannot be None"

        self.api = HfApi()

        trainer_state = None
        self.model_state = None

        if self.api.repo_exists(repo_id=config.repo_id):
            print(f"Found existing repo!")
            hf_trainer_state = HFTrainerState.from_pretrained_or_none(config)
            if hf_trainer_state is not None:
                trainer_state = hf_trainer_state.state
                print("Resume training...")
        else:
            repo = self.api.create_repo(repo_id=config.repo_id, private=True, exist_ok=True)
            print(f"New training. Created repo {repo}")

        super().__init__(config=config, state=trainer_state)
        
        self.add_callback("on_new_best_val_loss", lambda trainer, model : self.do_on_eval(model))
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)

    @staticmethod
    def from_config(config_file):
        import yaml

        with open(config_file) as f:
            try:
                doc = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)
                exit(1)
        
        config = TrainerConfig(**doc)
        return HFBackedTrainer(config)
    
    def train(self, model:nn.Module):
        if self.model_state is not None:
            print("Loading model state dict!")
            model.load_state_dict(self.model_state)
            print("Loaded model state dict!")

        super().train(model=model)

        # wait until jobs are complete
        self.executor.shutdown()

    def _upload_file(self, file_to_upload, target_path):
        self.api.upload_file(path_or_fileobj=file_to_upload, path_in_repo=target_path, repo_id=self.config.repo_id)
        print(f"Completed {file_to_upload}")

    def do_on_eval(self, model:Module):
        print(f"Callback iter: {self.state.iter_num}, best val loss: {self.state.best_val_loss}")

        if self.state.iter_num > 0:
            print(f"saving checkpoint to {self.config.out_dir}")

            if not os.path.exists(self.config.out_dir):
                os.makedirs(self.config.out_dir)

            # cancel previous tasks
            self.executor.shutdown(cancel_futures=True)
            
            # save trainer state
            hf_state = HFTrainerState(self.state, self.config)
            hf_state.save()
            # save model state
            HfModel.save(model, self.config.out_dir)

            # upload saved files at the background
            self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
            self.executor.submit(hf_state.upload_saved)
            self.executor.submit(HfModel.upload_saved, self.config.out_dir, self.config.repo_id)