from .trainer import Trainer, TrainerConfig, TrainingState
import os
from torch.nn import Module
import torch.nn as nn
from huggingface_hub import HfApi
from transformers.utils import cached_file
import torch
from concurrent.futures import ThreadPoolExecutor

model_file_name = 'pytorch_model.bin'
optim_file_name = 'trainer_state.bin'

class HFBackedTrainer(Trainer):
    num_workers = 2
    def __init__(self, config:TrainerConfig) -> None:
        assert config.repo_id is not None, "Repo id cannot be None"

        self.api = HfApi()

        trainer_state = None
        self.model_state = None

        if self.api.repo_exists(repo_id=config.repo_id):
            
            model_safetensors_file = cached_file(config.repo_id, model_file_name, _raise_exceptions_for_missing_entries=False)
            optimization_safetensors_file = cached_file(config.repo_id, optim_file_name, _raise_exceptions_for_missing_entries=False)

            if model_safetensors_file is not None:
                self.model_state = torch.load(model_safetensors_file)
            
            if optimization_safetensors_file is not None:
                trainer_state = torch.load(optimization_safetensors_file)

            print(f"Resume training. Using repo {config.repo_id}")
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
        model_out_path = os.path.join(self.config.out_dir, model_file_name)
        optim_out_path = os.path.join(self.config.out_dir, optim_file_name)

        if self.state.iter_num > 0:
            print(f"saving checkpoint to {self.config.out_dir}")

            if not os.path.exists(self.config.out_dir):
                os.makedirs(self.config.out_dir)

            # cancel previous tasks
            self.executor.shutdown(cancel_futures=True)

            torch.save(model.state_dict(), model_out_path)
            torch.save(self.state, optim_out_path)

            self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
            self.executor.submit(self._upload_file, model_out_path, model_file_name)
            self.executor.submit(self._upload_file, optim_out_path, optim_file_name)