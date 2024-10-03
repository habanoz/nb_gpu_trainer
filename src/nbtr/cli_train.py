from nbtr.train.trainer import TrainerConfig, Trainer
from nbtr.train.ddp_trainer import DDPTrainer
from nbtr.train.hf_backed_trainer import HFBackedTrainer
from nbtr.model.hf_model import HfModel
from nbtr.model.hf_trainer_config import HfTrainerConfig
from nbtr.model.hf_model_config import HfModelConfig
from nbtr.model.gpt2 import GPTConfig
import torch.distributed as dist
import torch
from dataclasses import replace
import argparse
import os
import logging

def setup_logging(rank:int=0):
    os.makedirs("logs", exist_ok=True)
    
    handlers = [logging.FileHandler(f"logs/trainer-{rank}.log"), stream_handler]
    
    if rank == 0:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        handlers.append(stream_handler)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=handlers)
    
def main_with_repo_id(repo_id):
    # prepare model
    hf_model = HfModel.from_pretrained(repo_id)

    # train
    hf_trainer_config = HfTrainerConfig.from_pretrained(repo_id=repo_id)
    hf_train(hf_trainer_config, hf_model)


def main_with_config(repo_id, data_dir, trainer_config_file, model_config_file, extras:dict):
    trainer_config = TrainerConfig.from_yaml(trainer_config_file)
    
    if data_dir is not None:
        trainer_config = replace(trainer_config, data_dir=data_dir)
    
    if len(extras)>0:
        trainer_config = replace(trainer_config, **extras)

    assert trainer_config.data_dir is not None

    # prepare model
    gpt_config = GPTConfig.from_yaml(model_config_file)
    hf_model_config = HfModelConfig(gpt_config=gpt_config)
    hf_model = HfModel.from_pretrained_or_config(repo_id=repo_id, hf_model_config=hf_model_config)

    # train
    hf_trainer_config = HfTrainerConfig(repo_id=repo_id, trainer_config=trainer_config)
    hf_train(hf_trainer_config, hf_model)

def hf_train(hf_trainer_config:HfTrainerConfig, hf_model):
    trainer = Trainer(hf_trainer_config.trainer_config)
    
    if os.getenv("RANK",-1)==-1:
        setup_logging()
        
        trainer = HFBackedTrainer(hf_trainer_config=hf_trainer_config, trainer=trainer)
        trainer.train(hf_model=hf_model)
    else:
        ## DDP training
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        
        setup_logging(rank=rank)
        torch.cuda.set_device(rank)
        
        trainer = DDPTrainer(trainer=trainer)
        trainer = HFBackedTrainer(hf_trainer_config=hf_trainer_config, rank=rank, trainer=trainer)
        trainer.train(hf_model=hf_model)
        
        dist.destroy_process_group()
        
    print("Training completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True, help="Model repository id")
    parser.add_argument("--data_dir", type=str, required=False, help="Data directory override.")
    parser.add_argument("--trainer_config_file", "-C", type=str, required=False, default=None, help="Trainer config file")
    parser.add_argument("--model_config_file", "-M", type=str, required=False, default=None, help="Model config file")
    args, extra = parser.parse_known_args()
    
    repo_id = args.repo_id
    data_dir = args.data_dir
    trainer_config_file = args.trainer_config_file
    model_config_file = args.model_config_file

    keys = [extra[i][2:] for i in range(0, len(extra),2)]
    values = [extra[i] for i in range(1, len(extra),2)]
    kv = {k:v for k,v in zip(keys, values)}
    
    if trainer_config_file is not None:
        assert model_config_file is not None
        main_with_config(repo_id, data_dir, trainer_config_file, model_config_file, kv)
    else:
        assert len(kv)==0, "Extra values are not supported when resuming from a repo!"
        main_with_repo_id(repo_id)
