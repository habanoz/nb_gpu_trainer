from nbtr.train.trainer import TrainerConfig, Trainer
from nbtr.train.ddp_trainer import DDPTrainer
from nbtr.train.hf_backed_trainer import HFBackedTrainer
from nbtr.model.hf_model import HfModel
from nbtr.model.hf_trainer_config import HfTrainerConfig
from nbtr.model.hf_model_config import HfModelConfig
from nbtr.model.gpt2 import GPTConfig
import torch.distributed as dist
import torch
from dataclasses import replace, asdict
import argparse
import os
from time import time

def load_trainer_config_file(repo_id, data_dir, trainer_config_file, extras):
    trainer_config = TrainerConfig.from_yaml(trainer_config_file)
    
    if data_dir is not None:
        trainer_config = replace(trainer_config, data_dir=data_dir)
    
    if trainer_config.wandb_run_name is None:
        trainer_config = replace(trainer_config, wandb_run_name=repo_id.split("/")[-1])
    
    if trainer_config.wandb_run_id is None:
        trainer_config = replace(trainer_config, wandb_run_id=str(int(time())))
    
    if len(extras)>0:
        trainer_config = replace(trainer_config, **extras)

    assert trainer_config.data_dir is not None
    assert trainer_config.wandb_run_name is not None
    assert trainer_config.wandb_run_id is not None
    assert trainer_config.wandb_project is not None
    
    return trainer_config

def main_with_repo_id(repo_id, hf_training_config, hf_model_config):
    hf_model = HfModel.from_pretrained_or_config(repo_id=repo_id, hf_model_config=hf_model_config, device="cpu")
    
    hf_train(hf_training_config, hf_model.config, hf_model, False)

def main_with_init(repo_id, init_repo_id, data_dir, trainer_config_file, extras:dict):
    trainer_config = load_trainer_config_file(repo_id, data_dir, trainer_config_file, extras)
    
    hf_model_config = HfModelConfig.from_pretrained(init_repo_id)
    hf_model = HfModel.from_pretrained(repo_id=init_repo_id, device="cpu")

    hf_trainer_config = HfTrainerConfig(repo_id=repo_id, trainer_config=trainer_config, init_repo_id=init_repo_id)
    
    hf_train(hf_trainer_config, hf_model_config, hf_model, True)
    
def main_with_config(repo_id, data_dir, trainer_config_file, model_config_file, extras:dict):
    trainer_config = load_trainer_config_file(repo_id, data_dir, trainer_config_file, extras)
    
    gpt_config = GPTConfig.from_yaml(model_config_file)
    hf_model_config = HfModelConfig(gpt_config=gpt_config)
    hf_model = HfModel.from_pretrained_or_config(repo_id=init_repo_id if init_repo_id else repo_id, hf_model_config=hf_model_config, device="cpu")

    hf_trainer_config = HfTrainerConfig(repo_id=repo_id, trainer_config=trainer_config)
    
    hf_train(hf_trainer_config, hf_model_config, hf_model, True)

def hf_train(hf_trainer_config:HfTrainerConfig, hf_model_config:HfModelConfig, hf_model:HfModel, save_config:bool):

    print("Model config:", hf_model_config._to_dict())
    print("Training config:", hf_trainer_config._to_dict())

    if os.getenv("RANK",-1)==-1:
        hf_train_local(hf_trainer_config, hf_model_config, hf_model, save_config)
    else:
        hf_train_ddp(hf_trainer_config, hf_model_config, hf_model, save_config)

def hf_train_ddp(hf_trainer_config, hf_model_config:HfModelConfig, hf_model, save_config:bool):
    dist.init_process_group("nccl")
    
    rank = dist.get_rank()
    rank = rank % torch.cuda.device_count()
    
    assert dist.get_world_size() <= torch.cuda.device_count() , "There are more processes than available cuda devices.!!!"
    
    if rank == 0 and save_config:
        save_configs(hf_trainer_config, hf_model_config)
    
    torch.cuda.set_device(rank)
        
    trainer = Trainer(hf_trainer_config.trainer_config, rank=rank, world_size=dist.get_world_size())
    trainer = DDPTrainer(trainer=trainer, rank=rank)
    trainer = HFBackedTrainer(hf_trainer_config=hf_trainer_config, trainer=trainer, rank=rank)
        
    hf_model.to(rank)

    dist.barrier()
        
    try:
        trainer.train(hf_model=hf_model)
    except Exception as e:
        print("Training failed with exception:"+str(e))
        
    dist.destroy_process_group()
        
    if rank == 0:
        print(f"Training completed.")

def hf_train_local(hf_trainer_config, hf_model_config:HfModelConfig, hf_model, save_config:bool):
    trainer = Trainer(hf_trainer_config.trainer_config)
    trainer = HFBackedTrainer(hf_trainer_config=hf_trainer_config, trainer=trainer)
    
    if save_config:
        save_configs(hf_trainer_config, hf_model_config)

    hf_model.to(device="cuda")
        
    trainer.train(hf_model=hf_model)
        
    print(f"Training completed.")

def get_hf_training_config_from_repo(repo_id):
    try:
        return HfTrainerConfig.from_pretrained(repo_id=repo_id)
    except Exception as e:
        print(f"Error: get_hf_training_config_from_repo {repo_id}: {e=}, {type(e)=}")
        return None

def get_hf_model_config_from_repo(repo_id):
    try:
        return HfModelConfig.from_pretrained(repo_id=repo_id)
    except Exception as e:
        print(f"Error: get_hf_model_config_from_repo {repo_id}: {e=}, {type(e)=}")
        return None

def save_configs(hf_trainer_config:HfTrainerConfig, hf_model_config:HfModelConfig):
    assert isinstance(hf_trainer_config, HfTrainerConfig)
    assert isinstance(hf_model_config, HfModelConfig)
    
    # save trainer config
    hf_trainer_config.save_pretrained(push_to_hub=True)
    
    # save model config
    out_dir = hf_trainer_config.trainer_config.out_dir
    hf_model_config.save_pretrained(out_dir, repo_id=hf_trainer_config.repo_id, push_to_hub=True)

    print("Configurations are saved.")

def parse(val:str):
    assert val is not None, "Value cannot be None"
    
    if val.lower in ("true", "false"):
        return bool(val)
    
    try:
        return float(val)
    except:
        pass
    
    try:
        return int(val)
    except:
        pass
    
    return val

    
if __name__ == '__main__':
    print("RANK", os.getenv("RANK", -1))

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True, help="Model repository id")
    parser.add_argument("--init_repo_id", type=str, required=False, help="Initial model repository id")
    parser.add_argument("--data_dir", type=str, required=False, help="Data directory override.")
    parser.add_argument("--trainer_config_file", "-C", type=str, required=False, default=None, help="Trainer config file")
    parser.add_argument("--model_config_file", "-M", type=str, required=False, default=None, help="Model config file")
    args, extra = parser.parse_known_args()
    
    repo_id = args.repo_id
    init_repo_id = args.init_repo_id
    data_dir = args.data_dir
    trainer_config_file = args.trainer_config_file
    model_config_file = args.model_config_file
    
    assert init_repo_id is None or model_config_file is None, "Init repo and model config file cannot be specified together."

    keys = [extra[i][2:] for i in range(0, len(extra),2)]
    values = [extra[i] for i in range(1, len(extra),2)]
    kv = {k: parse(v) for k,v in zip(keys, values)}
    
    hf_training_config = get_hf_training_config_from_repo(repo_id)
    hf_model_config = get_hf_model_config_from_repo(repo_id)

    try:
        if (trainer_config_file is None or (model_config_file is None and init_repo_id is None ))  and (hf_training_config is None or hf_model_config is None):
            print("This is a brand-new training job. Provide training configuration file and model configuration file!!!")  
        elif hf_training_config is not None:
            if (trainer_config_file is not None or model_config_file is not None):
                print("Configuration is fetched from the repo. Configuration files are ignored.!!!")
            assert hf_model_config is not None, "Model config not found in the repo. Create a new repo!!! "
            main_with_repo_id(repo_id, hf_training_config, hf_model_config)
        else:
            if init_repo_id:
                main_with_init(repo_id, init_repo_id, data_dir, trainer_config_file, kv)
            else:
                main_with_config(repo_id, data_dir, trainer_config_file, model_config_file, kv)
    except Exception as e:
        print("Training failed", e)
