from nbtr.train.trainer import TrainerConfig, Trainer
from nbtr.train.ddp_trainer import DDPTrainer
from nbtr.model.gpt2 import GPTConfig, GPT
import torch.distributed as dist
import torch
from nbtr.model.gpt2 import GPTConfig
from nbtr.train.trainer import TrainerConfig
from dataclasses import replace, asdict
import argparse
import os
from time import time

def main_with_config(data_dir, trainer_config_file, model_config_file, extras:dict):
    trainer_config = TrainerConfig.from_yaml(trainer_config_file)
    
    if data_dir is not None:
        trainer_config = replace(trainer_config, data_dir=data_dir)
    
    if trainer_config.wandb_run_id is None:
        trainer_config = replace(trainer_config, wandb_run_id=str(int(time())))
    
    model_extras = {k[6:]:v for k,v in extras.items() if k.startswith("model.")}
    trainer_extras = {k:v for k,v in extras.items() if not k.startswith("model.")}
    
    if len(trainer_extras)>0:
        trainer_config = replace(trainer_config, **trainer_extras)

    if trainer_config.wandb_run_name is None:
        trainer_config = replace(trainer_config, wandb_run_name=trainer_config.out_dir)

    assert trainer_config.out_dir is not None
    assert trainer_config.data_dir is not None
    assert trainer_config.wandb_run_name is not None
    assert trainer_config.wandb_run_id is not None
    assert trainer_config.wandb_project is not None
    
    gpt_config = GPTConfig.from_yaml(model_config_file)

    if len(model_extras)>0:
        gpt_config = replace(gpt_config, **model_extras)
    
    model = GPT(config=gpt_config)
    
    do_train(trainer_config, gpt_config, model, True)

def do_train(trainer_config:TrainerConfig, model_config:GPTConfig, model:GPT, save_config:bool):
    
    os.makedirs(trainer_config.out_dir,exist_ok=True)

    print("Model config:", model_config._to_dict())
    print("Training config:", trainer_config._to_dict())

    if os.getenv("RANK",-1)==-1:
        train_local(trainer_config, model_config, model, save_config)
    else:
        train_ddp(trainer_config, model_config, model, save_config)

def train_ddp(trainer_config, model_config:GPTConfig, model, save_config:bool):
    dist.init_process_group("nccl")
    
    rank = dist.get_rank()
    rank = rank % torch.cuda.device_count()
    
    assert dist.get_world_size() <= torch.cuda.device_count() , "There are more processes than available cuda devices.!!!"
    
    if rank == 0 and save_config:
        save_configs(trainer_config, model_config)
    
    torch.cuda.set_device(rank)
        
    trainer = Trainer(trainer_config, rank=rank, world_size=dist.get_world_size())
    trainer = DDPTrainer(trainer=trainer, rank=rank)
        
    model.to(rank)

    dist.barrier()
        
    try:
        trainer.train(model=model, raw_model=model)
    except Exception as e:
        print("Training failed with exception:"+str(e))
        
    dist.destroy_process_group()
        
    if rank == 0:
        print(f"Training completed.")

def train_local(trainer_config:TrainerConfig, model_config:GPTConfig, model, save_config:bool):
    assert isinstance(trainer_config, TrainerConfig)
    assert isinstance(model_config, GPTConfig)

    trainer = Trainer(trainer_config)

    if save_config:
        save_configs(trainer_config, model_config)

    model.to(device="cuda")
        
    trainer.train(model=model, raw_model=model)
        
    print(f"Training completed.")

def save_configs(trainer_config:TrainerConfig, gpt_config:GPTConfig):
    assert isinstance(trainer_config, TrainerConfig)
    assert isinstance(gpt_config, GPTConfig)
    
    # save trainer config
    trainer_config.to_yaml(f"{trainer_config.out_dir}/trainer_config.yaml")
    gpt_config.to_yaml(f"{trainer_config.out_dir}/model_config.yaml")

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
    parser.add_argument("--data_dir", type=str, required=False, help="Data directory override.")
    parser.add_argument("--trainer_config_file", "-C", type=str, required=False, default=None, help="Trainer config file")
    parser.add_argument("--model_config_file", "-M", type=str, required=False, default=None, help="Model config file")
    args, extra = parser.parse_known_args()
    
    data_dir = args.data_dir
    trainer_config_file = args.trainer_config_file
    model_config_file = args.model_config_file

    keys = [extra[i][2:] for i in range(0, len(extra),2)]
    values = [extra[i] for i in range(1, len(extra),2)]
    kv = {k: parse(v) for k,v in zip(keys, values)}

    main_with_config(data_dir, trainer_config_file, model_config_file, kv)
