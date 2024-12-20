from dataclasses import dataclass, asdict, field
import torch
import torch.nn as nn
import numpy as np
import os
import math
import time
import datetime
from collections import defaultdict
from typing import Any, Dict
from ..utils.mfu import estimate_mfu
from enum import Enum
from ..model.trainer_model import TrainerModel
from nbtr.utils.wb_logger import WandBLogger
from nbtr.data.ddl_gdown import DistributedDataLoader as DDL
@dataclass
class TrainerConfig:
    seed: int = 145
    
    seq_length: int = 1024
    gradient_accumulation_steps:int = 4
    batch_size: int = 8
    data_dir: str = None
    max_iters: int = 5000
    warmup_iters: int = 100
    grad_norm_clip: float = 1.0
    out_dir:str = None
    dtype: str = 'bfloat16'
    compile: bool=False
    gc: bool=False # gradient checkpointing
        
    # learning rate
    learning_rate: float = 1e-4
    decay_lr: bool = True
    lr_decay_iters: int = field(default=None) # if None use max_iters
    min_lr: float = 1e-6
    
    # optimizer
    weight_decay: float = None
    beta1: float = 0.9
    beta2: float= 0.95
    
    # logging
    log_interval: int = 10
    eval_interval: int = 250 # must be multiple of log interval
    eval_iters: int = 200
    promised_flops:float=65e12 # Tesla T4 on fp16
    ## wandb logging
    wandb_log: bool = False
    wandb_project: str = None
    wandb_run_name: str = None
    wandb_run_id: str = None
    
    def __post_init__(self):
        self.lr_decay_iters = self.lr_decay_iters if self.lr_decay_iters else self.max_iters
    
    @staticmethod
    def from_yaml(config_file:str):
        import yaml

        with open(config_file) as f:
            doc = yaml.safe_load(f)
        
        return TrainerConfig(**doc)        
    
@dataclass
class TrainingState:
    iter_num: int = 0
    best_val_loss: float = 1e+9
    optim_state: Any = field(default=None)

@dataclass
class EvalResult:
    loss: float
    perplexity: float

class TrainerEvent(Enum):
    ON_NEW_BEST_VAL_LOSS=0,
    ON_LAST_MICRO_BATCH=1,

TRAIN_DATA_FILES={
'c4tr_train_000000.bin.gz': '1h9auLCnbWjlIXzb2iu21A6cnGBmvnitZ',
 'c4tr_train_000001.bin.gz': '1pw_XrcupgESeIt6ukTx6ncYg35-gmYqh',
 'c4tr_train_000002.bin.gz': '1tScXSfxcnOeBJiOXGG3dCW7h9IxvREWP',
 'c4tr_train_000003.bin.gz': '1Csj1oPXkwPAlkpWctvMJ_qvMNN2h0ZTm',
 'c4tr_train_000004.bin.gz': '1m-qpvj9iXRnviqZuTeS_QfAo0rnNw68C',
 'c4tr_train_000005.bin.gz': '1GPBQE1a4PX4QGV4ZSNaPlqX8OFkIN6kg',
 'c4tr_train_000006.bin.gz': '1h0ZxTfXxacKmlhVldCL0Yn2G2vjxGfRH',
 'c4tr_train_000007.bin.gz': '1sfY9EXwugf1t1IW_vr8xDxMeUec8W1tm',
 'c4tr_train_000008.bin.gz': '1to-uDOpAewyKoxBK4Xmbm42CgMku1ozg',
 'c4tr_train_000009.bin.gz': '156GiUTG37QGx4svUs0nsuf7c12DUNVNQ',
 'c4tr_train_000010.bin.gz': '1YLTdQaTkB1ewpMMOtfqegZRf_7ka4qY0',
 'c4tr_train_000011.bin.gz': '14IdmBDSPV73V7_mQWgWbt6vOTM3z6V1C',
 'c4tr_train_000012.bin.gz': '1kPHD_3WBFmlD-sJzZ9qD-BObBZYantye',
 'c4tr_train_000013.bin.gz': '18N7vtOpuXNd2Fcjmo0_9mKtB8P3a8owN',
 'c4tr_train_000014.bin.gz': '1e-RPa0d19OECYE_nFZF97UEAPDF-1wBV',
 'c4tr_train_000015.bin.gz': '17IeMdEA4q--fGc1yLbaFKca_HNQxsnJ7',
 'c4tr_train_000016.bin.gz': '1uOWTEDEtlMZkLykMlXS_gjFMlaBrsbWI',
 'c4tr_train_000017.bin.gz': '1uA00FZApZOCGdlzv-nF7pZ4h5_9lucP1',
 'c4tr_train_000018.bin.gz': '1ML6wi1UjPK4-6LPezRPwpRwwimyoQ84z',
 'c4tr_train_000019.bin.gz': '1JzDbacBmPIyCEupv8VOn6tcVBjVP_VZu',
 'c4tr_train_000020.bin.gz': '1igbI797yOAUuwOPpFovVP2b00bHyyEDd',
 'c4tr_train_000021.bin.gz': '1PhdpPQRk_m1BewVQA3jZK9XadN4Co_jX',
 'c4tr_train_000022.bin.gz': '1IM_QHbiyarhUcWUGr69dpAU9CSP2IZ9x',
 'c4tr_train_000023.bin.gz': '1j6jrHO0mrzjeSCWBdUpjwW281R_pmI-W',
 'c4tr_train_000024.bin.gz': '1xzACoR8zRsQj_Q1CWzJW3oBP9ae92CRY',
 'c4tr_train_000025.bin.gz': '19g8sdIsOXoj26B4cJ9YFoGQ3g7bYDkfs',
 'c4tr_train_000026.bin.gz': '1I6TYQOAeVDzZNIUI5-sJnld91_d1n0jT',
 'c4tr_train_000027.bin.gz': '1OAV0bE9uL6cz111O1CHmeUBPpVZHOUcb',
 'c4tr_train_000028.bin.gz': '1Eai-LD_qFtu8V3G4_Wldbkr7tbA3DFgg',
 'c4tr_train_000029.bin.gz': '1WyBk0zKzQog8CFc-q3lXKOAsVZW427_L',
 'c4tr_train_000030.bin.gz': '17u5KgbfTzSQCN9iPJvGkNiPz1f-1y1yq',
 'c4tr_train_000031.bin.gz': '1Lce1IDNjqpcm3CIBdf9HO7bKuatulI2v',
 'c4tr_train_000032.bin.gz': '11e4c9PuuV-2oy_jTMxlf-yC2qhpPENWe',
 'c4tr_train_000033.bin.gz': '13hfUGHkRgFBp0AOiPKMd_Ef8ulEImPB9',
 'c4tr_train_000034.bin.gz': '1n-dOyIkPI6rS5vmU-mS6ztqRfi2ZK81j',
 'c4tr_train_000035.bin.gz': '1Hitzs0pQcTgMrIhZSm7KCjxYOY-pkF_U',
 'c4tr_train_000036.bin.gz': '1K3HIQkVBIfi4lRakvMx0rOXFtvmGtOd6',
 'c4tr_train_000037.bin.gz': '1AfpTxVfY-1A3JgELgQ5-omaj7IOSaMwQ',
 'c4tr_train_000038.bin.gz': '1AHMls-To0_IMRQ-omN2cCfatL5nKs-dc',
 'c4tr_train_000039.bin.gz': '1zJBmJVL4pByf4n8_5QyPdhaOAmGnQF7H',
 'c4tr_train_000040.bin.gz': '1OL_SYKfhBSB_d2xol9yQ2HmwDPz6YjiM',
 'c4tr_train_000041.bin.gz': '1H6gVEMNCHX6rYwSnRIEAjsdnuigU2WNz',
 'c4tr_train_000042.bin.gz': '1PPGF-Gn72Ec4W0SZ4mi8pEK9KnLuUN-7',
 'c4tr_train_000043.bin.gz': '1pgmXQnHbNTDvNDpgy9NdNF-R5amd8HhM',
 'c4tr_train_000044.bin.gz': '1oVrZwdEsga7PlWRMQwdVypBr8u8BKPwo',
 'c4tr_train_000045.bin.gz': '1qOo_EDigAa_jqNzHMXBLq92nuPKN_kzc',
 'c4tr_train_000046.bin.gz': '1Hfkusx-_9tgcmXBZ743BfqEi8EmRrXox',
 'c4tr_train_000047.bin.gz': '1i52ZLsxLvYH5zuz0yy7kf6Zu_LGzKrJ7',
 'c4tr_train_000048.bin.gz': '1fFRyRRWShCRh5xIOCUvwr09y-fl41mjP',
 'c4tr_train_000049.bin.gz': '1-w2djG74J0SOLZBmHEjXkcg4ItGUavpU',
 'c4tr_train_000050.bin.gz': '1sLzLfA2BlfP5UpHmE61KQQWW49i_FIJ8',
 'c4tr_train_000051.bin.gz': '1Z1KZjgTQF9mFYVvj7RktP15Bhs5dzOBx',
 'c4tr_train_000052.bin.gz': '1OQOHuRZeoFXkfgmUM-JnJNLzOPPFUmR0',
 'c4tr_train_000053.bin.gz': '1CD0ALx4dy4Kn6E2EjoBdj9MxGhIDbz5d',
 'c4tr_train_000054.bin.gz': '10dxzYbj2icI12pSo9a_PtsqJOqsugP3i',
 'c4tr_train_000055.bin.gz': '11k3kHpTnnNeQ5j0-Zkf7Mep_dVfydCqA',
 'c4tr_train_000056.bin.gz': '1R1qqqQIGtxH6N4skRFzq6LaiGayK3lXf',
 'c4tr_train_000057.bin.gz': '16Jq-cOLRv7g3TBphuLlVkRR1EZVV38FT',
 'c4tr_train_000058.bin.gz': '11b4wUs21Apyb7ILLL_b5hcbZB5v9ZyuG',
 'c4tr_train_000059.bin.gz': '1q-TRUdI4iTmtZ-R1rniccSiUBvdYY67p',
 'c4tr_train_000060.bin.gz': '1MTLD-RrgrYG1KvbgXzwAU-MwyqHCyqkZ',
 'c4tr_train_000061.bin.gz': '1rnr462u6vkPoj03hH3alqlvRCKFp3RYZ',
 'c4tr_train_000062.bin.gz': '145vpqJ7F4_KnTE3mgYA0iLmadU47oLmi',
 'c4tr_train_000063.bin.gz': '1dw8ubKVCTfWm1gg8YYWc6ToirClaqGBS',
 'c4tr_train_000064.bin.gz': '16cwhqt3YoGwV_EG1ot6YnX6q96CJGCIa',
 'c4tr_train_000065.bin.gz': '1LHB1LnwMyhVankH7UBeeMc6oP4hPbn9M',
 'c4tr_train_000066.bin.gz': '1igZbuTHJaVNkNVWLNtcVyMrd7vyfPqWy',
 'c4tr_train_000067.bin.gz': '1898-p3Wu_ksu5PeLTcCDylhpcCIehAmJ',
 'c4tr_train_000068.bin.gz': '17LeIg8GDKrNngSkpucmYQtkDyXiSBoxz',
 'c4tr_train_000069.bin.gz': '1eFcXKzjgncWsSYK3RoJQlwtXdSDZM93L',
 'c4tr_train_000070.bin.gz': '18xxhdYq8h1DPQtInfjonGdp7DVRD1vXF',
 'c4tr_train_000071.bin.gz': '1ttyaPhDIgp5ickJ3wiY7obXKiZ6MtQn0',
 'c4tr_train_000072.bin.gz': '1CK9x4G_PrYcWdx4pP8uxAWCti9XAUuw1',
 'c4tr_train_000073.bin.gz': '1N5kucm1JxCktLIklkg3BuaIRIxRzCrwm',
 'c4tr_train_000074.bin.gz': '198DU_B6-kOX7S1lwM55Pp1z-ElE7XaCa',
 'c4tr_train_000075.bin.gz': '1LYf3dMH9paiKOzJ7Xix1uq_CBDcWSljy',
 'c4tr_train_000076.bin.gz': '1lOtDdEnc7NJ9acp84tPi0VcFw6YS_Tzs',
 'c4tr_train_000077.bin.gz': '19I-ipUQcnoGKx2u4NC0e3gXStqhABHST',
 'c4tr_train_000078.bin.gz': '1wB3I_Rd7IXhhMYZ2PRIk36c0ys0pzQcm',
 'c4tr_train_000079.bin.gz': '1d2j5zDx0A3B-m_AEvGVLOrCrnlPf9b-J',
 'c4tr_train_000080.bin.gz': '1LQK2eV7UldmHuUC7FGcxdaAW-UkU_zHt',
 'c4tr_train_000081.bin.gz': '11QtoSzbD6b92sYrgwdv7YsBYSNltzI7L',
 'c4tr_train_000082.bin.gz': '1vXXGi3RCPbDCKGcDDzk9bXsujPDHBDgl',
 'c4tr_train_000083.bin.gz': '1o2PSuc_Z03Hlp12RAyd6OLt6zwmaAn-k',
 'c4tr_train_000084.bin.gz': '1AvZAgk2O9V6u3iCizxLHqxp-wICfQrR0',
 'c4tr_train_000085.bin.gz': '1upIlG4nluY8DrCGA72O6nUUo69ApWU1j',
 'c4tr_train_000086.bin.gz': '1FwUiPWi-Hb9TF73eG5m3j9jZAAkrjB5m',
 'c4tr_train_000087.bin.gz': '132Fl8lI-2uqKSJ5V7RqUXWgFNgXU52ev',
 'c4tr_train_000088.bin.gz': '1fe6vD8-kSp62iQVdhgeZq_G8LdHNvQPv',
 'c4tr_train_000089.bin.gz': '1_hszeRr5As61TvTdE7ul5cyxG_W16gjy',
 'c4tr_train_000090.bin.gz': '1rBYjQSyqXgByHhrAbkyHdCZMjdv5T2A3',
 'c4tr_train_000091.bin.gz': '1rrRYfAwLA9LEfM745t_l9Law4Zsh9ITR',
 'c4tr_train_000092.bin.gz': '1JBmCokAaXZpl3MqJ3iFjvuamVgXNezKs',
 'c4tr_train_000093.bin.gz': '1dXfYs3kyEx7EYH21S4Td5OZN8AK-U-PM',
 'c4tr_train_000094.bin.gz': '1HdvR2K_EGS76IMje25N-s-Omr3rtCIz9',
 'c4tr_train_000095.bin.gz': '1DdGWt-W6Iq-HuWlUS0yBRUSNWzhrhvhQ',
 'c4tr_train_000096.bin.gz': '1n1K1QFupM3gEZKg5v4jEtp5yIWfYSfhT',
 'c4tr_train_000097.bin.gz': '1gTsIudhK6u5glJ_1ylFqDoghd-9wEwnm',
 'c4tr_train_000098.bin.gz': '12uUirreFNIDxK_N5yPnC2DdnbWymsI2k',
 'c4tr_train_000099.bin.gz': '1eZ1eW_SkP0DsTceF-KBuuWdOMpoICD9U',
 'c4tr_train_000100.bin.gz': '14jPglMPF6b-1MrTIVaQxeoNavq0T4Bq6'
}

TRAIN_VAL_DATA_FILES = {
 'c4tr_train_000000.bin.gz': '1h9auLCnbWjlIXzb2iu21A6cnGBmvnitZ',
}
VAL_DATA_FILES={
  'c4tr_validation_000000.bin.gz': '1vckxf0pCGgA4lSDZAGyDIlEHsCHLsG9u',
}
NEWS_TR_VAL_DATA_FILES={
 'z_newstr_val_000000.bin': '18x7mHp91TXiEqScqoyYanj0wcS37OoFv'
}
class Trainer:
    def __init__(self, config:TrainerConfig, rank:int=0, world_size:int=1, state:TrainingState=None) -> None:
        self.config = config
        
        self.rank = rank
        self.world_size = world_size
        self.device = "cuda" if world_size == 1 else f"cuda:{rank}"
        self.batch_rng = torch.Generator()
        
        self.state = state if state is not None else TrainingState()
        assert self.state.iter_num < config.max_iters

        self.callbacks = defaultdict(list)

        dtype = torch.float16 if self.config.dtype=='float16' else torch.bfloat16

        assert torch.cuda.is_available(), "Cuda is not available. This training script requires an NVIDIA GPU!"
        assert dtype!=torch.bfloat16 or torch.cuda.is_bf16_supported(), "Bfloat data type is selected but it is not supported! Replace it with float16 data type."
        assert self.config.eval_interval % self.config.log_interval == 0, "Eval interval must be a multiple of log interval!"

        self.ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
        
        ## internal state
        self.skip_first_new_best_val_loss = True
        
        # train data loader
        # self.train_ddl = DDL(f"{self.config.data_dir}/*train_*.bin", self.config.batch_size, self.config.seq_length, self.rank, self.world_size)
        self.train_ddl = DDL(TRAIN_DATA_FILES, f"{self.config.data_dir}/train/", self.config.batch_size, self.config.seq_length, self.rank, self.rank, self.world_size)
        
        if self.rank==0:
            # validation data loaders
            #val_train_ddl = DDL(f"{self.config.data_dir}/train_000000.bin", self.config.batch_size, self.config.seq_length, self.rank, self.world_size)
            val_train_ddl = DDL(TRAIN_VAL_DATA_FILES, f"{self.config.data_dir}/train/", self.config.batch_size, self.config.seq_length, self.rank, self.rank, 1)
            # val_ddl = DDL(f"{self.config.data_dir}/val_000000.bin", self.config.batch_size, self.config.seq_length, self.rank, self.world_size)
            val_ddl = DDL(VAL_DATA_FILES, f"{self.config.data_dir}/val/", self.config.batch_size, self.config.seq_length, self.rank, self.rank, 1)
            # newstr_val_ddl = DDL(f"{self.config.data_dir}/z_newstr_val_000000.bin", self.config.batch_size, self.config.seq_length, self.rank, self.world_size)
            newstr_val_ddl = DDL(NEWS_TR_VAL_DATA_FILES, f"{self.config.data_dir}/newstr_val/", self.config.batch_size, self.config.seq_length, self.rank, self.rank, 1)
            
            self.val_loaders_dict = {"train":val_train_ddl, "val":val_ddl, "news_tr_val": newstr_val_ddl}
        else: self.val_loaders_dict = {}
    
    def set_state(self, state: TrainingState):
        assert state is not None, "state cannot be None"
        self.state =  state
        print(f"Restoring trainer state best_val_loss={state.best_val_loss}, iter_num={state.iter_num}")
        
    def add_callback(self, onevent: TrainerEvent, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: TrainerEvent, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: TrainerEvent, model):
        for callback in self.callbacks.get(onevent, []):
            try:
                callback(self, model)
            except Exception as e:
                print(f"Callback execution for event {onevent} failed: {e}")

    def on_new_best_val_loss(self, model:nn.Module):
        if not self.skip_first_new_best_val_loss:
            self.trigger_callbacks(TrainerEvent.ON_NEW_BEST_VAL_LOSS, model)
        else:
            self.skip_first_new_best_val_loss = False

    def update_lr(self, it, optimizer):
        if not self.config.decay_lr:
            return self.config.learning_rate
        
        lr = self.get_lr(it)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_lr(self, it):
        return self.get_cosine_lr(it)
    
    # learning rate decay scheduler (cosine with warmup)
    def get_cosine_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.config.warmup_iters:
            return self.config.learning_rate * it / self.config.warmup_iters
        
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.config.lr_decay_iters:
            return self.config.min_lr
        
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.config.warmup_iters) / (self.config.lr_decay_iters - self.config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)
    
    def get_wsd_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.config.warmup_iters:
            return self.config.learning_rate * it / self.config.warmup_iters
        
        N = self.config.max_iters
        warmdown_ratio = 0.2
        N_decay = N*warmdown_ratio
        N_before_decay = N-N_decay
                
        # 2) stable
        if it <= N_before_decay:
            return self.config.learning_rate
        
        # 3) warmdown
        
        decay_lr = self.config.learning_rate * (1-np.sqrt( (it-N_before_decay) / N_decay))
        assert decay_lr>=0,"Negative not valid!"
        return decay_lr

    def _configure_optimizers(self, model):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in model.named_parameters()}
        
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        betas = (self.config.beta1, self.config.beta2)
        optimizer = torch.optim.AdamW(optim_groups, lr=self.config.learning_rate, betas=betas, fused=True)
        
        if self.state.optim_state is not None:
            optimizer.load_state_dict(self.state.optim_state)
        
        return optimizer
    
    def _init_logging(self):
        if self.config.wandb_log and self.rank==0:
            import wandb
            # wandb.require("core")
            wandb.init(project=self.config.wandb_project, name=self.config.wandb_run_name, id=self.config.wandb_run_id, resume="allow", config=asdict(self.config))
    
    @torch.inference_mode()
    def evaluate(self, model:nn.Module)->Dict[str,EvalResult]:
        
        out = {}
        model.eval()
        for split in self.val_loaders_dict.keys():
            data_loader = self.val_loaders_dict[split]
            data_loader.reset()
            
            losses = torch.zeros(self.config.eval_iters)
            
            for k in range(self.config.eval_iters):
                X, Y = data_loader.next_batch()
                X, Y = X.to(self.device), Y.to(self.device)
                
                with self.ctx:
                    _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = EvalResult(loss=losses.mean().item(), perplexity=losses.exp().mean().item())
        model.train()
        return out

    def train(self, model, raw_model, compile:bool=None):
        
        # assert model.device == DEVICE, f"Only CUDA device is supported for training. Model is in :{model.device}"
        assert self.config.seq_length == raw_model.config.seq_length, f"Sequence length for model and trainer is not equal {self.config.seq_length}!={raw_model.config.seq_length}"

        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

        #if (compile is None and self.config.compile) or compile:
        print("compiling the model...")
        model = torch.compile(model)
        print("compiling the model done!")
    
        scaler = torch.amp.GradScaler(self.device, enabled=(self.config.dtype == 'float16'))
        optimizer = self._configure_optimizers(model)

        # start training
        model.train()
        
        if self.config.gc:
            model.gradient_checkpointing=True
        
        with self.init_logger(raw_model) as wb_run:
            start_iter = 0 if self.state.iter_num == 0 else self.state.iter_num+1
            running_fwd_bwd_tokens_per_sec = 0
            running_iter_time = 0
            mfu = 0
            t0 = 0
            
            # if resuming a run, move data loader position to correct place
            if start_iter > 0:
                self.train_ddl.replay_next_batch(start_iter, self.config.gradient_accumulation_steps)
            
            X, Y = self.train_ddl.next_batch()
            X, Y = X.to(self.device), Y.to(self.device)
            
            # start_iter is to accommodate for resuming training
            # If training is resuming, random state is reshuffled to avoid using same samples for training...
            torch.manual_seed(self.config.seed+self.rank+start_iter)
            self.batch_rng.manual_seed(self.config.seed+self.rank+start_iter)
            
            if self.state.iter_num > 0:
                self.skip_first_new_best_val_loss = False
            elif self.rank == 0:
                # do not evaluate before-hand if resuming...
                # self.do_eval(raw_model, optimizer, running_fwd_bwd_tokens_per_sec, running_iter_time, start_iter, self.get_lr(0), wb_run, 0.0)
                pass
            
            for it in range(start_iter, self.config.max_iters+1):
                
                # determine current lr rate and update params if needed
                lr = self.update_lr(it, optimizer)
                
                # forward the model
                if t0 == 0:
                    t0 = time.time()
                for micro_batch in range(self.config.gradient_accumulation_steps):
                    if self.world_size>1 and micro_batch == self.config.gradient_accumulation_steps - 1:
                        self.trigger_callbacks(TrainerEvent.ON_LAST_MICRO_BATCH, model)
                    with self.ctx:
                        _, loss = model(X, Y)
                        loss = loss / self.config.gradient_accumulation_steps
                        
                    # immediately async prefetch next batch while model is doing the forward pass on the GPU
                    X, Y = self.train_ddl.next_batch()
                    X, Y = X.to(self.device), Y.to(self.device)

                    scaler.scale(loss).backward()
                
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_norm_clip)
                
                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad(set_to_none=True)
                
                if self.rank == 0 and it % self.config.log_interval == 0:
                    loss_sum = loss.item() * self.config.gradient_accumulation_steps # GPU/CPU sync point
                    
                    dt = time.time() - t0
                    t0 = 0
                                
                    iters_since_last_log = self.config.log_interval if it > 0 else 1
                    fwd_bwd_tokens = self.config.batch_size * self.config.seq_length * self.config.gradient_accumulation_steps * iters_since_last_log * self.world_size
                    
                    fwd_bwd_tokens_per_sec = fwd_bwd_tokens / dt
                    iter_time = dt / iters_since_last_log
                    
                    if it > 0:
                        if running_fwd_bwd_tokens_per_sec==0:
                            running_fwd_bwd_tokens_per_sec = fwd_bwd_tokens_per_sec
                            running_iter_time = iter_time
                        else:
                            running_fwd_bwd_tokens_per_sec = 0.9*running_fwd_bwd_tokens_per_sec + 0.1*fwd_bwd_tokens_per_sec
                            running_iter_time = 0.9* running_iter_time + 0.1 * iter_time
                        
                        mfu = estimate_mfu(model=raw_model,fwdbwd_per_iter=self.config.batch_size * self.config.gradient_accumulation_steps, flops_promised=self.config.promised_flops,dt=iter_time)
                        
                    print(f"iter {it}: loss {loss_sum:.4f}, iter_time {iter_time*1000:.2f}ms, run_iter_time {running_iter_time*1000:.2f}ms, fb_toks/sec {fwd_bwd_tokens_per_sec:.2f}, run_fb_toks/sec {running_fwd_bwd_tokens_per_sec:.2f}, mfu {mfu:.3f}")

                    if it>0 and it % self.config.eval_interval == 0:
                        self.do_eval(raw_model, optimizer, running_fwd_bwd_tokens_per_sec, running_iter_time, it, lr, wb_run, mfu)

    def init_logger(self, model):
        return WandBLogger(enabled=(self.config.wandb_log and self.rank==0), project=self.config.wandb_project, name=self.config.wandb_run_name, id=self.config.wandb_run_id, config=asdict(self.config)|asdict(model.config))

    def do_eval(self, model, optimizer, running_fwd_bwd_tokens_per_sec, time_per_iter, it, lr, wb_run, mfu):
        if self.rank != 0:
            return
        
        t0 = time.time()
        eval_out = self.evaluate(model)
                
        wb_run.log({
                    "iter": it,
                    "train/loss": eval_out['train'].loss,
                    "val/loss": eval_out['val'].loss,
                    "val/perplexity": eval_out['val'].perplexity,
                    "newstr_val/loss": eval_out['news_tr_val'].loss,
                    "lr": lr,
                    "tokens/sec": running_fwd_bwd_tokens_per_sec,
                    "mfu": mfu,
                })
            
        
        new_val_loss = eval_out['val'].loss
        if  new_val_loss < self.state.best_val_loss:
            self.state = TrainingState(iter_num=it, best_val_loss=new_val_loss, optim_state=optimizer.state_dict())
            
            self.on_new_best_val_loss(model)
                
        t1 = time.time()
        dt = t1 - t0

        if running_fwd_bwd_tokens_per_sec> 0:
            # remaining training iterations
            rem_iters = self.config.max_iters - it
            rem_fwd_bwd_time = rem_iters * time_per_iter
            # remaining evaluation iterations
            rem_evals = rem_iters // self.config.eval_interval
            rem_evals_time = dt * rem_evals
            # remaining time
            eta = rem_fwd_bwd_time + rem_evals_time 
        else:
            eta = 0.0

        print(f"Eval iter {it}: train loss {eval_out['train'].loss:.4f}, val loss {eval_out['val'].loss:.4f}, val perplexity {eval_out['val'].perplexity:.4f}, ETA {datetime.timedelta(seconds=eta)}")