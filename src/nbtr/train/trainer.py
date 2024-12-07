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

TRAIN_DATA_FILES = {
 "train_000000.bin": "1bBTywzrmvokX57LP5pe2GSTO5N-N5xXS",
 "train_000001.bin": "1eHp4uCQveLJOMHeCtrcm-arhPvlzgJgo",
 "train_000002.bin": "1W8HxpP5rDTLuY_XsCqke-fQsqGRP1q3A",
 "train_000003.bin": "1HB_DhGeKNZe2nMJ58_RRiTLiPGGBvsjr",
 "train_000004.bin": "1W0bvzaGRAQAziL_n_05LrTsvMtVryd-z",
 "train_000005.bin": "1F6AtAoXcHYGrZ7eDeuo-w2oqAFylIgXJ",
 "train_000006.bin": "1sdHk5ds0YiWq1KYiVIo0N5hL9t-yFCck",
 "train_000007.bin": "1pAbWDCy6M43M5XkYRxyCCYOgHAcaRtSw",
 "train_000008.bin": "1fPaJEDd493oN8lq7fL2d8pdiytG5vLxD",
 "train_000009.bin": "1cftwTPHJymZptrANm-GEbdc0AoIcY-rS",
 "train_000010.bin": "1l6b6sEzD7L0F5x_6QFQNx6qAsSFhyHn8",
 "train_000011.bin": "1VtH7iUBe7U_I2Y3nSJiD0dpv5cbU6sVM",
 "train_000012.bin": "1hnUs5drfWNndZroHtuif9EwJ_6YHTU5t",
 "train_000013.bin": "1tGSLsmDSj6Epn9_SEPxWtqUU9pqhpIlf",
 "train_000014.bin": "1ALLXmxoP6bAtwYQ-6Q3ytVOJOH3B61z7",
 "train_000015.bin": "10Ges75n8EEwIm8yfrls-ZQrSSCfHtAGt",
 "train_000016.bin": "1xyBBvCjlbTNmqo91cx-B4NUIlyWPOrui",
 "train_000017.bin": "1YKZT02HXK3W0KVJIJ2nN7WyqcqAcQz8t",
 "train_000018.bin": "1-6wJgvP27CSQt9TsVXCeY2OT8roT3Uvk",
 "train_000019.bin": "1RDq2GPR2OlAEnpmCNs5QSKMrqb8eHYeP",
 "train_000020.bin": "1iA6SyH9aVv7zNhSne3pVBXig_rWaGdK7",
 "train_000021.bin": "1PyVnPAH8vzFl--YUCg7xyhXpyRMry9ot",
 "train_000022.bin": "1Ou7-PM58b9HNf4YwpqkIG8MTuf9msZwh",
 "train_000023.bin": "1_Mwx9s15pE-dQVuYGqVtdsCXYEQIlPma",
 "train_000024.bin": "1tO6INRwyWFjVdg8umJYTBE8cym0mqc0y",
 "train_000025.bin": "1do2lZ3EQUt1uj_9YTwWSa6vNOPPH5db0",
 "train_000026.bin": "1hvFt915BSJz3-DxzTpnwfW21hafIAwgP",
 "train_000027.bin": "1CHNDYwcSYfZ5Qaz5OF1808rftVOW3OiX",
 "train_000028.bin": "1Z7VdiL5HI6mVQYvaDAAYRY4cl-M1RxnA",
 "train_000029.bin": "19g6q2N2Ep3CQy36of4PqzM7rddsFcx0C",
 "train_000030.bin": "1sTY8n1XzXpSAH2ZV9Ei_y_HgrT5XzM5k",
 "train_000031.bin": "1Bcaszdn6dLjboAB7PFucyNi_in05ygvn",
 "train_000032.bin": "1g42ypsof0TJOsbk6RGwDwR_m604TM4Gb",
 "train_000033.bin": "1dZa0hG26NYy9-pBxyIdx5Jhoy6YktU82",
 "train_000034.bin": "1ZffZvErW0Q6l3jFU7nFqQBEeW_mWnBHZ",
 "train_000035.bin": "1BoUxXHI-Ni3bqYX2Y1Ws7uHThD-nNOm7",
 "train_000036.bin": "18fWIUYgDc9X4kdNPvQkho1W26CExC9jW",
 "train_000037.bin": "1572L_4qdYBgCP6yz1_uieB5XJGHXzpSZ",
 "train_000038.bin": "1jJKZJUihtHVp8g4njPoD-DiAIkmDjomD",
 "train_000039.bin": "14QYJWyfcXB2dDiEab5-zxTCbjDxI0mJj",
 "train_000040.bin": "1kWmgzIaMb4nTXlnVipwTio7p7oldrkvt",
 "train_000041.bin": "1ZvUYQhT36TKK__r6PW9QNvoLSMUbrvG7",
 "train_000042.bin": "1QzW70fzg5Vm2GSZkKhQRaHWr0eS6BLrt",
 "train_000043.bin": "1Ie5274MfTaJ4KKW1Ve2mG4DzfdoZq_e_",
 "train_000044.bin": "1S03JsNm7hoJskMO1M2zFvsMb4qVwUZKO",
 "train_000045.bin": "1277FG-MdAvkLy4bCY6sPGahnjGzqXDv-",
 "train_000046.bin": "1NkTowgSN2-uzX_t3GJCB6ZJRPlEYHB7I",
 "train_000047.bin": "1yOwRNOS1LYnZRZatWhke-wAswHYg2YCl",
 "train_000048.bin": "1cALZP-hXJxBsRbZtStNUVZeIb_UCEj4K",
 "train_000049.bin": "1o95LDlRqExrQE5bF51QttnN_BHcimMKf",
 "train_000050.bin": "1VVnqOzxFAVzWULjwWQsHxtaUa4IS9FL8",
 "train_000051.bin": "1_FYw1qb_L-y8pp1KOa3hxyhOyglt5yd-",
 "train_000052.bin": "1wUaquPK47azKgKOB-6OPl-zlArCh3bUq",
 "train_000053.bin": "1MaSwWox9uxWPrdVFrIPeMeHMaZkdWIJA",
 "train_000054.bin": "1ePImhivBH35I2934qEVaCzSiS5mW-ph8",
 "train_000055.bin": "1txLDwceva6i3Z6Zzppgo4zOTBsOHH1La",
 "train_000056.bin": "1wEgQ3HaSJmoa8iTnXG8Vvpsf0bEfi3uU",
 "train_000057.bin": "1G6WpyewKGcXwEkDlXFMcNshtTTfmitNI",
 "train_000058.bin": "1eqY4dbTSOmMSCX9IVk4mARB5yVcmZ3vz",
 "train_000059.bin": "1HDH4XSEJCxRTaLdGYfzLcfyIp97OKBdh",
 "train_000060.bin": "1Ffcn7UqNqCiMrEjE8uKHTTnQI6At0_Xy",
 "train_000061.bin": "1m_4IVCUJP3yEa8Sr9ICQGGys_eSaMxUf",
 "train_000062.bin": "1ISwbqApkC1Ht8PLWDsvr7TBRLhPwYw4s",
 "train_000063.bin": "1nMsMxCedZsZX6IC7ZPCSFL46-ZeZo0CP",
 "train_000064.bin": "1_RmgtyJ1VvzY0jfUu-JG_ApFg9aF81z8",
 "train_000065.bin": "1gJvM6m8AiEHdSiHAVfXWvKpAPqT28Z1d",
 "train_000066.bin": "19-J8CKs6VWsFQvB7cAIao2vNEuluCMZS",
 "train_000067.bin": "1mEWnthzvW_JMusavlvvXTpDP8LSDELOI",
 "train_000068.bin": "17sC-W8HSCO85kN_U-wluzwSDz4azwG3u",
 "train_000069.bin": "1ybNQszJ93uBJrlMdOx-qBVG_zFjR66-2",
 "train_000070.bin": "1FiC7-g1ZbzoFvnTC2G9Bjjy9RNlT0Klv",
 "train_000071.bin": "1OelfmWIAIbCVlEOMZXgsfW2i9qblLNhk",
 "train_000072.bin": "18icsHifOYr7Fihg0Z7FlH8-YHJ7TIPeb",
 "train_000073.bin": "1RkyyMLvfqurOE6gv8n4CVlT6KyvgRcfO",
 "train_000074.bin": "17YNBjeCCWhBsdaLZto9624D7uZIvkofC",
 "train_000075.bin": "1nevJZ6_jMiItUy-TjMRImFfMj6NXGPEl",
 "train_000076.bin": "1DZbKIypaHyHA1tCprjCeFMkcgsN-LqBj",
 "train_000077.bin": "1zKjY0AX8Jh3hiHv1Eh4DjpNY4O46oHrm",
 "train_000078.bin": "1mAjuZXc7ZRBO0POzJAJYrMo8zOvqBvP4",
 "train_000079.bin": "1LDNBGYbxSNmsf9o3i3l0Lu6v8u_QqfeX",
 "train_000080.bin": "1UOA__T21jQuscgqx-KqXz_IsyIyHVo5b",
 "train_000081.bin": "1QYr0uGo4EqXtDHSjohagCEZiMmatKOm4",
 "train_000082.bin": "1AGFu6He4ovyL440K_SwluenXPp2EoKCn",
 "train_000083.bin": "1kTnIYxFswRZb7eKCvxEbCFQSRh7dL2Tj",
 "train_000084.bin": "1aErJzVOp7p9V1uLSJ-wFr_J3kY7bjZlD",
 "train_000085.bin": "1CKHao8FL-eFavgCr1gk1I6C9hn5cmMEI",
 "train_000086.bin": "1rpIrOG9vRf5bERG6EK_gnK3dYLpYKmJu",
 "train_000087.bin": "13qn6zq7inqVGQR5I4ynhUTe6EFO3Fk4J",
 "train_000088.bin": "1qbsmk1GyB5HYx1-xE-oenEm_4BCRL869",
 "train_000089.bin": "1fTPXuR-EbwbNyqzsAqJg9_DHROAzQPk2",
 "train_000090.bin": "1UN8tWG2aDMsHhmQB7G8hJwKTaIGn7otQ",
 "train_000091.bin": "1xacIkTpC6CbqHEGdXxkDHmQOpcNqCodF",
 "train_000092.bin": "1Jq_cs08zIv130RSMF8_SoxEO3rmTG7Ub",
 "train_000093.bin": "13to6seKsekNnjP70kyxBg_8YhxqHivdq",
 "train_000094.bin": "1my-DHyTlPyWi4CDuwC3hRm-D8Pgicpkx",
 "train_000095.bin": "1Bp5Su972mDsi08kWvXtWwXySJkIPMEPO",
 "train_000096.bin": "1id6UZzkUL6BEJbeoQ7hrfLtppWUwo5pG",
 "train_000097.bin": "1cBvWQ1PDjzU7X96FiTtuQuYSp-YqxU5h",
 "train_000098.bin": "1c0ZwUDBs9zvUQp407Qp5132dSUlzSsh0",
 "train_000099.bin": "1pfe98hFTUFNGbElHDbCYGMO34BUmLPnc",
 "train_000100.bin": "1L53Gp9kXZNFqTE1PeZuktzcDGXQPx7ns",
 "train_000101.bin": "1zqpOhm7PNH6M1RkmF25XUDVRwA63W3qr",
 "train_000102.bin": "18Qa1omQl92QaxtHQ9uOKnwGhvdtnbhdA",
 "train_000103.bin": "1gkZMnNLIZ5p1Gf10deVvztWCUQbIzTaz",
 "train_000104.bin": "1v8gLAzOB0hfDGu-ipohHCMT1fXCNGfq_",
 "train_000105.bin": "1SCqRpMu5Hg-B4EbWZEeTn9hv8kj2mGCX",
 "train_000106.bin": "1cnmdVgjK8CNBs84G2XbkxO1HOIqhtrwF",
 "train_000107.bin": "1uUWJjKkkNAdkSxBw-OSf9HWqb_E5KfD9",
 "train_000108.bin": "1zqysW3nQn_cCyKz4gQDVa-HQwPDX26yu",
 "train_000109.bin": "1G90V3sFy8WvkaXXtufJTHetNKYuNruBh",
 "train_000110.bin": "1xzifp6nKfW2w8aQqvqmNn5jSWzN6Z68R",
 "z_newstr_train_000000.bin": "1M0iMuNRVVEhzBw5NBwBtiafYOCt4UaUb",
 "z_newstr_train_000001.bin": "147_FF8DuPY8NiMsKla8D9tUoZ6Qv9f7_",
 "z_newstr_train_000002.bin": "1zKDsEe3v9NaKtXW-c2X4TSHx6xxYFIky",
 "z_newstr_train_000003.bin": "1X4ai9ODdxw7YwAXCmniCc0oX2KNiZIMU",
 "z_newstr_train_000004.bin": "1aOH5hd4GJWA1cctJH94g9rW2mbYDcU2d",
 "z_newstr_train_000005.bin": "1Qdtv3IcjdLceaIkN99gArCvnZiIQjiky",
 "z_newstr_train_000006.bin": "1OnlufUSY3_llQSKeyOwFXmiJhVmlN07X",
 "z_newstr_train_000007.bin": "1XJlpwKntrflQWrZn7lhSsECsQ8mbGOY1",
 "z_newstr_train_000008.bin": "1jR1zaH7CdJfHoYBXmWbBmyCbOQogYfRl",
 "z_newstr_train_000009.bin": "1w2_2GVRmZVjLIQ-xIRxdmx9nv16gdvkp",
 "z_newstr_train_000010.bin": "1QOzB5pwbmWkJSKqxcy4Bqr4a4cEcMox3",
 }
VAL_DATA_FILES={
 "val_000000.bin": "12IxOvGIBTO1wgp9C_P6BdVvQGuYCQpbj"
}
NEWS_TR_VAL_DATA_FILES={
 "z_newstr_val_000000.bin": "1yhIDFDVgG2pezrY_sHLxeRV-NJOdmJrL"
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
        self.train_ddl = DDL(f"{self.config.data_dir}/*train_*.bin", self.config.batch_size, self.config.seq_length, self.rank, self.world_size)
        
        # validation data loaders
        val_train_ddl = DDL(f"{self.config.data_dir}/train_000000.bin", self.config.batch_size, self.config.seq_length, self.rank, self.world_size)
        val_ddl = DDL(f"{self.config.data_dir}/val_000000.bin", self.config.batch_size, self.config.seq_length, self.rank, self.world_size)
        newstr_val_ddl = DDL(f"{self.config.data_dir}/z_newstr_val_000000.bin", self.config.batch_size, self.config.seq_length, self.rank, self.world_size)
        
        self.val_loaders_dict = {"train":val_train_ddl, "val":val_ddl, "news_tr_val": newstr_val_ddl}
    
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
        for split in ['train', 'val', 'news_tr_val']:
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