from dataclasses import dataclass, asdict, field
import torch
import torch.nn as nn
import numpy as np
import os
import math
import time
import datetime
from collections import defaultdict
from typing import Any, Dict, List
from ..utils.mfu import estimate_mfu
from enum import Enum
from ..model.trainer_model import TrainerModel
from nbtr.utils.training_logger import TrainingLogger
from nbtr.utils.mup_session import MupSession


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
    ## training logging
    log_to: List[str] = None
    wandb_project: str = None
    wandb_run_name: str = None
    wandb_run_id: str = None

    # MUP
    mup_enabled:bool = False # Whether to use muP. If False then all other mup variables are ignored
    mup_disable_attention_scaling = False # Uses 1/sqrt(d_head) attn scaling instead of 1/d_head (Only needed for the step-by-step coord check in the blog)
    mup_disable_hidden_lr_scaling = False # Disables muP hidden LR adjustment (Only needed for the step-by-step coord check in the blog)
    mup_width_multiplier = 1.0 # mup_width_multiplier = width / base_width where base_width is typically 256
    mup_input_alpha = 1.0 # Optional tunable multiplier applied to input embedding forward pass output
    mup_output_alpha = 1.0 # Optional tunable multiplier applied to output unembedding forward pass output
    mup_enable_coord_check_logging = False # If True will track the output.abs().mean() of various layers throughout training
    
    def __post_init__(self):
        self.lr_decay_iters = self.lr_decay_iters if self.lr_decay_iters else self.max_iters
    
    @staticmethod
    def from_yaml(config_file:str):
        import yaml

        with open(config_file) as f:
            doc = yaml.safe_load(f)
        
        return TrainerConfig(**doc)

    def to_yaml(self, config_file:str):
        import yaml

        with open(config_file, "w") as f:
            yaml.dump(asdict(self), f, indent=2)

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
    
class Trainer:
    def __init__(self, config:TrainerConfig, rank:int=0, world_size:int=1, state:TrainingState=None) -> None:
        self.config = config
        
        self.rank = rank
        self.world_size = world_size
        self.device = "cuda" if world_size == 1 else f"cuda:{rank}"
        
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

    def get_batch(self, split):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == 'train':
            data = np.memmap(os.path.join(self.config.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(self.config.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        
        ix = torch.randint(len(data) - self.config.seq_length, (self.config.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+self.config.seq_length]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.config.seq_length]).astype(np.int64)) for i in ix])
        
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(self.device, non_blocking=True)
        y = y.pin_memory().to(self.device, non_blocking=True)

        return x, y

    def update_lr(self, it, optimizer):
        if not self.config.decay_lr:
            return self.config.learning_rate
        
        lr = self.get_lr(it)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
        
    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it):
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
    
    @torch.inference_mode()
    def evaluate(self, model:nn.Module)->Dict[str,EvalResult]:
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.config.eval_iters)
            perplexities = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                X, Y = self.get_batch(split)
                with self.ctx:
                    _, loss = model(X, Y)
                losses[k] = loss.item()
                perplexities[k] = torch.exp(loss).item()
            out[split] = EvalResult(loss=losses.mean().item(), perplexity=perplexities.mean().item())
        model.train()
        return out

    def train(self, model, raw_model, compile:bool=None):
        
        # assert model.device == DEVICE, f"Only CUDA device is supported for training. Model is in :{model.device}"
        assert self.config.seq_length == raw_model.config.seq_length, f"Sequence length for model and trainer is not equal {self.config.seq_length}!={raw_model.config.seq_length}"

        torch.manual_seed(self.config.seed)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

        if (compile is None and self.config.compile) or compile:
            print("compiling the model...")
            model = torch.compile(model)
            print("compiling the model done!")
        
        scaler = torch.amp.GradScaler(self.device, enabled=(self.config.dtype == 'float16'))
        optimizer = self._configure_optimizers(model)

        # start training
        model.train()
        
        if self.config.gc:
            model.gradient_checkpointing=True
        
        mup = MupSession(model=model, config=self.config) 
        with TrainingLogger(self.config.log_to, config=self.config) as logger:
            X, Y = self.get_batch('train')
            
            start_iter = 0 if self.state.iter_num == 0 else self.state.iter_num+1
            running_fwd_bwd_tokens_per_sec = 0
            running_iter_time = 0
            mfu = 0
            t0 = 0
            
            # if resuming a training, replay torch.randints to avoid sampling the same examples.
            if self.state.iter_num > 0:
                self.skip_first_new_best_val_loss = False
                random_seed_replay_count = self.state.iter_num *self.config.gradient_accumulation_steps + (self.state.iter_num//self.config.eval_interval)*self.config.eval_iters
                for i in range(random_seed_replay_count):
                    # I am not sure whether range matters
                    torch.randint(1024, (self.config.batch_size,))
            elif self.rank == 0:
                # do not evaluate before-hand if resuming...
                self.do_eval(raw_model, optimizer, running_fwd_bwd_tokens_per_sec, running_iter_time, start_iter, self.config.learning_rate, logger, mup.last_coord_check_dict, 0.0)
            
            for it in range(start_iter, self.config.max_iters+1):
                
                # determine current lr rate and update params if needed
                lr = self.update_lr(it, optimizer)
                
                # forward the model
                if t0 == 0:
                    t0 = time.time()
                
                with next(mup):
                    for micro_batch in range(self.config.gradient_accumulation_steps):
                        if self.world_size>1 and micro_batch == self.config.gradient_accumulation_steps - 1:
                            self.trigger_callbacks(TrainerEvent.ON_LAST_MICRO_BATCH, model)
                        with self.ctx:
                            _, loss = model(X, Y)
                            loss = loss / self.config.gradient_accumulation_steps
                            
                        # immediately async prefetch next batch while model is doing the forward pass on the GPU
                        X, Y = self.get_batch('train')

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
                        self.do_eval(raw_model, optimizer, running_fwd_bwd_tokens_per_sec, running_iter_time, it, lr, logger, mup.last_coord_check_dict, mfu)

    def do_eval(self, model, optimizer, running_fwd_bwd_tokens_per_sec, time_per_iter, it, lr, train_logger, mup_iter, mfu):
        if self.rank != 0:
            return
        
        t0 = time.time()
        eval_out = self.evaluate(model)

        log_message = {
                    "iter": it,
                    "train/loss": eval_out['train'].loss,
                    "val/loss": eval_out['val'].loss,
                    "val/perplexity": eval_out['val'].perplexity,
                    "lr": lr,
                    "tokens/sec": running_fwd_bwd_tokens_per_sec,
                    "mfu": mfu,
                }

        if mup_iter:
            log_message |= mup_iter

        train_logger.log(log_message)
            
        
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