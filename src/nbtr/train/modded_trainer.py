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

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % int(os.environ['WORLD_SIZE']) == int(os.environ['RANK']):
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    g *= max(1, g.size(0)/g.size(1))**0.5
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()
                
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
        
        ix = torch.randint(len(data) - self.config.seq_length, (self.config.batch_size,), generator=self.batch_rng)
        x = torch.stack([torch.from_numpy((data[i:i+self.config.seq_length]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.config.seq_length]).astype(np.int64)) for i in ix])
        
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(self.device, non_blocking=True)
        y = y.pin_memory().to(self.device, non_blocking=True)

        return x, y


    def get_wsd_lr(self, it):
        assert it <= self.config.max_iters
        # 1) linear warmup for warmup_iters steps
        if it < self.config.warmup_iters:
            return (it+1) / self.config.warmup_iters
        # 2) constant lr for a while
        elif it < self.config.max_iters - self.config.warmup_iters:
            return 1.0
        # 3) linear warmdown
        else:
            warmdown_iters = self.config.max_iters*0.2 # 20%
            decay_ratio = (self.config.max_iters - it) / warmdown_iters
        return decay_ratio
    
    def get_cosine_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.config.warmup_iters:
            return (it+1) / self.config.warmup_iters
        
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.config.lr_decay_iters:
            # return self.config.min_lr / self.config.learning_rate
            return 0.1
        
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.config.warmup_iters) / (self.config.lr_decay_iters - self.config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        # return (self.config.min_lr / self.config.learning_rate) + coeff * (self.config.learning_rate - self.config.min_lr)/self.config.learning_rate
        return 0.1 * coeff * 0.9
    
    def _configure_optimizers(self, model):

        # init the optimizer(s)
        optimizer1 = torch.optim.Adam([model.transformer.wte.weight], lr=0.3, betas=(self.config.beta1, self.config.beta2), fused=True)
        optimizer2 = torch.optim.Adam([model.lm_head.weight], lr=0.003, betas=(self.config.beta1, self.config.beta2), fused=True)
        optimizer3 = Muon(model.transformer.h.parameters(), lr=self.config.learning_rate, momentum=self.config.beta2)
        optimizers = [optimizer1, optimizer2, optimizer3]

        if self.state.optim_state is not None:
            raise Exception("State load not supported yet!!!!")
            # optimizer.load_state_dict(self.state.optim_state)
        
        return optimizers
    
    def _init_logging(self):
        if self.config.wandb_log and self.rank==0:
            import wandb
            # wandb.require("core")
            wandb.init(project=self.config.wandb_project, name=self.config.wandb_run_name, id=self.config.wandb_run_id, resume="allow", config=asdict(self.config))
    
    @torch.inference_mode()
    def evaluate(self, model:nn.Module)->Dict[str,EvalResult]:
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                X, Y = self.get_batch(split)
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

        if (compile is None and self.config.compile) or compile:
            print("compiling the model...")
            model = torch.compile(model)
            print("compiling the model done!")
        
        scaler = torch.amp.GradScaler(self.device, enabled=(self.config.dtype == 'float16'))
        optimizers = self._configure_optimizers(raw_model)
        schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, self.get_wsd_lr) for opt in optimizers]

        # start training
        model.train()
        
        if self.config.gc:
            model.gradient_checkpointing=True
        
        with self.init_logger(raw_model) as wb_run:
            X, Y = self.get_batch('train')
            
            start_iter = 0 if self.state.iter_num == 0 else self.state.iter_num+1
            running_fwd_bwd_tokens_per_sec = 0
            running_iter_time = 0
            mfu = 0
            t0 = 0
            
            # start_iter is to accommodate for resuming training
            # If training is resuming, random state is reshuffled to avoid using same samples for training...
            torch.manual_seed(self.config.seed+self.rank+start_iter)
            self.batch_rng.manual_seed(self.config.seed+self.rank+start_iter)
            
            if self.state.iter_num > 0:
                self.skip_first_new_best_val_loss = False
            elif self.rank == 0:
                # do not evaluate before-hand if resuming...
                self.do_eval(raw_model, optimizers, running_fwd_bwd_tokens_per_sec, running_iter_time, start_iter, schedulers, wb_run, 0.0)
            
            for it in range(start_iter, self.config.max_iters+1):
                
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
                    X, Y = self.get_batch('train')

                    scaler.scale(loss).backward()
                
                for optimizer in optimizers:
                    scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_norm_clip)
                    
                for optimizer in optimizers:
                    scaler.step(optimizer)
                    
                scaler.update()
                optimizers.zero_grad(set_to_none=True)
                
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
                        self.do_eval(raw_model, optimizers, running_fwd_bwd_tokens_per_sec, running_iter_time, it, schedulers, wb_run, mfu)

    def init_logger(self, model):
        return WandBLogger(enabled=(self.config.wandb_log and self.rank==0), project=self.config.wandb_project, name=self.config.wandb_run_name, id=self.config.wandb_run_id, config=asdict(self.config)|asdict(model.config))

    def do_eval(self, model, optimizer, running_fwd_bwd_tokens_per_sec, time_per_iter, it, schedulers, wb_run, mfu):
        if self.rank != 0:
            return
        
        t0 = time.time()
        eval_out = self.evaluate(model)
                
        wb_run.log({
                    "iter": it,
                    "train/loss": eval_out['train'].loss,
                    "val/loss": eval_out['val'].loss,
                    "val/perplexity": eval_out['val'].perplexity,
                    "lr": schedulers[-1].get_last_lr(),
                    "tokens/sec": running_fwd_bwd_tokens_per_sec,
                    "mfu": mfu,
                })
            
        
        new_val_loss = eval_out['val'].loss
        if  new_val_loss < self.state.best_val_loss:
            # self.state = TrainingState(iter_num=it, best_val_loss=new_val_loss, optim_state=optimizer.state_dict())
            self.state = TrainingState(iter_num=it, best_val_loss=new_val_loss, optim_state=dict())
            
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