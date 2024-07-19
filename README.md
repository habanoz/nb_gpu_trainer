# GPT GPU Trainer for notebooks

A notebook first training training repository to use cloud notebook (with GPU) providers. It saves training state to huggingface to resume after a crash. It logs to wandb to monitor progress.  

Model code is copied from the haber-gpt repository. Training script is built up-on haber-gpt repository. 

- Trainer: Cuda targeted, Mixed-precision, flash-attention enabled model training. Expects dataset to be in a flat numpy array format.
- HFBackedTrainer: A wrapper around the Trainer. It integrates with the huggingface hub to save and retrieve the model and optimizer state.

## Configuration

`config` directory contains example configuration files.

### Model configuration

```yaml
seq_length: 512
vocab_size: 8192
n_embed: 384
n_head: 6
n_layer: 6
dropout: 0.2
```

### Training configuration

```yaml
seq_length: 512
batch_size: 32
ds_repo_id : habanoz/eco-news-tr
data_dir: haber-data
warmup_iters: 100
learning_rate:  0.001
lr_decay_iters: 5000
max_iters: 5000
min_lr: 0.0001
weight_decay: 0.1
beta1:  0.9
beta2: 0.99
compile: False
decay_lr: True
grad_norm_clip:  1.0
seed: 145
# output
out_dir: haber-gpt
# logging 
log_interval: 10
eval_interval: 250
eval_iters: 200
wandb_log: True
wandb_project: NB-Haber-GPT-Training
wandb_run_name: haber-gpt-v1.0
wandb_run_id: "1721342904"
repo_id: habanoz/haber-gpt-v1.0
```

## How to train

You first need to install.
```bash
pip install -e .
```


First, train your tokenizer.

```python
Trainer(trainer_cfg.ds_repo_id,trainer_cfg.repo_id, trainer_cfg.out_dir).train()
```

Then prepare your data:

```python
tokenizer = Tokenizer.from_pretrained(trainer_cfg.repo_id)
tokenizer.encode_ds_from_hub(trainer_cfg.ds_repo_id, trainer_cfg.data_dir)
```

Train:
```python
model = GPT.from_config(model_cfg_file)
model.to("cuda")

trainer = HFBackedTrainer.from_config(trainer_cfg_file)
trainer.train(model)
```

## Example notebook to start training

Checkout `notebooks` directory for examples. 

It is recommended to clone this repository and edit the configuration per your needs then use it in the notebook to train. Here is a colab notebook to begin with.

https://colab.research.google.com/drive/14jIzTnKvharpRV9gzl9hRGJhn8pvfx83?usp=sharing

## Acknowledgement

This work is based on Haber-GPT which is inspired/based on nanoGPT/minGPT of Andrej Karpathy.  
