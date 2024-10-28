# Single-GPU Launching
LAUNCHER=python

# Multi-GPU Launching (single node)
#GPU=2
#LAUNCHER=torchrun --standalone --nproc_per_node=$GPU

LAYERS=2

# for width in 256 512 1024 2048
for width in 1024
do
    # for lr in 0.125 0.0625 0.03125 0.015625 0.0078125 0.00390625 0.001953125 0.0009765625 0.00048828125 0.000244140625 0.0001220703125 0.00006103515625
    for lr in 0.000244140625 0.0001220703125 0.00006103515625
    do
        # for seed in 1 2 3
        for seed in 2 3
        do
            head_size=64
            n_heads=$((width / head_size))
            min_lr=$(awk "BEGIN {print $lr/10}")
            out_dir="mup_examples/mutransfer_lr_owt/sp/out/width${width}_depth${LAYERS}_seed${seed}_lr${lr}"
            $LAUNCHER -m nbtr.cli_train \
                --trainer_config "config/news_trainer.yml" \
                --model_config "config/news_model.yml" \
                --out_dir $out_dir \
                --data_dir 'data/news-tr-1.8M-tokenizer-8k' \
                --model.n_head $n_heads \
                --model.n_embed $width \
                --learning_rate $lr \
                --min_lr $min_lr \
                --seed $seed
        done
    done
done