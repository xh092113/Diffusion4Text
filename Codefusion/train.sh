#!/bin/bash
OUT_DIR="Diffusion4Text/Codefusion"
DATA_PATH="/Your/pretrain/data/path"
python -u Diffusion4Text/Codefusion/Genie_Pretrain.py \
--checkpoint_path=$OUT_DIR \
--model_channels 512 --in_channel 512 --out_channel 512 --vocab_size 30522 \
--config_name="bert-base-uncased" --token_emb_type="random" --model_arch="s2s_CAT" \
--diffusion_steps 1200 --noise_schedule="sqrt" --training_mode="s2s" \
--schedule_sampler="uniform" --pre_max_len 128 --mask_pro 0.3 --seed 2023 \
--data_path=$DATA_PATH \
--batch_size 64 --lr 5e-04 --warmup_steps 300000 --train_type="S2S_Diffusion" \
--eval_interval 2000 --log_interval 2000 --save_interva 20000