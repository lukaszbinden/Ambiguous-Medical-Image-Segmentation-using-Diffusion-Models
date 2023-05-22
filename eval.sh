#!/bin/bash

MODEL_FLAGS="--image_size 128 --num_channels 64 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"

python scripts/segmentation_sample.py  --data_dir ./data/testing  --model_path ./results/savedmodel000500.pt --num_ensemble=4 $MODEL_FLAGS $DIFFUSION_FLAGS


