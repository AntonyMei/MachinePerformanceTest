set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
# PongNoFrameskip-v4, BreakoutNoFrameskip-v4
#rm ./models/*
python -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 1 main.py \
  --env BreakoutNoFrameskip-v4 --case atari --opr train --force \
  --seed 0 \
  --use_priority \
  --use_max_priority \
  --amp_type 'torch_amp' \
  --info 'EfficientZero-V1'
