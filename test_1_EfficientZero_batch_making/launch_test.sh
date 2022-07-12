set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
python main.py \
  --env BreakoutNoFrameskip-v4 --case atari --opr worker --force \
  --cpu_actor 1 --gpu_actor 1 --worker_node_id 0 \
  --seed 0 \
  --use_priority \
  --use_max_priority \
  --amp_type 'torch_amp' \
  --info 'EfficientZero-V1'
