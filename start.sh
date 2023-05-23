# train
export NCCL_IB_DISABLE=1

#/root/miniconda3/envs/transreid/bin/python -m torch.distributed.launch --nnodes=$HOST_NUM --node_rank=$INDEX --nproc_per_node $HOST_GPU_NUM --master_addr $CHIEF_IP --master_port 669 train_pretrain.py MODEL.DIST_TRAIN True

/root/miniconda3/envs/transreid/bin/python -m torch.distributed.launch --nnodes=$HOST_NUM --node_rank=$INDEX --nproc_per_node $HOST_GPU_NUM --master_addr $CHIEF_IP --master_port 669 train_cuhkpedes_gpu.py MODEL.DIST_TRAIN True
#/root/miniconda3/envs/transreid/bin/python -m torch.distributed.launch --nnodes=$HOST_NUM --node_rank=$INDEX --nproc_per_node $HOST_GPU_NUM --master_addr $CHIEF_IP --master_port 669 train_icfg_gpu.py MODEL.DIST_TRAIN True











