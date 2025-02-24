ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
port=${ports[0]}

echo "total workers: ${ARNOLD_WORKER_NUM}"
echo "cur worker id: ${ARNOLD_ID}"
echo "gpus per worker: ${ARNOLD_WORKER_GPU}"
echo "master ip: ${METIS_WORKER_0_HOST}"
echo "master port: ${port}"

export WANDB_API_KEY=""
export WANDB_MODE="offline"
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=${ARNOLD_RDMA_DEVICE}
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
# export NCCL_P2P_DISABLE=1
export PYTHONFAULTHANDLER=1
export MASTER_PORT=$port
export MASTER_ADDR=$METIS_WORKER_0_HOST
export WORK_DIR=your_path/Dora/pytorch_lightning

source your_path/miniconda3/bin/activate Dora
cd your_path/Dora/pytorch_lightning


torchrun \
        --nnodes $ARNOLD_WORKER_NUM \
        --node_rank $ARNOLD_ID \
        --nproc_per_node $ARNOLD_WORKER_GPU \
        --master_addr $METIS_WORKER_0_HOST \
        --master_port $port \
        ./launch.py    --train \
        --gpu 0,1,2,3,4,5,6,7 \
        --config $WORK_DIR/configs/shape-autoencoder/Dora-VAE-train.yaml  \
        trainer.num_nodes=$ARNOLD_WORKER_NUM > train_$ARNOLD_ID.log 2>&1



        