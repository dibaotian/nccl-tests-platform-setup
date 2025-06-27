# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

export NCCL_SOCKET_IFNAME=eno4
export GLOO_SOCKET_IFNAME=eno4
export OMP_NUM_THREADS=1

export NCCL_IB_DISABLE=1       # 禁用 InfiniBand
export NCCL_P2P_DISABLE=1      # 禁用点对点通信
export NCCL_SHM_DISABLE=0      # 启用共享内存

# torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 --master-addr=10.161.176.120 --master-port=1234 nccl_test.py
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 --master-addr=10.161.176.120 --master-port=1234 test_ddp.py




