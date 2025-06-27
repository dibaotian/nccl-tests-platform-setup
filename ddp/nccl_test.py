# encoding: utf8

# torchrun --nnodes=1 --nproc_per_node=2  rccl_test.py --local_world_size=2

# python -m torch.utils.collect_env

"""
single node
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_SOCKET_IFNAME=en04 OMP_NUM_THREADS=1 \
RDZV_ID=456 RDZV_ENDPOINT=10.161.176.13:1234 RDZV_BACKEND=c10d \
torchrun --nproc_per_node=1 \
         --nnodes=1 \
         --node_rank=0 \
         --master-addr=10.161.176.13 \
         --master-port=1234 \
         nccl_test.py
"""

"""
multinode
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_SOCKET_IFNAME=eno4 GLOO_SOCKET_IFNAME=eno4 \
OMP_NUM_THREADS=1 RDZV_ID=456 RDZV_ENDPOINT=10.161.176.120:1234 RDZV_BACKEND=c10d \
torchrun --nproc_per_node=1 \
         --nnodes=2 \
         --node_rank=1 \
         --master-addr=10.161.176.120 \
         --master-port=1234 \
         nccl_test.py
         
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_SOCKET_IFNAME=eno4 GLOO_SOCKET_IFNAME=eno4 \
OMP_NUM_THREADS=1 RDZV_ID=456 RDZV_ENDPOINT=10.161.176.120:1234 RDZV_BACKEND=c10d \
torchrun --nproc_per_node=1 \
         --nnodes=2 \
         --node_rank=1 \
         nccl_test.py
"""


import logging
import os
import torch
from datetime import datetime
import torch.distributed as dist
from torch.distributed import ReduceOp
from datetime import timedelta

def print_env_vars():
    # 定义需要读取的环境变量
    important_vars = [
        "NCCL_DEBUG",
        "NCCL_SOCKET_IFNAME",
        "OMP_NUM_THREADS",
        "MASTER_ADDR",
        "MASTER_PORT",
        "RDZV_ID",
        "RDZV_ENDPOINT",
        "RDZV_BACKEND",
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
    ]
    
    print("==== Important Environment Variables ====")
    for var in important_vars:
        value = os.environ.get(var, "Not Set")
        print(f"{var}: {value}")
    
def get_local_rank():
    
    if not dist.is_initialized():
        dist.init_process_group(
        backend="nccl",
        init_method="env://",  # 使用环境变量方式初始化
    )
        
    global_rank = dist.get_rank()
    num_gpus_per_node = torch.cuda.device_count()
    return global_rank % num_gpus_per_node

def get_local_ranks_per_node():
    # 初始化分布式环境
    if not dist.is_initialized():
        dist.init_process_group(
        backend="nccl",
        init_method="env://",  # 使用环境变量方式初始化
    )

    # 获取全局信息
    world_size = dist.get_world_size()       # 总进程数
    local_rank = int(os.environ["LOCAL_RANK"])  # 当前进程的本地 rank
    node_rank = int(os.environ.get("NODE_RANK", 0))  # 当前节点 rank

    # 获取每个节点的 local rank
    ranks_on_this_node = []
    for global_rank in range(world_size):
        if global_rank // torch.cuda.device_count() == node_rank:
            ranks_on_this_node.append(global_rank)

    return {
        "local_rank": local_rank,
        "all_ranks_on_node": ranks_on_this_node,
    }
    
def dist_scatter():
    
    local_rank = get_local_rank()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if local_rank == 0:
        print("RANK",rank)
        print("scatter:")
        
    
    # # Tensor for each process
    # tensor = torch.zeros(world_size).to(0)
    # before_tensor = tensor.clone()
    
     # Tensor for each process (created on GPU)
    device = torch.device(f"cuda:{local_rank}")
    tensor = torch.zeros(world_size).to(device)  # 放在 GPU 上
    before_tensor = tensor.clone()
    
    # Process 0 initializes scatter_list
    if rank == 0:
        scatter_list = [
            torch.ones(world_size, device=device) * (i + 1) for i in range(world_size)
        ]
        print(f"scatter_list created: {scatter_list}")
    else:
        scatter_list = None
        
    try:
        dist.scatter(tensor, scatter_list, src=0)
        print(f"scatter, rank: {rank}, before scatter: {repr(before_tensor)}, after scatter: {repr(tensor)}")
    except Exception as e:
        print(f"Rank {rank}: Error during scatter: {e}")
        raise e

def main():
    
    if not dist.is_initialized():
        dist.init_process_group(
        backend="nccl",
        init_method="env://",  # 使用环境变量方式初始化
    )
    
    local_rank = get_local_rank()
    
    if(local_rank == 0):
        
        print_env_vars()
        
        print("==== Important System info ====")
        
        print("System time:", datetime.now())
        
        print("Torch version", torch.__version__)
    
        if torch.version.hip is not None:
            print(f"ROCm HIP Version: {torch.version.hip}")
        else:
            print("ROCm is not available.")
            
        
        # print("torch cuda version",torch.version.cuda)
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"Number of GPUs: {num_gpus}")
            print(torch.cuda.get_device_name(0))
        else:
            print("No GPUs found!")
            
        print(f"Backend: {dist.get_backend()}")
        
        # 如果使用 RCCL，则版本号可以通过内部日志确
        print("Torch distributed RCCL version:", torch.cuda.nccl.version())
        
        print("==== Important System info ====")
        print("Global RANK", dist.get_rank())
        info = get_local_ranks_per_node()
        print(f"Local Rank: {info['local_rank']}")
        print(f"All Ranks on Current Node: {info['all_ranks_on_node']}")
        print("=========================================")
        
    print(f"Rank {dist.get_rank()} initialized!")
    

if __name__ == "__main__":
    
    
    # # 设置网络接口
    # os.environ["NCCL_SOCKET_IFNAME"] = "ens27f0"
    
    # # 配置参数
    # world_size = 2  # 机器总数
    # master_ip = "10.170.6.199"  # 主节点 IP 地址
    # master_port = "12355"  # 通信端口
    
    main()
    dist_scatter()
    print(f"Complete!")
    
    try:
        dist.destroy_process_group()
    except Exception as e:
        print(f"Error destroying process group: {e}")
        
    exit()
