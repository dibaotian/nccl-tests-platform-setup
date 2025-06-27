import torch.distributed as dist
import torch
import os
from datetime import timedelta

def get_local_rank():
    
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=10))
        
    global_rank = dist.get_rank()
    num_gpus_per_node = torch.cuda.device_count()
    return global_rank % num_gpus_per_node

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
    
    dist.barrier()
    
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
    
    dist.barrier()

def main():
    # 设置网络接口为 eth0
    os.environ["NCCL_SOCKET_IFNAME"] = "eth0"

    if torch.version.hip is not None:
        print(f"ROCm Version: {torch.version.hip}")
    else:
        print("ROCm is not available.")
        
    print(torch.cuda.get_device_name(0))

    print("torch version", torch.__version__)
    print("torch cuda version",torch.version.cuda)
        
    # dist.init_process_group(backend="nccl")  # 使用 "nccl"（适用于 GPU）
    dist.init_process_group(backend="nccl", init_method="tcp://192.168.222.11:1234", rank=1, world_size=2)
    # dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:1234", rank=0, world_size=1) # this does not work also

    print(f"Backend: {dist.get_backend()}")

    rank=dist.get_rank()
    print(f"Global Rank {rank} initialized!")

    # 创建数据
    # 因为目前每个机器上只有一个GPU，所以local_rank = 0 , 所有的tensor都初始化在这个gpu上
    tensor_send = torch.ones(100).to(0)
    tensor_recv = torch.zeros(100).to(0)

    # 在 rank 0 发送数据给 rank 1
    if rank == 0:
        dist.send(tensor=tensor_send, dst=1)
        print(f"Rank {rank} sent tensor: {tensor_send}")

    # 在 rank 1 接收来自 rank 0 的数据
    elif rank == 1:
        dist.recv(tensor=tensor_recv, src=0)
        print(f"Rank {rank} received tensor: {tensor_recv}")

    # 在 rank 1 发送数据给 rank 0
    if rank == 1:
        dist.send(tensor=tensor_send, dst=0)
        print(f"Rank {rank} sent tensor: {tensor_send}")

    # 在 rank 0 接收来自 rank 1 的数据
    elif rank == 0:
        dist.recv(tensor=tensor_recv, src=1)
        print(f"Rank {rank} received tensor: {tensor_recv}")
        
    dist_scatter()

    # 确保正确销毁进程组
    try:
        dist.destroy_process_group()
    except Exception as e:
        print(f"Rank {rank}: Error destroying process group: {e}")
    
if __name__ == "__main__":
    main()
