import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

def setup(rank, world_size, master_ip, master_port):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = master_ip
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()

def train(rank, world_size, master_ip, master_port):
    # 初始化分布式环境
    setup(rank, world_size, master_ip, master_port)

    # 定义简单的模型和数据集
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)

    # 使用 DistributedSampler 确保数据在多个进程间不重复
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)

    # 定义简单的模型、损失和优化器
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10)).cuda(rank)
    ddp_model = DDP(model, device_ids=[rank])  # 分布式封装

    criterion = nn.CrossEntropyLoss().cuda(rank)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # 训练循环
    for epoch in range(5):
        sampler.set_epoch(epoch)  # 确保每个 epoch 的数据划分不同
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(rank), target.cuda(rank)

            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}")

    cleanup()

if __name__ == "__main__":
    # 配置参数
    world_size = 2  # 机器总数
    master_ip = "10.170.6.199"  # 主节点 IP 地址
    master_port = "12355"  # 通信端口

    # 每台机器运行一个进程，分别设置 rank 为 0 和 1
    rank = int(os.environ['RANK'])  # 从环境变量获取 rank
    train(rank, world_size, master_ip, master_port)


