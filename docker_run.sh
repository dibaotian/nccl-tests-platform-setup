# nvidia-docker run -ti -e NVIDIA_VISIBLE_DEVICES=all --privileged --net=host -v /home:/home  -w /home  --name nccl  nvcr.io/nvidia/pytorch:24.03-py3  tail  -f /dev/null
# nvidia-docker run --hostname rdma01  -ti -e NVIDIA_VISIBLE_DEVICES=all --privileged --net=host -v /home:/home  -w /home  --name nccl  nccl_rdma01:min  tail  -f /dev/null

nvidia-docker run -d --hostname rdma01 -ti \
  -e NVIDIA_VISIBLE_DEVICES=all --privileged --net=host \
  -v /home:/home -w /home --hostname rdma01 --name nccl nccl_rdma01:min \
  bash -c "service ssh start && tail -f /dev/null"

# nvidia-docker run -d --hostname rdma02 -ti \
#   -e NVIDIA_VISIBLE_DEVICES=all --privileged --net=host \
#   -v /home:/home -w /home --hostname rdma02 --name nccl nccl_rdma02:min \
#   bash -c "service ssh start && tail -f /dev/null"

# 运行nccl container 使用host的网络
# nvidia-docker run -d --hostname rdma01 -ti \
#   -e NVIDIA_VISIBLE_DEVICES=all --privileged --net=host \
#   -v /home:/home  -w /home --hostname rdma01 --name nccl  \
#   nvcr.io/nvidia/pytorch:24.03-py3  tail  -f /dev/null
