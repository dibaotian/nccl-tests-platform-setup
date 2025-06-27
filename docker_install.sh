apt install -y docker
apt install -y docker.io
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update
apt-get install -y nvidia-docker2
apt-get install -y nvidia-container-toolkit
systemctl restart docker

# 运行nccl container 使用host的网络
nvidia-docker run -ti -e NVIDIA_VISIBLE_DEVICES=all --privileged --net=host -v /home:/home  -w /home  --name nccl  nvcr.io/nvidia/pytorch:24.03-py3  tail  -f /dev/null


# 运行container 以后需要安装和配置
sudo docker exec -it nccl bash

apt-get update
apt-get install -y openssh-server

# 配置 /etc/ssh/sshd_config 将默认的端口更改为 2222
sed -i 's/#Port 22/Port 2222/' /etc/ssh/sshd_config
# 确保允许 root 用户登录
sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# 设置root 用户密码
echo 'root:xilinx' | chpasswd

#启动 SSH 服务 
service ssh start

# 使用以下命令检查 2222 端口是否在监听：
netstat -tuln | grep 2222


# 容器中，切换到 root 用户，并生成 SSH 密钥对
ssh-keygen -t rsa -b 2048

# 分发公钥：将生成的公钥复制到另一个容器的 root 用户的 ~/.ssh/authorized_keys 文件中。
ssh-copy-id -i ~/.ssh/id_rsa.pub -p 2222 root@192.168.222.22

# 更新docker里的rdma库
cd rdma-core
cp -rf build/include/* /usr/include/
cp -rf build/lib/* /usr/lib64/
cp -rf build/lib/* /usr/lib/x86_64-linux-gnu
cp -rf build/bin/* /usr/bin

# 更新后正常情况运行ibv_devinfo可找到设备
ibv_devinfo 
# libibverbs: Warning: couldn't load driver 'libirdma-rdmav25.so': libirdma-rdmav25.so: cannot open shared object file: No such file or directory
# XIB provider version: 2.0.2
# xib_alloc_context cmd_fd 3 size 0x200000 mmap_key 0x0
# xib_alloc_context db_addr 0x7f91eb9b9000 size 0x200000 mmap_key 0x0
# get dev_attr from kernel:  max_qp 1024 max_qp_wr 1024 max_sge 16 max_sge_rd 256 max_cq 1024 max_cqe 1048576 max_mr 1024 max_pd 1024 max_srq 0 max_srq_wr 0 max_srq_sge 0 max_mr_size 0x10000000
# hca_id: xib_0
#         transport:                      InfiniBand (0)
#         fw_ver:                         2.0.002
#         node_guid:                      024b:77ff:fe8e:b174
#         sys_image_guid:                 024b:77ff:fe8e:b174
#         vendor_id:                      0x0000
#         vendor_part_id:                 0
#         hw_ver:                         0x0
#         phys_port_cnt:                  1
#                 port:   1
#                         state:                  PORT_ACTIVE (4)
#                         max_mtu:                4096 (5)
#                         active_mtu:             1024 (3)
#                         sm_lid:                 0
#                         port_lid:               0
#                         port_lmc:               0x00
#                        link_layer:             Ethernet

# 在docker里验证rdma是否可以正常运行
# 进入example目录
@server
./rdma_rc_example -d xib_0 -g 1 -t write  -s 1024

@client
./rdma_rc_example -d xib_0 -g 1 -t write  -s 1024 192.168.222.11
# 观察是否能够正常执行
#如果找不到ib_dev, ldd 一下库的路径，如果不是rdma_core 库copy的路径，使用环境变量指定一下LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

# 编译nccl-test需要mpi, nccl和cuda的库和头文件路径
# 查询mpi的库和头文件
mpicc --show
gcc -I/usr/local/include -L/usr/local/lib -Wl,-rpath -Wl,/usr/local/lib -Wl,--enable-new-dtags -lmpi

# 查询nccl的库和头文件
locate nccl.h
locate libnccl.so
# 常见的结果：
/usr/include/nccl.h （头文件）
/usr/lib/x86_64-linux-gnu/libnccl.so 或 /usr/local/cuda/lib64/libnccl.so （库文件




# 编译nccl-test
# export MPI_HOME=/usr/local/mpi
export MPI_INCLUDE=/usr/local/include
export MPI_HOME=/usr/local/lib
export CUDA_HOME=/usr/local/cuda
export NCCL_HOME=/usr/lib/x86_64-linux-gnu/

make MPI=1 MPI_HOME=$MPI_HOME CUDA_HOME=$CUDA_HOME NCCL_HOME=$NCCL_HOME CXXFLAGS="-I$MPI_INCLUDE"




#创建新的镜像--docker commit <CONTAINER_ID_OR_NAME> <NEW_IMAGE_NAME>:<TAG>
sudo docker commit  nccl nccl_rdma01:min

