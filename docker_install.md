<!-- toc -->
#  1. setup in container

## 1.1. setup docker env(in host)
'''      
apt install -y docker  
apt install -y docker.io  
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)  
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list  
apt-get update  
apt-get install -y nvidia-docker2  
apt-get install -y nvidia-container-toolkit  
systemctl restart docker  
'''    

## 1.2. 运行nccl container 使用host的网络
'''  
nvidia-docker run -ti -e NVIDIA_VISIBLE_DEVICES=all --privileged --net=host -v /home:/home  -w /home  --name nccl  nvcr.io/nvidia/pytorch:24.03-py3  tail  -f /dev/null
'''  

## 1.3. 运行container 然后进行docker内的安装和配置
'''  
sudo docker exec -it nccl bash   进入docker  
apt-get update  
apt-get install -y openssh-server
'''    

## 1.4. 配置 /etc/ssh/sshd_config 将默认的端口更改为 2222
'''  
sed -i 's/#Port 22/Port 2222/' /etc/ssh/sshd_config
'''  

## 1.5. 确保允许 root 用户登录
'''  
sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
'''  

## 1.6. 设置root 用户密码
'''  
echo 'root:xilinx' | chpasswd
'''  

## 1.7. 启动 SSH 服务
'''   
service ssh start  
'''  

## 1.8. (在host上)使用以下命令检查 2222 端口是否在监听：
'''  
netstat -tuln | grep 2222
'''

## 1.9. 容器中，切换到 root 用户，并生成 SSH 密钥对
'''
ssh-keygen -t rsa -b 2048  
'''  

## 1.10. 分发公钥：将生成的公钥复制到另一个容器的 root 用户的 ~/.ssh/authorized_keys 文件中
'''  
ssh-copy-id -i ~/.ssh/id_rsa.pub -p 2222 root@192.168.222.22  
'''  

## 1.11. 更新docker里的rdma库(这个库编译参考host_install.md)
'''  
cd rdma-core  
cp -rf build/include/* /usr/include/  
cp -rf build/lib/* /usr/lib64/  
cp -rf build/lib/* /usr/lib/x86_64-linux-gnu  
cp -rf build/bin/* /usr/bin  
'''  

## 1.12. 更新后正常情况运行ibv_devinfo可找到设备
'''  
ibv_devinfo  
'''  


## 1.13. 在docker里验证rdma是否可以正常运行
'''  
进入example目录  
@server  
./rdma_rc_example -d xib_0 -g 1 -t write  -s 1024  

@client  
./rdma_rc_example -d xib_0 -g 1 -t write  -s 1024 192.168.222.11  


如果找不到ib_dev, ldd 一下库的路径，如果不是rdma_core 库copy的路径，使用环境变量指定一下LD_LIBRARY_PATH  
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH 
''' 

## 1.14. 编译nccl-test需要mpi, nccl和cuda的库和头文件路径
'''
查询mpi的库和头文件
gcc -I/opt/hpcx/ompi/include -I/opt/hpcx/ompi/include/openmpi -I/opt/hpcx/ompi/include/openmpi/opal/mca/hwloc/hwloc201/hwloc/include -I/opt/hpcx/ompi/include/openmpi/opal/mca/event/libevent2022/libevent -I/opt/hpcx/ompi/include/openmpi/opal/mca/event/libevent2022/libevent/include -L/opt/hpcx/ompi/lib -Wl,-rpath -Wl,/opt/hpcx/ompi/lib -Wl,--enable-new-dtags -lmpi 

查询nccl的库和头文件  
locate nccl.h  
locate libnccl.so  

常见的结果  
/usr/include/nccl.h （头文件）   
/usr/lib/x86_64-linux-gnu/libnccl.so 或 /usr/local/cuda/lib64/libnccl.so(库文件)  
'''

## 1.15. 编译nccl-test 
'''  
export MPI_HOME=/opt/hpcx/ompi/ 
export MPI_INCLUDE=/opt/hpcx/ompi/include  
export CUDA_HOME=/usr/local/cuda  
export NCCL_HOME=/usr/lib/x86_64-linux-gnu  

make MPI=1 MPI_HOME=$MPI_HOME CUDA_HOME=$CUDA_HOME NCCL_HOME=$NCCL_HOME CXXFLAGS="-I$MPI_INCLUDE"  
or
make MPI=1 MPI_HOME=$MPI_HOME CUDA_HOME=$CUDA_HOME NCCL_HOME=$NCCL_HOME INCLUDES="-I$MPI_INCLUDE"
'''




## 1.16. 创建新的镜像 （docker commit <CONTAINER_ID_OR_NAME> <NEW_IMAGE_NAME>:<TAG>）
'''   
sudo docker commit  nccl nccl_rdma01:min   
'''   


## 1.17. 使用docker安装的参考文档
'''  
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

'''  
