  1 卸载cuda
  sudo apt-get purge cuda-*
  sudo apt autoremove
  sudo apt clean
  检查一下还有没有残留的cuda文件
  ls -l /usr/lib | grep cuda*
  ls -l /usr/local/ | grep cuda*
  如果还有
  sudo rm -rf /usr/local/cuda*
  sudo rm -rf /usr/include/cuda*

  2 卸载nvidia driver
  sudo apt remove --purge '^nvidia-.*'
  sudo apt autoremove
  sudo apt clean
  检查一下还有没有残留的cuda文件
  ls -l /usr/lib | grep nvidia*
  ls -l /usr/local/ | grep nvidia*
  如果还有
  sudo rm -rf /usr/local/nvidia*

  3 卸载nccl
  dpkg -l  | grep nccl
  sudo apt-get purge libnccl2 libnccl-dev 
  sudo apt-get purge nccl-tools



  3 dpkg 检查并卸载
  dpkg -l | grep -i cuda
  sudo apt-get purge cuda-* nvidia-cuda-* 

  4 清除系统中的残余库和头文件
  sudo rm -rf /etc/X11/xorg.conf
  sudo rm -rf /var/lib/dkms/nvidia*

  5 检查环境变量
  grep -i 'cuda' ~/.bashrc ~/.profile ~/.bash_profile
  如果仍然有 export PATH=/usr/local/cuda/bin:$PATH 或 export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
  需要手动删除这些行并运行source ~/.bashrc
  

  4 安装driver
  参考文档 https://docs.nvidia.com/cuda/cuda-installation-guide-linux/  （TL;TR）

  根据GPU的型号选择对应的driver版本, https://www.nvidia.com/en-us/drivers/
                                   https://www.nvidia.com/en-us/drivers/unix/
  例如针对Tesla v100：
  '''
  Driver Version:	570.124.06
  Release Date:	Mon Mar 03, 2025
  Operating System:	Linux 64-bit Ubuntu 22.04
  CUDA Toolkit:	12.8
  Language:	English (US)
  File Size:	492.37 MB
  '''

  wget https://us.download.nvidia.com/tesla/570.124.06/nvidia-driver-local-repo-ubuntu2204-570.124.06_1.0-1_amd64.deb

  sudo cp nvidia-driver-local-repo-ubuntu2204-570.124.06_1.0-1_amd64.deb /tmp/

  sudo apt install /tmp/nvidia-driver-local-repo-ubuntu2204-570.124.06_1.0-1_amd64.deb

  sudo cp /var/nvidia-driver-local-repo-ubuntu2204-570.124.06/nvidia-driver-local-F4FD6868-keyring.gpg /usr/share/keyrings/

  echo "deb [signed-by=/usr/share/keyrings/nvidia-keyring.gpg] file:///nvidia-driver-local-repo-ubuntu2204-570.124.06 ./" | sudo tee /etc/apt/sources.list.d/nvidia-local.list

  sudo apt update

  apt search nvidia-driver   #找到对应的driver版本例如nvidia-driver-570

  sudo apt install nvidia-driver-570

  <!-- sudo cp /home/xilinx/nccl/cuda_install/nvidia-driver-local-repo-ubuntu2204-535.230.02_1.0-1_amd64.deb /tmp/
  sudo apt install /tmp/nvidia-driver-local-repo-ubuntu2204-535.230.02_1.0-1_amd64.deb
  sudo cp /var/nvidia-driver-local-repo-ubuntu2204-535.230.02/nvidia-driver-local-C62A4C86-keyring.gpg /usr/share/keyrings/
  sudo apt update
  sudo apt install nvidia-driver-535 -->
  
  安装完成后重启系统
  验证验证:  nvidia-smi

  4 安装cuda
  确定和driver匹配的cuda版本
  https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
  https://docs.nvidia.com/deploy/cuda-compatibility/

  下载CUDA TOOlkit Archive的地址
  https://developer.nvidia.com/cuda-toolkit-archive
  最后选择runfile（local）

  '''
  针对 Driver Version:	570.124.06， 可以查询到CUDA 12.8 GA ----（linux Driver version）>=570.26

  wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
  sudo sh cuda_12.8.0_570.86.10_linux.run

  '''


  2
  安装的过程中不要选择安装驱动
  '''
  ===========
  = Summary =
  ===========

  Driver:   Not Selected
  Toolkit:  Installed in /usr/local/cuda-12.8/

  Please make sure that
  -   PATH includes /usr/local/cuda-12.8/bin
  -   LD_LIBRARY_PATH includes /usr/local/cuda-12.8/lib64, or, add /usr/local/cuda-12.8/lib64 to /etc/ld.so.conf and run ldconfig as root

  To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-12.8/bin
  ***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 570.00 is required for CUDA 12.8 functionality to work.
  To install the driver using this installer, run the following command, replacing <CudaInstaller> with the name of this run file:
      sudo <CudaInstaller>.run --silent --driver

  Logfile is /var/log/cuda-installer.log
  '''

  安装完成后声明环境变量
  echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
  echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
  source ~/.bashrc

  安装完成后检查nvcc版本、
  nvcc --version
  nvcc: NVIDIA (R) Cuda compiler driver
  Copyright (c) 2005-2025 NVIDIA Corporation
  Built on Wed_Jan_15_19:20:09_PST_2025
  Cuda compilation tools, release 12.8, V12.8.61
  Build cuda_12.8.r12.8/compiler.35404655_0


  5 安装nccl

  检查使用CUDA12.2的版本，需要查找并安装支持 CUDA 12.2 的 NCCL 版本
  使用 apt-cache 搜索可用版本
  apt-cache madison libnccl2
  '''
  libnccl2 | 2.25.1-1+cuda12.8 | https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  Packages
  libnccl2 | 2.25.1-1+cuda12.4 | https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  Packages
  libnccl2 | 2.25.1-1+cuda12.2 | https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  Packages

  找到合适的版本以后，例如2.25.1-1+cuda12.8
  安装指定版本
  sudo apt install -y libnccl2=2.25.1-1+cuda12.8 libnccl-dev=2.25.1-1+cuda12.8

  默认安装
  sudo apt install libnccl2 libnccl-dev

  '''
  
  检查nccl 版本,要保证多有机器上的都一样
  dpkg -l | grep nccl
  ii  libnccl-dev                                           2.25.1-1+cuda12.8                       amd64        NVIDIA Collective Communication Library (NCCL) Development Files
  ii  libnccl2                                              2.25.1-1+cuda12.8                       amd64        NVIDIA Collective Communication Library (NCCL) Runtime

  libnccl2 2.25.1-1+cuda12.8 这个命名格式表示该 NCCL 版本的详细信息：

  解析命名格式
  libnccl2：NCCL（NVIDIA Collective Communications Library）的共享库包名称，libnccl2 表示它是 NCCL 2.x 版本的运行时库。
  2.25.1：NCCL 的版本号，表示它是 2.25.1 版本。
  -1：打包的版本号，通常用于区分同一 NCCL 版本的不同打包或修订版本（可能包含小修复或调整）。
  +cuda12.8：表示该 NCCL 版本是专门为 CUDA 12.8 版本编译和优化的


  6 下载nccl test 
  cd nccl-test
  make clean

  # 编译nccl-test

  # 编译nccl-test需要mpi, nccl和cuda的库和头文件路径
  # 查询mpi的库和头文件
  mpicc --show
  gcc -I/usr/local/include -L/usr/local/lib -Wl,-rpath -Wl,/usr/local/lib -Wl,--enable-new-dtags -lmpi


  # 查询nccl的库和头文件

  sudo updatedb
  sudo ldconfig


  $>locate nccl.h
  /usr/local/include/nccl.h
  

  $>locate libnccl.so
  /usr/local/lib/libnccl.so
  /usr/local/lib/libnccl.so.2
  /usr/local/lib/libnccl.so.2.24.3

  # 使用 ldconfig 检查 NCCL 共享库
  ldconfig -p | grep nccl
  libnccl.so (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libnccl.so
  这表示 libnccl.so 位于 /usr/lib/x86_64-linux-gnu/。


  export MPI_HOME=/usr/local/
  export CUDA_HOME=/usr/local/cuda-12.8/
  export NCCL_HOME=/usr/lib/x86_64-linux-gnu/
  export NCCL_INCLUDE=/usr/include/

  make MPI=1 MPI_HOME=$MPI_HOME CUDA_HOME=$CUDA_HOME NCCL_HOME=$NCCL_HOME CXXFLAGS="-I$MPI_HOME/include -I$NCCL_INCLUDE"

  #build & install rdma driver
  modprobe ib_core
  insmod linux-drivers/xilinx_pci.ko
  insmod linux-drivers/s-nic.ko
  insmod linux-drivers/xilinx_ib.ko dyndbg=+p

  #build rdma-core
  cd rdma-core
  ./build.sh

  #安装rdma-core的库到系统
  cp -rf build/include/* /usr/include/
  cp -rf build/lib/* /usr/lib64/
  cp -rf build/lib/* /usr/lib/x86_64-linux-gnu/
  cp -rf build/bin/* /usr/bin/

  #验证是否成功
  ibv_devinfo 




  如果是在docker内
  mpicc --show
gcc -I/opt/hpcx/ompi/include -I/opt/hpcx/ompi/include/openmpi -I/opt/hpcx/ompi/include/openmpi/opal/mca/hwloc/hwloc201/hwloc/include -I/opt/hpcx/ompi/include/openmpi/opal/mca/event/libevent2022/libevent -I/opt/hpcx/ompi/include/openmpi/opal/mca/event/libevent2022/libevent/include -L/opt/hpcx/ompi/lib -Wl,-rpath -Wl,/opt/hpcx/ompi/lib -Wl,--enable-new-dtags -lmpi

  export MPI_HOME=/opt/hpcx/ompi
  export CUDA_HOME=/usr/local/cuda-12.4/
  export NCCL_HOME=/usr/lib/x86_64-linux-gnu/

  make MPI=1 MPI_HOME=$MPI_HOME CUDA_HOME=$CUDA_HOME NCCL_HOME=$NCCL_HOME CXXFLAGS="-I$MPI_HOME/include"




  使用docker安装的参考文档
  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

