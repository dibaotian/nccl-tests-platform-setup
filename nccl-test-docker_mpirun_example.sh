# in the test the hostfile setting as following
# 192.168.222.11 slots=1
# 192.168.222.22 slots=1 


# for MPI test
mpirun --allow-run-as-root --mca plm_rsh_agent "ssh -p2222" -np 2 --hostfile hostfile3 hostname

mpirun --allow-run-as-root --mca plm_rsh_agent "ssh -p2222" -np 2 --hostfile hostfile3 nvidia-smi

mpirun --allow-run-as-root --mca plm_rsh_agent "ssh -p2222" -np 2 --hostfile hostfile3 nvidia-smi

# for nccl test

NCCL_DEBUG=INFO mpirun --mca plm_rsh_agent "ssh -p2222" -np 2 --allow-run-as-root -x NCCL_SOCKET_IFNAME=eth0 --hostfile hostfile3 ../docker-nccl-tests/build/all_reduce_perf -b 1M -e 8M -f 2

NCCL_DEBUG=Info mpirun --mca plm_rsh_agent "ssh -p2222" \
                --allow-run-as-root -np 2 \
                --hostfile hostfile3 \
                -x NCCL_SOCKET_IFNAME=eth0  \
                -x NCCL_IB_DISABLE=0 \
                -x NCCL_TOPO_DUMP_FILE=./topo2bond.txt \
                -x NCCL_IB_HCA=xib_0 \
                -x NCCL_IB_GID_INDEX=1 \
                -x NCCL_NET_PLUGIN=none \
                ../docker-nccl-tests/build/all_reduce_perf -b 1M -e 16M -f 2 -g 1 

mpirun -np 2 \
  --mca plm_rsh_agent "ssh -p2222" \
  --allow-run-as-root \
  --hostfile hostfile3 \
  -x NCCL_DEBUG=INFO \
  -x NCCL_IB_GID_INDEX=1  \
  -x NCCL_DEBUG_SUBSYS=INIT,NET \
  -x NCCL_IB_DISABLE=0 \
  -x NCCL_SOCKET_IFNAME=eth0 \
  ../nccl-tests/build/all_reduce_perf -b 8M -e 16MM -f 2 -g 1 


NCCL_DEBUG=TRACE  mpirun -np 2\
                      --allow-run-as-root \
                      --hostfile hostfile3 \
                      --mca btl tcp,self \
                      -mca btl_tcp_if_include eth0 \
                      -x NCCL_IB_DISABLE=0 \
                      -x NCCL_IB_GID_INDEX=1  \
                      -x NCCL_SOCKET_IFNAME=eth0 \
                      -x NCCL_TOPO_DUMP_FILE=./topo2bond.txt \
                      -x NCCL_IB_HCA=xib_0 \
                      --oversubscribe  \
                     ../docker-nccl-tests/build/all_reduce_perf -b 1M -e 16M -f 2 -g 1

# -x NCCL_IB_TX_DEPTH=128 \

mpirun -np 2 \
       --mca plm_rsh_agent "ssh -p2222" \
       --allow-run-as-root \
       --hostfile hostfile3 \
       -mca orte_base_help_aggregate 0 \
       -mca btl_base_verbose 100 \
       -mca btl tcp,self \
       -mca btl_tcp_if_include eth0  \
       -x NCCL_IB_HCA=xib_0 \
       -x NCCL_IB_DISABLE=0 \
       -x NCCL_SOCKET_IFNAME=eth0 \
       -x NCCL_IB_GID_INDEX=1  \
       -x NCCL_DEBUG=INFO \
       -x NCCL_TOPO_DUMP_FILE=./topo2fpgardma.txt \
       --oversubscribe   \
       ../docker-nccl-tests/build/all_reduce_perf -b 1M -e 128M -f 2 -g 1

NCCL_DEBUG=Info mpirun --mca plm_rsh_agent "ssh -p2222" \
            --allow-run-as-root \
            -np 2 --mca btl_tcp_if_include eth0 \
            --hostfile hostfile3 \
            -x NCCL_SOCKET_IFNAME=eth0  \
            -x NCCL_IB_HCA=xib_0 \
            -x NCCL_IB_DISABLE=0 \
             -x NCCL_IB_GID_INDEX=1  \
            ../nccl-tests/build/all_reduce_perf -b 1M -e 32M -f 2 -g 1