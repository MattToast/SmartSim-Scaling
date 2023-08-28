#!/bin/bash

#SBATCH -N 140
#SBATCH --exclusive
#SBATCH -t 10:00:00

cd ..
module load slurm
python driver.py throughput_standard --client_nodes=[128] \
                                     --clients_per_node=[8] \
                                     --db_nodes=[8] \
                                     --db_cpus=[36] \
                                     --tensor_bytes=[1024] \
                                     --net_ifname=ipogif0 \
                                     --run_db_as_batch=False \
                                     --launcher=slurm \
                                     --languages=["python"] \
                                     --iterations=100

# Commented ideal vals bc horizon is small
# python driver.py throughput_standard --client_nodes=[128,256,512] \
#                                      --clients_per_node=[8,16,32,64,128,256] \
#                                      --db_nodes=[8,16,32] \
#                                      --tensor_bytes=[1024,8192,16384,32769,65538,131076,262152,524304,1024000] \
