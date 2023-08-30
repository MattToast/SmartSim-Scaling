#!/bin/bash

#SBATCH -N 162
#SBATCH --exclusive
#SBATCH -t 10:00:00

cd ..
module load slurm
/lus/scratch/drozt/miniconda3/envs/scale/bin/python driver.py throughput_standard \
                                     --client_nodes=[1,32,64,128] \
                                     --clients_per_node=[1,8,16,32,64] \
                                     --db_nodes=[8,16,32] \
                                     --db_cpus=[36] \
                                     --tensor_bytes=[8,16,32,1024,8192,16384,32769,65538,131076,262152,524304,1024000] \
                                     --net_ifname=ipogif0 \
                                     --run_db_as_batch=False \
                                     --launcher=slurm \
                                     --languages=["python"] \
                                     --iterations=100
