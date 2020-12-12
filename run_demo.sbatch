#!/bin/bash

#SBATCH -J bert_classify         # Job 名
#SBATCH -o bert_finetune_demo.log   # 输出文件，也可删除此行由系统自动指定.
#SBATCH -N 1               # 申请节点数
#SBATCH -G 2
#SBATCH -w compute2

module purge
module add miniconda3
module add cuda-10.2
conda activate torch_env

srun python -u run.py 2>&1

