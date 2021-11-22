#!/bin/bash
#SBATCH --job-name=dataset_preprocess
#SBATCH --output=data_out.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --time=3:00
#SBATCH --mail-type=ALL
#SBATCH --mem=247

module load miniconda
conda activate py3_env2

python data.py

