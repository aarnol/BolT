#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --partition=amilan

#SBATCH -c 24
#SBATCH --output=prep_hcp-%j.out
#SBATCH --account=ucb474_asc2

module purge

module load anaconda 
conda activate BolT

cd /projects/alar6830/BolT

python prep.py --dataset hcpWM --atlas sphere --name sphere107 --fnirs True 

echo "done prepping"
