#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --partition=amem
#SBATCH --qos=mem

#SBATCH -c 16
#SBATCH --output=prep_hcp-%j.out
#SBATCH --account=ucb474_asc1

module purge

module load anaconda 
conda activate BolT

cd /projects/alar6830/BolT

python prep.py --dataset hcpWM --atlas brodmann --name fNIRS --fnirs True

echo "done prepping"
