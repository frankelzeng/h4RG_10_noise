#!/bin/bash
#SBATCH --account=PCON0003
#SBATCH --job-name=CZ_pixel_50_createDF
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --exclusive
#SBATCH --mem=8gb

source activate sktime-dev
python scratch.py
