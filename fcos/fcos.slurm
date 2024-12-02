#!/bin/bash
#SBATCH --job-name=training_job        # Job name
#SBATCH --output=training_output.log   # Output log file
#SBATCH --gres=gpu:1                   # Request 1 GPU (adjust as needed)
#SBATCH --time=12:00:00                # Maximum run time (12 hours)
#SBATCH --mem=16G                      # Request 16GB of memory
#SBATCH --cpus-per-task=4              # Allocate 4 CPU cores
#SBATCH --account=yxy1421_csds465      # Ensure the correct account
#SBATCH --partition=markov_gpu         # Request the markov_gpu partition
#SBATCH --nodelist=classt07,classt08   # Specify idle nodes (modify based on availability)
# Load necessary modules
module load Python/3.12.3-GCCcore-13.3.0
module load cuda/11.3.1
export PYTHONUSERBASE=$HOME/.usr/local/python/3.12.3

# Activate virtual environment (if created)
# source ~/myenv/bin/activate

# Run the Python script
python ~/Computer-Vision-Final-Project/fcos/fcos.py