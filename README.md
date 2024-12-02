# Computer-Vision-Final-Project

## Running on HPC

```bash
ssh CASEID@markov.case.edu
```

Request CPU compute node:
```bash
srun --mem=8gb --pty /bin/bash
```

```bash
module avail python
```

```bash
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/11.3.1
```

```bash
export PYTHONUSERBASE=$HOME/.usr/local/python/3.12.3
mkdir -p $PYTHONUSERBASE
```

Then install Torch/CUDA from example instructions [here](https://sites.google.com/a/case.edu/hpcc/hpc-cluster/markov-software/software-installation-guide/installing-local-python-modules).

Other helpful links:
 - https://sites.google.com/a/case.edu/hpcc/hpc-cluster/markov-software/programming-computing-languages/python2-python3/python-virtual-environments
 - https://sites.google.com/a/case.edu/hpcc/guides-and-training/helpful-references/hpc-environment/batch-job-interactive-job-submissions?authuser=0

```bash
sbatch fcos.slurm
squeue --me
cat training_output.log
```