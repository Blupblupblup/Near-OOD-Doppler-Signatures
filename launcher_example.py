from datetime import datetime
start_time = datetime.now()

import os
import subprocess
from utils import *

nbr_seeds = 2
nbr_epochs = 300

lr_init = 0.0001
lr_milestones = [200]
lr_decay = 0.1
weight_decay = 1e-6

batchsize = 1000
train_ratio = 0.9
nbr_pulse_per_scan = 64
nbr_targets = 3000

nbr_modes_SAD = 1
SAD_ratio = 0.01

arch = "net0"
arch_suffix = "_"+arch
representation = "sp"
dataset_name = "{}targets-64bins-{}".format(nbr_targets, representation)

current_dir = os.path.dirname(os.path.abspath(__file__)) + '/'

experiment_name = "example_experiments"
init_comparison_df(experiment_name)

for nbr_modes in [1, 2]:

    # for examples with other methods presented in paper see launched using launcher_paper_shallow_experiments.py and launcher_paper_deep_experiments.py

    subprocess.run(
        ["python", current_dir + "deep_svdd.py", "dsvdd-ssldata-away_saddata-away" + arch_suffix,
         dataset_name,
         str(nbr_modes),
         str(nbr_epochs), str(batchsize),
         str(lr_init), str(lr_decay), str(lr_milestones), str(weight_decay), str(nbr_seeds), str(train_ratio),
         str(nbr_pulse_per_scan),
         str(nbr_targets), "SAD+SSL", "ssldata-away_saddata-away", arch, str(nbr_modes_SAD), str(SAD_ratio), experiment_name])

time_elapsed = datetime.now() - start_time
print("Complete execution time:", time_elapsed)