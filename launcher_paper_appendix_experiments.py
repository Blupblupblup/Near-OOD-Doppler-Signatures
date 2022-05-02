from datetime import datetime
start_time = datetime.now()

import os
import subprocess
from utils import *

nbr_seeds = 10
nbr_epochs = 300

lr_init = 0.0001
lr_milestones = [200]
lr_decay = 0.1
weight_decay = 1e-6

batchsize = 1000
train_ratio = 0.9
nbr_pulse_per_scan = 64
nbr_targets = 3000

nbr_RPs = 1000
unit_norm = int(True)

nbr_modes_SAD = 1
SAD_ratio = 0.01

arch = "net0"
arch_suffix = "_"+arch
representation = "sp"
dataset_name = "{}targets-64bins-{}".format(nbr_targets, representation)

current_dir = os.path.dirname(os.path.abspath(__file__)) + '/'

experiment_name = "appendix_deep_experiments"
init_comparison_df(experiment_name)

for estimator_RPs in ["mean", "max"]:
    for nbr_modes in [1, 2]:

        ################################################################### DEEP EXPERIMENTS WITHOUT POLLUTION

        #### RPO & adaptations

        subprocess.run(
            ["python", current_dir + "deep_rpo.py", "deep-rpo" + arch_suffix + "_" + estimator_RPs, dataset_name,
             str(nbr_modes),
             str(nbr_epochs), str(batchsize),
             str(lr_init), str(lr_decay), str(lr_milestones), str(weight_decay), str(nbr_seeds), str(train_ratio),
             str(nbr_pulse_per_scan),
             str(nbr_targets), estimator_RPs, str(nbr_RPs), "AD", "deep-rpo", arch, str(nbr_modes_SAD),
             str(SAD_ratio), experiment_name])

        subprocess.run(
            ["python", current_dir + "deep_rpo.py", "drpo-ssldata-sslcentroid" + arch_suffix + "_" + estimator_RPs, dataset_name,
             str(nbr_modes),
             str(nbr_epochs), str(batchsize),
             str(lr_init), str(lr_decay), str(lr_milestones), str(weight_decay), str(nbr_seeds), str(train_ratio),
             str(nbr_pulse_per_scan),
             str(nbr_targets), estimator_RPs, str(nbr_RPs), "SSL", "ssldata-sslcentroid", arch, str(nbr_modes_SAD),
             str(SAD_ratio), experiment_name])

        subprocess.run(
            ["python", current_dir + "deep_rpo.py", "drpo-ssldata-away" + arch_suffix + "_" + estimator_RPs, dataset_name,
             str(nbr_modes),
             str(nbr_epochs), str(batchsize),
             str(lr_init), str(lr_decay), str(lr_milestones), str(weight_decay), str(nbr_seeds), str(train_ratio),
             str(nbr_pulse_per_scan),
             str(nbr_targets), estimator_RPs, str(nbr_RPs), "SSL", "ssldata-away", arch, str(nbr_modes_SAD),
             str(SAD_ratio), experiment_name])

        subprocess.run(
            ["python", current_dir + "deep_rpo.py", "drpo-saddata-sadcentroid" + arch_suffix + "_" + estimator_RPs, dataset_name,
             str(nbr_modes),
             str(nbr_epochs), str(batchsize),
             str(lr_init), str(lr_decay), str(lr_milestones), str(weight_decay), str(nbr_seeds), str(train_ratio),
             str(nbr_pulse_per_scan),
             str(nbr_targets), estimator_RPs, str(nbr_RPs), "SAD", "saddata-sadcentroid", arch, str(nbr_modes_SAD),
             str(SAD_ratio), experiment_name])

        subprocess.run(
            ["python", current_dir + "deep_rpo.py", "drpo-saddata-away" + arch_suffix + "_" + estimator_RPs, dataset_name,
             str(nbr_modes),
             str(nbr_epochs), str(batchsize),
             str(lr_init), str(lr_decay), str(lr_milestones), str(weight_decay), str(nbr_seeds), str(train_ratio),
             str(nbr_pulse_per_scan),
             str(nbr_targets), estimator_RPs, str(nbr_RPs), "SAD", "saddata-away", arch, str(nbr_modes_SAD),
             str(SAD_ratio), experiment_name])

        subprocess.run(
            ["python", current_dir + "deep_rpo.py", "drpo-ssldata-sslcentroid_saddata-away" + arch_suffix + "_" + estimator_RPs, dataset_name, str(nbr_modes),
             str(nbr_epochs), str(batchsize),
             str(lr_init), str(lr_decay), str(lr_milestones), str(weight_decay), str(nbr_seeds), str(train_ratio),
             str(nbr_pulse_per_scan),
             str(nbr_targets), estimator_RPs, str(nbr_RPs), "SAD+SSL", "ssldata-sslcentroid_saddata-away", arch,
             str(nbr_modes_SAD), str(SAD_ratio), experiment_name])

        subprocess.run(
            ["python", current_dir + "deep_rpo.py", "drpo-ssldata-away_saddata-sadcentroid" + arch_suffix + "_" + estimator_RPs, dataset_name,
             str(nbr_modes),
             str(nbr_epochs), str(batchsize),
             str(lr_init), str(lr_decay), str(lr_milestones), str(weight_decay), str(nbr_seeds), str(train_ratio),
             str(nbr_pulse_per_scan),
             str(nbr_targets), estimator_RPs, str(nbr_RPs), "SAD+SSL", "ssldata-away_saddata-sadcentroid", arch,
             str(nbr_modes_SAD), str(SAD_ratio), experiment_name])

        subprocess.run(
            ["python", current_dir + "deep_rpo.py", "drpo-ssldata-sslcentroid_saddata-sadcentroid" + arch_suffix + "_" + estimator_RPs, dataset_name,
             str(nbr_modes),
             str(nbr_epochs), str(batchsize),
             str(lr_init), str(lr_decay), str(lr_milestones), str(weight_decay), str(nbr_seeds), str(train_ratio),
             str(nbr_pulse_per_scan),
             str(nbr_targets), estimator_RPs, str(nbr_RPs), "SAD+SSL", "ssldata-sslcentroid_saddata-sadcentroid", arch,
             str(nbr_modes_SAD), str(SAD_ratio), experiment_name])

        subprocess.run(
            ["python", current_dir + "deep_rpo.py", "drpo-ssldata-away_saddata-away" + arch_suffix + "_" + estimator_RPs,
             dataset_name,
             str(nbr_modes),
             str(nbr_epochs), str(batchsize),
             str(lr_init), str(lr_decay), str(lr_milestones), str(weight_decay), str(nbr_seeds), str(train_ratio),
             str(nbr_pulse_per_scan),
             str(nbr_targets), estimator_RPs, str(nbr_RPs), "SAD+SSL", "ssldata-away_saddata-away", arch,
             str(nbr_modes_SAD), str(SAD_ratio), experiment_name])

        ################################################################### DEEP EXPERIMENTS WITH POLLUTION

        #### RPO & adaptations

        subprocess.run(
            ["python", current_dir + "deep_rpo.py", "drpo-saddata-pollution" + arch_suffix + "_" + estimator_RPs,
             dataset_name,
             str(nbr_modes),
             str(nbr_epochs), str(batchsize),
             str(lr_init), str(lr_decay), str(lr_milestones), str(weight_decay), str(nbr_seeds), str(train_ratio),
             str(nbr_pulse_per_scan),
             str(nbr_targets), estimator_RPs, str(nbr_RPs), "SAD", "saddata-pollution", arch,
             str(nbr_modes_SAD), str(SAD_ratio), experiment_name])

        subprocess.run(
            ["python", current_dir + "deep_rpo.py", "drpo-ssldata-sslcentroid_saddata-pollution" + arch_suffix + "_" + estimator_RPs,
             dataset_name,
             str(nbr_modes),
             str(nbr_epochs), str(batchsize),
             str(lr_init), str(lr_decay), str(lr_milestones), str(weight_decay), str(nbr_seeds), str(train_ratio),
             str(nbr_pulse_per_scan),
             str(nbr_targets), estimator_RPs, str(nbr_RPs), "SAD+SSL", "ssldata-sslcentroid_saddata-pollution", arch,
             str(nbr_modes_SAD), str(SAD_ratio), experiment_name])

        subprocess.run(
            ["python", current_dir + "deep_rpo.py", "drpo-ssldata-away_saddata-pollution" + arch_suffix + "_" + estimator_RPs,
             dataset_name,
             str(nbr_modes),
             str(nbr_epochs), str(batchsize),
             str(lr_init), str(lr_decay), str(lr_milestones), str(weight_decay), str(nbr_seeds), str(train_ratio),
             str(nbr_pulse_per_scan),
             str(nbr_targets), estimator_RPs, str(nbr_RPs), "SAD+SSL", "ssldata-away_saddata-pollution", arch,
             str(nbr_modes_SAD), str(SAD_ratio), experiment_name])

time_elapsed = datetime.now() - start_time
print("Complete execution time:", time_elapsed)