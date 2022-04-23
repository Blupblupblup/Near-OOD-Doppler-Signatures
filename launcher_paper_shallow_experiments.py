from datetime import datetime
start_time = datetime.now()

import os
import subprocess
from utils import *

nbr_seeds = 1

batchsize = 1000
train_ratio = 0.9
nbr_pulse_per_scan = 64
nbr_targets = 3000

nbr_RPs = 1000
estimator_rps = "max"
unit_norm = int(True)
pca_explained_var = 0.95
n_components_pca = 20
n_components_nnpca = 10

riem_metric_tpca = "log"
tpca_nbr_components = 20

current_dir = os.path.dirname(os.path.abspath(__file__)) + '/'

experiment_name = "paper_shallow_experiments"
init_comparison_df(experiment_name)

for nbr_modes in [1,2]:

    ##################################################################

    representation = "sp"
    dataset_name = "{}targets-64bins-{}".format(nbr_targets, representation)

    subprocess.run(["python", current_dir + "shallow_pca.py", "isolation-forest", "IF", dataset_name, representation, str(nbr_modes), str(pca_explained_var),
                    str(nbr_seeds), str(train_ratio), str(nbr_pulse_per_scan), str(nbr_targets), experiment_name])

    subprocess.run(["python", current_dir + "shallow_pca.py", "local-outlier-factor", "LOF", dataset_name, representation, str(nbr_modes), str(pca_explained_var),
                    str(nbr_seeds), str(train_ratio), str(nbr_pulse_per_scan), str(nbr_targets), experiment_name])

    subprocess.run(["python", current_dir + "shallow_pca.py", "one-class-svm", "OCSVM", dataset_name, representation, str(nbr_modes), str(pca_explained_var),
                    str(nbr_seeds), str(train_ratio), str(nbr_pulse_per_scan), str(nbr_targets), experiment_name])

    subprocess.run(["python", current_dir + "shallow_pca.py", "RPO-{}-{}".format(estimator_rps, nbr_RPs), "RPO",
                    dataset_name, representation, str(nbr_modes), str(pca_explained_var), str(nbr_seeds), str(train_ratio),
                    str(nbr_pulse_per_scan), str(nbr_targets), str(nbr_RPs), estimator_rps, str(unit_norm), experiment_name])

    subprocess.run(["python", current_dir + "shallow_pca.py", "nn-pca-{}-{}".format(n_components_pca, n_components_nnpca), "NN-PCA",
                    dataset_name, representation, str(nbr_modes), str(pca_explained_var), str(nbr_seeds), str(train_ratio),
                    str(nbr_pulse_per_scan), str(nbr_targets), experiment_name, str(n_components_pca), str(n_components_nnpca)])

    ##################################################################

    representation = "cov"
    dataset_name = "{}targets-64bins-{}".format(nbr_targets, representation)

    subprocess.run(["python", current_dir + "shallow_pca.py", "isolation-forest", "IF", dataset_name, representation, str(nbr_modes), str(pca_explained_var),
                    str(nbr_seeds), str(train_ratio), str(nbr_pulse_per_scan), str(nbr_targets), experiment_name])

    subprocess.run(["python", current_dir + "shallow_pca.py", "local-outlier-factor", "LOF", dataset_name, representation, str(nbr_modes), str(pca_explained_var),
                    str(nbr_seeds), str(train_ratio), str(nbr_pulse_per_scan), str(nbr_targets), experiment_name])

    subprocess.run(["python", current_dir + "shallow_pca.py", "one-class-svm", "OCSVM", dataset_name, representation, str(nbr_modes), str(pca_explained_var),
                    str(nbr_seeds), str(train_ratio), str(nbr_pulse_per_scan), str(nbr_targets), experiment_name])

    subprocess.run(["python", current_dir + "shallow_pca.py", "RPO-{}-{}".format(estimator_rps, nbr_RPs), "RPO",
                    dataset_name, representation, str(nbr_modes), str(pca_explained_var), str(nbr_seeds), str(train_ratio),
                    str(nbr_pulse_per_scan), str(nbr_targets), str(nbr_RPs), estimator_rps, str(unit_norm), experiment_name])

    subprocess.run(["python", current_dir + "shallow_pca.py", "nn-pca-{}-{}".format(n_components_pca, n_components_nnpca), "NN-PCA",
                    dataset_name, representation, str(nbr_modes), str(pca_explained_var), str(nbr_seeds), str(train_ratio),
                    str(nbr_pulse_per_scan), str(nbr_targets), experiment_name, str(n_components_pca), str(n_components_nnpca)])

    ###################################################################

    # TangentPCA on SPD covariance input representations with geomstats as dimensionality reduction

    representation = "cov"
    dataset_name = "{}targets-64bins-{}".format(nbr_targets, representation)

    subprocess.run(["python", current_dir + "shallow_tpca.py", "isolation-forest-riem", "IF", dataset_name, str(nbr_modes),
                    str(nbr_seeds), str(train_ratio), str(nbr_pulse_per_scan), str(nbr_targets), riem_metric_tpca, str(tpca_nbr_components), experiment_name])

    subprocess.run(["python", current_dir + "shallow_tpca.py", "local-outlier-factor-riem", "LOF", dataset_name, str(nbr_modes),
                    str(nbr_seeds), str(train_ratio), str(nbr_pulse_per_scan), str(nbr_targets), riem_metric_tpca, str(tpca_nbr_components), experiment_name])

    subprocess.run(["python", current_dir + "shallow_tpca.py", "one-class-svm-riem", "OCSVM", dataset_name, str(nbr_modes),
                    str(nbr_seeds), str(train_ratio), str(nbr_pulse_per_scan), str(nbr_targets), riem_metric_tpca, str(tpca_nbr_components), experiment_name])

    subprocess.run(["python", current_dir + "shallow_tpca.py", "RPO-{}-{}-riem".format(estimator_rps, nbr_RPs), "RPO",
                    dataset_name, str(nbr_modes), str(nbr_seeds), str(train_ratio),
                    str(nbr_pulse_per_scan), str(nbr_targets), riem_metric_tpca, str(tpca_nbr_components),
                    str(nbr_RPs), estimator_rps, str(unit_norm), experiment_name])

    subprocess.run(["python", current_dir + "shallow_tpca.py", "nn-tpca-riem-{}-{}".format(tpca_nbr_components, n_components_nnpca), "NN-PCA", dataset_name, str(nbr_modes),
                    str(nbr_seeds), str(train_ratio), str(nbr_pulse_per_scan), str(nbr_targets), riem_metric_tpca, str(tpca_nbr_components),
                    experiment_name, str(n_components_nnpca)])

time_elapsed = datetime.now() - start_time
print("Complete execution time:", time_elapsed)