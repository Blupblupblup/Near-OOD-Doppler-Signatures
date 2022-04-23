import datetime
import sys
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import seaborn as sns
import random

from training import *
from rpo import *
from norm_negated_pca import *

import geomstats.geometry.spd_matrices as spd
from geomstats.learning.pca import TangentPCA

def shallow_tpca(AD_name, shallow_name, dataset, nbr_modes, nbr_seeds, train_ratio,
                     nbr_pulse_per_scan, nbr_targets, riem_metric_tpca, tpca_nbr_components,
                 date_string, nbr_projections=1000, estimator="mean",
                     unit_norm=True, n_components_nntpca=10):

    print("\n {}".format(AD_name))

    # Default device to 'cpu' if cuda is not available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    classes = [1, 2, 3, 4]

    results_dicts_list = []

    for exp_index, seed in enumerate(range(nbr_seeds)):

        print("\n seed: {}".format(seed))

        # https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        g = torch.Generator()
        g.manual_seed(seed)

        # useless here but required to generate the dataset using the generic function
        nbr_modes_SAD = 1
        SAD_ratio = 0.01
        batchsize = 1000

        normal_cls = np.random.choice(classes, nbr_modes, replace=False).tolist()
        outlier_cls = [cls for cls in classes if cls not in normal_cls]
        SAD_cls = np.random.choice(outlier_cls, nbr_modes_SAD, replace=False).tolist()

        radar_dataset = simulated_radar_dataset(train_ratio, normal_cls, SAD_cls, SAD_ratio, batchsize, nbr_targets, nbr_pulse_per_scan)
        radar_dataset.get_dataloaders(generator=g)

        train_sig = radar_dataset.train_spds_2D
        complete_train_sig = radar_dataset.complete_train_spds_2D
        val_sig = radar_dataset.val_spds_2D
        test_sig = radar_dataset.test_spds_2D

        dim = train_sig[0].shape[1]
        if riem_metric_tpca=="log":
            riem_metric = spd.SPDMetricLogEuclidean(dim)
        elif riem_metric_tpca=="aff":
            riem_metric = spd.SPDMetricAffine(dim)
        else:
            raise ValueError("Unsupported riem_metric_tpca. Use log or aff.")

        # TangentPCA() does not accept a float like PCA() to automatically determine the # of components to keep
        tpca = TangentPCA(metric=riem_metric, n_components=tpca_nbr_components)

        train_sig_reduced = tpca.fit_transform(train_sig)
        complete_train_sig_reduced = tpca.transform(complete_train_sig)
        val_sig_reduced = tpca.transform(val_sig)
        test_sig_reduced = tpca.transform(test_sig)

        ytest = radar_dataset.ytest

        complete_ytrain_AD = radar_dataset.complete_ytrain_AD
        yval_AD = radar_dataset.yval_AD
        ytest_AD = radar_dataset.ytest_AD

        if shallow_name=="IF":
            clf = IsolationForest(n_estimators=100, max_samples='auto', bootstrap=True, random_state=seed).fit(train_sig_reduced)
        elif shallow_name=="LOF":
            clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True).fit(train_sig_reduced)
        elif shallow_name=="OCSVM":
            clf = OneClassSVM(nu=0.01).fit(train_sig_reduced)
        elif shallow_name=="RPO":
            clf = Random_Projection_Outlyingness(nproj=nbr_projections, unit_norm=unit_norm, device=device, estimator=estimator).fit(train_sig_reduced)
        elif shallow_name=="NN-PCA":
            clf = Norm_Negated_PCA(n_components=n_components_nntpca)
        else:
            raise ValueError("Unsupported shallow_name AD method. Use one of: IF, LOF, OCSVM, RPO.")

        y_score_complete_train = clf.decision_function(complete_train_sig_reduced)
        y_score_val = clf.decision_function(val_sig_reduced)
        y_score_test = clf.decision_function(test_sig_reduced)

        train_AUC = roc_auc_score(complete_ytrain_AD, y_score_complete_train*-1)
        valid_AUC = roc_auc_score(yval_AD, y_score_val*-1)
        test_AUC = roc_auc_score(ytest_AD, y_score_test*-1)

        ###################################################################

        # only one graph: no loss and AUC during training because no epochs;
        # also IsolationForest().decision_function() scoring function available after .fit() only
        violinplot = sns.violinplot(x=ytest.tolist(), y=y_score_test.tolist())
        fig = violinplot.get_figure()
        fig.savefig('./results/' + dataset + '/' + AD_name + '/{}/seed{}_normal{}.png'.format(date_string, seed, normal_cls))
        plt.close()

        ###################################################################

        # report -1 when parameter doesn't make sense for a shallow learning method
        results_dicts_list = store_comparison_results(results_dicts_list, train_AUC, valid_AUC, test_AUC, dataset,
                                                      AD_name, "shallow", "AD", AD_name, normal_cls, outlier_cls,
                                                      SAD_cls, SAD_ratio,
                                                      nbr_modes, -1, -1, -1, -1, -1, seed)

    results_df = pd.DataFrame(results_dicts_list)

    results_df = pd.concat([results_df], ignore_index=True)
    results_df_path = './results/' + dataset + '/' + AD_name + '/{}/experiment_results.csv'.format(date_string)
    results_df.to_csv(results_df_path)

    return results_df

if __name__ == '__main__':

    AD_name = str(sys.argv[1])
    shallow_name = str(sys.argv[2])
    dataset = str(sys.argv[3])
    nbr_modes = int(sys.argv[4])
    nbr_seeds = int(sys.argv[5])
    train_ratio = float(sys.argv[6])
    nbr_pulse_per_scan = int(sys.argv[7])
    nbr_targets = int(sys.argv[8])
    riem_metric_tpca = str(sys.argv[9])
    tpca_nbr_components = int(sys.argv[10])
    if shallow_name == "RPO":
        nbr_projections = int(sys.argv[11])
        estimator = str(sys.argv[12])
        unit_norm = bool(int(sys.argv[13]))
        experiment_name = str(sys.argv[14])
    elif shallow_name == "NN-PCA":
        experiment_name = str(sys.argv[11])
        n_components_nntpca = int(sys.argv[12])
    else:
        experiment_name = str(sys.argv[11])

    date_string = '{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.datetime.now())
    results_directory = './results/' + dataset + '/' + AD_name + '/{}/'.format(date_string)
    Path(results_directory).mkdir(parents=True, exist_ok=True)
    Path('./comparison_graphs/').mkdir(parents=True, exist_ok=True)

    results_df_path = './results/' + dataset + '/{}/experiment_results.csv'.format(date_string)

    if str(sys.argv[2]) == "RPO":
        results_df = shallow_tpca(AD_name=AD_name, shallow_name=shallow_name, dataset=dataset, nbr_modes=nbr_modes,nbr_seeds=nbr_seeds, train_ratio=train_ratio,
                                      nbr_pulse_per_scan=nbr_pulse_per_scan, nbr_targets=nbr_targets, riem_metric_tpca=riem_metric_tpca,
                                  tpca_nbr_components=tpca_nbr_components,
                                      date_string=date_string, nbr_projections=nbr_projections,
                                      estimator=estimator, unit_norm=unit_norm)
    if str(sys.argv[2]) == "NN-PCA":
        results_df = shallow_tpca(AD_name=AD_name, shallow_name=shallow_name, dataset=dataset, nbr_modes=nbr_modes,nbr_seeds=nbr_seeds, train_ratio=train_ratio,
                                      nbr_pulse_per_scan=nbr_pulse_per_scan, nbr_targets=nbr_targets, riem_metric_tpca=riem_metric_tpca,
                                  tpca_nbr_components=tpca_nbr_components,
                                      date_string=date_string, n_components_nntpca=n_components_nntpca)
    else:
        results_df = shallow_tpca(AD_name=AD_name, shallow_name=shallow_name, dataset=dataset, nbr_modes=nbr_modes,nbr_seeds=nbr_seeds, train_ratio=train_ratio,
                                      nbr_pulse_per_scan=nbr_pulse_per_scan, nbr_targets=nbr_targets,
                                  riem_metric_tpca=riem_metric_tpca, tpca_nbr_components=tpca_nbr_components, date_string=date_string)

    # add results to last_results_df.csv for comparison graphs generation
    global_results_df_file_path = 'comparison_graphs/last_results_df_{}.csv'.format(experiment_name)
    global_results_df = pd.read_csv(global_results_df_file_path, index_col=False)
    frames = [global_results_df, results_df]
    global_results_df = pd.concat(frames, ignore_index=True)
    global_results_df.to_csv(global_results_df_file_path, index=False)