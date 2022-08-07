import torch
import scipy.io as spio
import numpy as np
import pandas as pd
import torch.utils.data as data_utils
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import warnings

def init_comparison_df(experiment_name=""):

    df_file_path = 'comparison_graphs/last_results_df_{}.csv'.format(experiment_name)
    df = pd.DataFrame(columns=['dataset', 'arch', 'AD_name', 'loss_name', 'supervision', 'norm_classes', 'outl_classes', 'SAD_classes',
                               'SAD_ratio', 'auc_train', 'auc_valid', 'auc_test', 'nbr_modes', 'AUCepoch', 'nbr_epochs', 'lr_init',
                               'lr_milestones', 'batchsize', 'seed'])
    df.to_csv(df_file_path, index=False)

def store_comparison_results(results_dicts_list, auc_train, auc_valid, auc_test, dataset, AD_name, arch, supervision, loss, normal_cls,
                             outlier_cls, SAD_cls, SAD_ratio, nbr_modes, best_AUC_epoch, nbr_epochs, lr_init, lr_milestones, batchsize, seed):
    results_dicts_list.append({'dataset': dataset,  'arch': arch, 'AD_name': AD_name, 'loss_name': loss, 'supervision': supervision,
                               'norm_classes': normal_cls, 'outl_classes': outlier_cls, 'SAD_classes': SAD_cls, 'SAD_ratio': SAD_ratio,
                               'auc_train': auc_train, 'auc_valid': auc_valid, 'auc_test': auc_test,
                               'nbr_modes': nbr_modes, 'AUCepoch': best_AUC_epoch, 'nbr_epochs': nbr_epochs,
                               'lr_init': lr_init, 'lr_milestones': lr_milestones, 'batchsize': batchsize,
                               'seed': seed})
    return results_dicts_list

def get_train_set(train_sets, ytrain, normal_cls, SAD_cls=[], SAD=False, SAD_ratio=0.01):
    """
    SAD samples are always there, but not necessarily used in the training loop (cf. training.py and loss computation).
    Since the SAD samples are taken from otherwise discarded training set anomalous samples, this is not an issue
    for unsupervised AD/OOD experiments.
    """
    if SAD:
        selected_cls = []
        yselected_cls = []
        selected_cls_SAD = []
        yselected_cls_SAD = []
        size_normal = 0
        for norm_cls in normal_cls:
            size_normal += train_sets[norm_cls - 1].size()[0]
            selected_cls.append(train_sets[norm_cls - 1])
            yselected_cls.append(ytrain[norm_cls - 1])
        for anom_cls in SAD_cls:
            shuffled_train_idx = np.array(range(train_sets[anom_cls - 1].size()[0]))
            np.random.shuffle(shuffled_train_idx)
            selected_idx_SAD = shuffled_train_idx[:int((SAD_ratio*size_normal)//len(SAD_cls))] # SAD_ratio in total, so divide # of anomalous training samples between all SAD classes
            selected_cls_SAD.append(train_sets[anom_cls - 1][selected_idx_SAD])
            yselected_cls_SAD.append(ytrain[anom_cls - 1][selected_idx_SAD])
        return torch.cat(selected_cls), torch.cat(yselected_cls), torch.cat(selected_cls_SAD), torch.cat(yselected_cls_SAD)
    else: # need to keep that for the definition of train_spds and train_spdsv2, would be cleaner to include the latter in the above scheme though
        selected_cls = []
        yselected_cls = []
        for norm_cls in normal_cls:
            selected_cls.append(train_sets[norm_cls - 1])
            yselected_cls.append(ytrain[norm_cls - 1])
        return torch.cat(selected_cls), torch.cat(yselected_cls)

def convert_labels(labels, normal_cls):
    # 1 for anomalies, and 0 for normal samples as in https://github.com/lukasruff/Deep-SVDD/blob/master/src/isoForest.py
    scores_labels = [0 if y in normal_cls else 1 for y in labels]
    return scores_labels

def get_train_test_valid(simdata,train_ratio,cls_idx):
    nbr_train = int(train_ratio*simdata.shape[0])
    nbr_val = int((simdata.shape[0]-nbr_train)/2) # take half of the residual samples
    train = simdata[:nbr_train]
    val = simdata[nbr_train:nbr_train+nbr_val]
    test = simdata[nbr_train+nbr_val:]
    ytrain = torch.ones(train.shape[0])*cls_idx
    yval = torch.ones(val.shape[0])*cls_idx
    ytest = torch.ones(test.shape[0])*cls_idx
    return train,ytrain,val,yval,test,ytest

def init_center_c(device, train_loader, net, eps=0.1):
    """
    Initialize hypersphere center c as the mean from an initial forward pass on the data.
    Source: https://github.com/lukasruff/Deep-SVDD-PyTorch/blob/master/src/optim/deepSVDD_trainer.py
    """
    n_samples = 0
    c = torch.zeros(net.rep_dim, device=device)

    net.eval()
    with torch.no_grad():
        for data in train_loader:
            # get the inputs of the batch
            inputs, _ = data
            inputs = inputs.to(device)
            outputs = net(inputs)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c

def init_ssl_sad_centers_c(device, train_loader, net, eps=0.1, sad_ssl_heuristic=False):
    """
    Initialize hypersphere center c as the mean from an initial forward pass on the data, one centroid per training data
    is computed to provide all the latent reference points experimented on. Experiment can be conducted without using
    all the centroids computed here.
    Source of inspiration: https://github.com/lukasruff/Deep-SVDD-PyTorch/blob/master/src/optim/deepSVDD_trainer.py
    """
    n_samples = 0
    n_samples_SSL = 0
    n_samples_SAD = 0
    c = torch.zeros((net.rep_dim, 3), device=device)

    net.eval()
    with torch.no_grad():
        for data in train_loader:
            # get the inputs of the batch
            inputs, labels = data
            inputs = inputs.to(device)
            outputs = net(inputs)
            c[:, 0] += torch.sum(outputs[labels[:,1] == 0], dim=0) # normal training samples with label 0
            n_samples += outputs[labels[:, 1] == 0].size()[0]
            c[:, 1] += torch.sum(outputs[labels[:,1] == 1], dim=0) # rotated normal/SSL training samples with label 1
            n_samples_SSL += outputs[labels[:, 1] == 1].size()[0]
            c[:, 2] += torch.sum(outputs[labels[:,1] == 2], dim=0) # labeled anomalies/SAD training samples with label 2
            n_samples_SAD += outputs[labels[:, 1] == 2].size()[0]

    c[:,0] /= n_samples
    if n_samples_SAD != 0: # check if training data includes SAD samples, cf. "supervision" hyperparameter
        if sad_ssl_heuristic:
            c[:, 2] = 2 * c[:, 0]
        else:
            c[:, 2] /= n_samples_SAD
    if n_samples_SSL != 0: # check if training data includes SSL samples, cf. "supervision" hyperparameter
        if sad_ssl_heuristic:
            c[:, 1] = 3 * c[:,0]
        else:
            c[:, 1] /= n_samples_SSL

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    if torch.equal(c[:,0], c[:,1]):
        c[:, 1] = c[:, 1] * 2 # push away in every dimension, Deep SVDD heuristic for normality centroid unchanged
        # raise ValueError("Equal latent centroids for SSL and normal samples.")
        print(("Equal latent centroids for SSL and normal samples."))
    if torch.equal(c[:,0], c[:,2]):
        c[:, 2] = c[:, 2] * 3  # push away in every dimension, Deep SVDD heuristic for normality centroid unchanged
        # raise ValueError("Equal latent centroids for SAD and normal samples.")
        print("Equal latent centroids for SAD and normal samples.")
    return c

def init_centers_c_kmeans_MSVDD(device, train_loader, x_train, net, nbr_centroids=3, batch_size=128, seed=1):
    """
    DeepMSVDD paper implementation.
    """

    net.eval()
    with torch.no_grad():
        for data, targets in train_loader:
            inputs, labels = data.to(device), targets.to(device)
            outputs = net(inputs)
            try:
                complete_outputs = torch.cat((complete_outputs, outputs), dim=0)
            except UnboundLocalError:
                complete_outputs = outputs

    complete_outputs = complete_outputs.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=nbr_centroids, random_state=seed).fit(complete_outputs)

    # recreate train dataloader in which labels are the indexes of the associated centers
    train_set = TensorDataset(x_train, torch.from_numpy(kmeans.labels_))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return torch.from_numpy(kmeans.cluster_centers_).to(device), train_loader

def filter_centers_DMSVDD(hyperspheres_center, radius):
    return hyperspheres_center[radius > 0.0]

def update_radius_DMSVDD(hyperspheres_center, nu, train_loader, net, device):
    """
    https://epubs.siam.org/doi/pdf/10.1137/1.9781611976236.13
    https://github.com/zghafoori/Deep-Multi-Sphere-SVDD/blob/670ba3c7604347d249758b49f1865c51616c6a3c/src/opt/sgd/train.py
    """

    net.eval()
    with torch.no_grad():
        for data, targets in train_loader:
            inputs, labels = data.to(device), targets.to(device)
            outputs = net(inputs)
            try:
                complete_outputs = torch.cat((complete_outputs, outputs), dim=0)
            except UnboundLocalError:
                complete_outputs = outputs

    # now that populated-enough centers are updated, update the associated radius
    dist_to_centers = torch.cdist(complete_outputs, hyperspheres_center)
    dist_to_best_center, best_center_idx = torch.min(dist_to_centers, dim=1)
    centers, centers_occurrence = torch.unique(best_center_idx, return_counts=True)

    # handle centers with zero population that disappeared due to the torch.unique()
    new_centers_occurrence = torch.zeros((hyperspheres_center.size()[0],)).long().to(device)
    new_centers_occurrence[centers] = centers_occurrence
    centers_occurrence = new_centers_occurrence

    good_centers = centers_occurrence > nu * torch.max(centers_occurrence)

    radius = torch.zeros((hyperspheres_center.size()[0],)).to(device)
    for center_idx in range(hyperspheres_center.size()[0]):
        try:
            radius[center_idx] = torch.quantile(dist_to_centers[best_center_idx == center_idx, center_idx],q=1-nu)
        except RuntimeError: # handle centroids without samples, which can't yield any quantile
            radius[center_idx] = 0.0

    radius[~good_centers] = 0.0 # centroids with samples but not enough

    net.train()
    return radius

# def get_train_test_valid(simdata,train_ratio,cls_idx):
#     nbr_train = int(train_ratio*simdata.shape[0])
#     nbr_val = int((simdata.shape[0]-nbr_train)/2) # take half of the residual samples
#     train = simdata[:nbr_train]
#     val = simdata[nbr_train:nbr_train+nbr_val]
#     test = simdata[nbr_train+nbr_val:]
#     ytrain = torch.ones(train.shape[0])*cls_idx
#     yval = torch.ones(val.shape[0])*cls_idx
#     ytest = torch.ones(test.shape[0])*cls_idx
#     return train,ytrain,val,yval,test,ytest

def min_max_normalize(input_tensor, tensor_min, tensor_max):
    tensor_min_max_normalized = (input_tensor - tensor_min) / (tensor_max - tensor_min)
    return tensor_min_max_normalized

class simulated_radar_dataset():
    def __init__(self, train_ratio, normal_cls, SAD_cls, SAD_ratio, batchsize, nbr_targets, nbr_pulse_per_scan):
        self.train_ratio = train_ratio
        self.normal_cls = normal_cls
        self.SAD_cls = SAD_cls
        self.SAD_ratio = SAD_ratio
        self.batchsize = batchsize
        self.nbr_targets = nbr_targets
        self.nbr_pulse_per_scan = nbr_pulse_per_scan

    def get_dataloaders(self, generator=None, cov_regularization=1):

        tgt_type1 = spio.loadmat('./data/helicopters_{}_1_{}.mat'.format(self.nbr_targets, self.nbr_pulse_per_scan), simplify_cells=True)['tgts']
        tgt_type2 = spio.loadmat('./data/helicopters_{}_2_{}.mat'.format(self.nbr_targets, self.nbr_pulse_per_scan), simplify_cells=True)['tgts']
        tgt_type3 = spio.loadmat('./data/helicopters_{}_4_{}.mat'.format(self.nbr_targets, self.nbr_pulse_per_scan), simplify_cells=True)['tgts']
        tgt_type4 = spio.loadmat('./data/helicopters_{}_6_{}.mat'.format(self.nbr_targets, self.nbr_pulse_per_scan), simplify_cells=True)['tgts']
        
        nbr_targets = len(tgt_type1) # equal for all targets types
        nbr_scans = tgt_type1[0]['signature'].shape[0]
        nbr_pulsesperscan = tgt_type1[0]['signature'].shape[1] # defines the number of features / frequency bins for the SPD representation
        
        sp1 = torch.zeros((nbr_targets, nbr_scans, nbr_pulsesperscan))
        sp2 = torch.zeros((nbr_targets, nbr_scans, nbr_pulsesperscan))
        sp3 = torch.zeros((nbr_targets, nbr_scans, nbr_pulsesperscan))
        sp4 = torch.zeros((nbr_targets, nbr_scans, nbr_pulsesperscan))
        
        spd1 = torch.zeros((nbr_targets, nbr_pulsesperscan, nbr_pulsesperscan))
        spd2 = torch.zeros((nbr_targets, nbr_pulsesperscan, nbr_pulsesperscan))
        spd3 = torch.zeros((nbr_targets, nbr_pulsesperscan, nbr_pulsesperscan))
        spd4 = torch.zeros((nbr_targets, nbr_pulsesperscan, nbr_pulsesperscan))

        for tgt_idx in range(nbr_targets):
            sp1[tgt_idx] = torch.from_numpy(tgt_type1[tgt_idx]['signature'][:,:,0])
            sp2[tgt_idx] = torch.from_numpy(tgt_type2[tgt_idx]['signature'][:,:,0])
            sp3[tgt_idx] = torch.from_numpy(tgt_type3[tgt_idx]['signature'][:,:,0])
            sp4[tgt_idx] = torch.from_numpy(tgt_type4[tgt_idx]['signature'][:,:,0])

            spd1[tgt_idx] = torch.from_numpy(tgt_type1[tgt_idx]['sigcov'])
            spd2[tgt_idx] = torch.from_numpy(tgt_type2[tgt_idx]['sigcov'])
            spd3[tgt_idx] = torch.from_numpy(tgt_type3[tgt_idx]['sigcov'])
            spd4[tgt_idx] = torch.from_numpy(tgt_type4[tgt_idx]['sigcov'])

        max_sp = torch.max(torch.cat((10*torch.log(sp1), 10*torch.log(sp2), 10*torch.log(sp3), 10*torch.log(sp4))))
        min_sp = torch.min(torch.cat((10*torch.log(sp1), 10*torch.log(sp2), 10*torch.log(sp3), 10*torch.log(sp4))))
        max_spd = torch.max(torch.cat((spd1, spd2, spd3, spd4)))
        min_spd = torch.min(torch.cat((spd1, spd2, spd3, spd4)))

        sp1 = min_max_normalize(10*torch.log(sp1), min_sp, max_sp)
        sp2 = min_max_normalize(10*torch.log(sp2), min_sp, max_sp)
        sp3 = min_max_normalize(10*torch.log(sp3), min_sp, max_sp)
        sp4 = min_max_normalize(10*torch.log(sp4), min_sp, max_sp)

        # use quantile per sample, not global quantile
        quantile_coef = 0.85
        quantile_sp1 = torch.quantile(sp1.view(sp1.size()[0],-1), q=quantile_coef, dim=1)
        quantile_sp2 = torch.quantile(sp2.view(sp2.size()[0],-1), q=quantile_coef, dim=1)
        quantile_sp3 = torch.quantile(sp3.view(sp3.size()[0],-1), q=quantile_coef, dim=1)
        quantile_sp4 = torch.quantile(sp4.view(sp4.size()[0],-1), q=quantile_coef, dim=1)

        quantile_sp1 = torch.repeat_interleave(quantile_sp1.unsqueeze(1), sp1.size(1), dim=1)
        quantile_sp2 = torch.repeat_interleave(quantile_sp2.unsqueeze(1), sp2.size(1), dim=1)
        quantile_sp3 = torch.repeat_interleave(quantile_sp3.unsqueeze(1), sp3.size(1), dim=1)
        quantile_sp4 = torch.repeat_interleave(quantile_sp4.unsqueeze(1), sp4.size(1), dim=1)

        quantile_sp1 = torch.repeat_interleave(quantile_sp1.unsqueeze(2), sp1.size(2), dim=2)
        quantile_sp2 = torch.repeat_interleave(quantile_sp2.unsqueeze(2), sp2.size(2), dim=2)
        quantile_sp3 = torch.repeat_interleave(quantile_sp3.unsqueeze(2), sp3.size(2), dim=2)
        quantile_sp4 = torch.repeat_interleave(quantile_sp4.unsqueeze(2), sp4.size(2), dim=2)

        sp1_activespeeds = torch.where(sp1 > quantile_sp1, torch.ones_like(sp1), torch.zeros_like(sp1))
        sp2_activespeeds = torch.where(sp2 > quantile_sp2, torch.ones_like(sp2), torch.zeros_like(sp2))
        sp3_activespeeds = torch.where(sp3 > quantile_sp3, torch.ones_like(sp3), torch.zeros_like(sp3))
        sp4_activespeeds = torch.where(sp4 > quantile_sp4, torch.ones_like(sp4), torch.zeros_like(sp4))

        sp1_activespeeds = torch.repeat_interleave((torch.sum(sp1_activespeeds, dim=1) > 0).unsqueeze(dim=1), sp1_activespeeds.size()[1], dim=1)
        sp2_activespeeds = torch.repeat_interleave((torch.sum(sp2_activespeeds, dim=1) > 0).unsqueeze(dim=1), sp2_activespeeds.size()[1], dim=1)
        sp3_activespeeds = torch.repeat_interleave((torch.sum(sp3_activespeeds, dim=1) > 0).unsqueeze(dim=1), sp3_activespeeds.size()[1], dim=1)
        sp4_activespeeds = torch.repeat_interleave((torch.sum(sp4_activespeeds, dim=1) > 0).unsqueeze(dim=1), sp4_activespeeds.size()[1], dim=1)

        sp1 = torch.where(sp1_activespeeds.type(torch.bool), sp1, torch.zeros_like(sp1))
        sp2 = torch.where(sp2_activespeeds.type(torch.bool), sp2, torch.zeros_like(sp2))
        sp3 = torch.where(sp3_activespeeds.type(torch.bool), sp3, torch.zeros_like(sp3))
        sp4 = torch.where(sp4_activespeeds.type(torch.bool), sp4, torch.zeros_like(sp4))

        train_sp1,ytrain1,val_sp1,yval1,test_sp1,ytest1 = get_train_test_valid(sp1,self.train_ratio,1)
        train_spd1,_,val_spd1,_,test_spd1,_ = get_train_test_valid(spd1,self.train_ratio,1)

        train_sp2,ytrain2,val_sp2,yval2,test_sp2,ytest2 = get_train_test_valid(sp2,self.train_ratio,2)
        train_spd2,_,val_spd2,_,test_spd2,_ = get_train_test_valid(spd2,self.train_ratio,2)

        train_sp3,ytrain3,val_sp3,yval3,test_sp3,ytest3 = get_train_test_valid(sp3,self.train_ratio,3)
        train_spd3,_,val_spd3,_,test_spd3,_ = get_train_test_valid(spd3,self.train_ratio,3)

        train_sp4,ytrain4,val_sp4,yval4,test_sp4,ytest4 = get_train_test_valid(sp4,self.train_ratio,4)
        train_spd4,_,val_spd4,_,test_spd4,_ = get_train_test_valid(spd4,self.train_ratio,4)

        complete_train_sps = torch.cat((train_sp1,train_sp2,train_sp3,train_sp4))
        complete_train_spds = torch.cat((train_spd1,train_spd2,train_spd3,train_spd4))
        self.complete_ytrain = torch.cat((ytrain1,ytrain2,ytrain3,ytrain4))

        val_sps = torch.cat((val_sp1,val_sp2,val_sp3,val_sp4))
        val_spds = torch.cat((val_spd1,val_spd2,val_spd3,val_spd4))
        self.yval = torch.cat((yval1,yval2,yval3,yval4))

        test_sps = torch.cat((test_sp1,test_sp2,test_sp3,test_sp4))
        test_spds = torch.cat((test_spd1,test_spd2,test_spd3,test_spd4))
        self.ytest = torch.cat((ytest1,ytest2,ytest3,ytest4))

        self.complete_ytrain_AD = convert_labels(self.complete_ytrain, self.normal_cls)
        self.yval_AD = convert_labels(self.yval, self.normal_cls)
        self.ytest_AD = convert_labels(self.ytest, self.normal_cls)

        #############
        ### TRAIN ###
        #############

        ############# SPDs datasets: no SAD or SSL samples, training is only made of normal training samples

        train_spds, self.ytrain_spds = get_train_set((train_spd1, train_spd2, train_spd3, train_spd4), (ytrain1, ytrain2, ytrain3, ytrain4), self.normal_cls)
        shuffled_train_spds_idx = np.array(range(train_spds.size()[0]))
        np.random.shuffle(shuffled_train_spds_idx)
        train_spds = train_spds[shuffled_train_spds_idx]
        self.ytrain_spds = self.ytrain_spds[shuffled_train_spds_idx]

        ############# SPs datasets: SAD and SSL samples included, with double-column labels to also provide nothing/SSL/SAD labels

        train_sps_norm, ytrain_sps_norm, train_sps_anom, ytrain_sps_anom = get_train_set((train_sp1, train_sp2, train_sp3, train_sp4), (ytrain1, ytrain2, ytrain3, ytrain4),
                                                                                         self.normal_cls, self.SAD_cls, SAD=True, SAD_ratio=self.SAD_ratio)
        train_sps_norm_t = torch.transpose(train_sps_norm, 1, 2)
        train_sps_norm_2D_sad_ssl = torch.cat((train_sps_norm, train_sps_norm_t, train_sps_anom))
        ytrain_sad_ssl1 = torch.cat((ytrain_sps_norm, ytrain_sps_norm, ytrain_sps_anom))  # original labels don't change and are deduplicated
        ytrain_sad_ssl2 = torch.cat((torch.zeros_like(ytrain_sps_norm), torch.ones_like(ytrain_sps_norm), 2*torch.ones_like(ytrain_sps_anom)))  # 0 for not rotated, 1 for rotated/SSL sample, 2 for SAD/labelled anomaly sample
        self.ytrain_sps_norm_2D_sad_ssl = torch.cat((ytrain_sad_ssl1.unsqueeze(dim=1), ytrain_sad_ssl2.unsqueeze(dim=1)), dim=1) # 2D labels: first column is class label, second column is nothing, SSL or SAD label

        # shuffle train samples before dataloader creation, since shallow methods don't use the dataloader version of the dataset, not sure this is actually required but just in case
        shuffled_train_sps_idx = np.array(range(train_sps_norm_2D_sad_ssl.size()[0]))
        np.random.shuffle(shuffled_train_sps_idx)
        train_sps_norm_2D_sad_ssl = train_sps_norm_2D_sad_ssl[shuffled_train_sps_idx]
        self.ytrain_sps_norm_2D_sad_ssl = self.ytrain_sps_norm_2D_sad_ssl[shuffled_train_sps_idx]

        # mask to use only relevant samples to create dataloaders with batches excluding any sample useless for training. No mask for the full setup SSL+SAD which is already available.
        train_sps_mask = self.ytrain_sps_norm_2D_sad_ssl[:, 1] == 0 # no SAD, no SSL
        train_sps_ssl_mask = torch.logical_or(self.ytrain_sps_norm_2D_sad_ssl[:, 1] == 0,self.ytrain_sps_norm_2D_sad_ssl[:, 1] == 1) # no SAD samples for training
        train_sps_sad_mask = torch.logical_or(self.ytrain_sps_norm_2D_sad_ssl[:, 1] == 0,self.ytrain_sps_norm_2D_sad_ssl[:, 1] == 2) # no SSL samples for training

        self.train_sps_norm_2D = train_sps_norm_2D_sad_ssl[train_sps_mask].float()
        self.train_dataset_sps_norm_2D = data_utils.TensorDataset(self.train_sps_norm_2D, self.ytrain_sps_norm_2D_sad_ssl[train_sps_mask])
        self.train_loader_sps_norm_2D = data_utils.DataLoader(self.train_dataset_sps_norm_2D, batch_size=self.batchsize, shuffle=True, generator=generator)

        self.train_sps_norm_ssl_2D = train_sps_norm_2D_sad_ssl[train_sps_ssl_mask].float()
        self.train_dataset_sps_norm_ssl_2D = data_utils.TensorDataset(self.train_sps_norm_ssl_2D, self.ytrain_sps_norm_2D_sad_ssl[train_sps_ssl_mask])
        self.train_loader_sps_norm_ssl_2D = data_utils.DataLoader(self.train_dataset_sps_norm_ssl_2D, batch_size=self.batchsize, shuffle=True, generator=generator)

        self.train_sps_norm_sad_2D = train_sps_norm_2D_sad_ssl[train_sps_sad_mask].float()
        self.train_dataset_sps_norm_sad_2D = data_utils.TensorDataset(self.train_sps_norm_sad_2D, self.ytrain_sps_norm_2D_sad_ssl[train_sps_sad_mask])
        self.train_loader_sps_norm_sad_2D = data_utils.DataLoader(self.train_dataset_sps_norm_sad_2D, batch_size=self.batchsize, shuffle=True, generator=generator)

        self.train_sps_norm_ssl_sad_2D = train_sps_norm_2D_sad_ssl.float()
        self.train_dataset_sps_norm_ssl_sad_2D = data_utils.TensorDataset(self.train_sps_norm_ssl_sad_2D, self.ytrain_sps_norm_2D_sad_ssl)
        self.train_loader_sps_norm_ssl_sad_2D = data_utils.DataLoader(self.train_dataset_sps_norm_ssl_sad_2D, batch_size=self.batchsize, shuffle=True, generator=generator)

        # upper triangular 1D features of covariance SPD representation
        self.train_spds_norm_1Dut = min_max_normalize(train_spds[:,torch.triu_indices(nbr_pulsesperscan,nbr_pulsesperscan)[0],torch.triu_indices(nbr_pulsesperscan,nbr_pulsesperscan)[1]], min_spd, max_spd).float()
        self.train_dataset_spds_norm_1Dut = data_utils.TensorDataset(self.train_spds_norm_1Dut, self.ytrain_spds)
        self.train_loader_spds_norm_1Dut = data_utils.DataLoader(self.train_dataset_spds_norm_1Dut, batch_size=self.batchsize, shuffle=True, generator=generator)

        self.train_spds_norm_2D = min_max_normalize(train_spds, min_spd, max_spd).float()
        self.train_dataset_spds_norm_2D = data_utils.TensorDataset(self.train_spds_norm_2D, self.ytrain_spds)
        self.train_loader_spds_norm_2D = data_utils.DataLoader(self.train_dataset_spds_norm_2D, batch_size=self.batchsize, shuffle=True, generator=generator)

        # not normalized 2D covariance SPD representation available to train SPD neural network
        reg_train = torch.from_numpy(np.repeat(np.expand_dims(np.eye(train_spds[0].size()[1]), axis=0), repeats=train_spds.size()[0], axis=0))*cov_regularization
        self.train_spds_2D = (train_spds+reg_train).float()
        self.train_dataset_spds_2D = data_utils.TensorDataset(self.train_spds_2D, self.ytrain_spds)
        self.train_loader_spds_2D = data_utils.DataLoader(self.train_dataset_spds_2D, batch_size=self.batchsize, shuffle=True, generator=generator)

        ######################
        ### COMPLETE TRAIN ###
        ######################

        self.complete_train_sps_norm_2D = complete_train_sps.float()
        self.complete_train_dataset_sps_norm_2D = data_utils.TensorDataset(self.complete_train_sps_norm_2D, self.complete_ytrain)
        self.complete_train_loader_sps_norm_2D = data_utils.DataLoader(self.complete_train_dataset_sps_norm_2D, batch_size=self.batchsize, shuffle=True, generator=generator)
        
        self.complete_train_sps_norm_1D = complete_train_sps.view(complete_train_sps.size()[0],-1).float()
        self.complete_train_dataset_sps_norm_1D = data_utils.TensorDataset(self.complete_train_sps_norm_1D, self.complete_ytrain)
        self.complete_train_loader_sps_1D = data_utils.DataLoader(self.complete_train_dataset_sps_norm_1D, batch_size=self.batchsize, shuffle=True, generator=generator)

        self.complete_train_spds_norm_1Dut = min_max_normalize(complete_train_spds[:,torch.triu_indices(nbr_pulsesperscan,nbr_pulsesperscan)[0],torch.triu_indices(nbr_pulsesperscan,nbr_pulsesperscan)[1]], min_spd, max_spd).float()
        self.complete_train_dataset_spds_norm_1Dut = data_utils.TensorDataset(self.complete_train_spds_norm_1Dut, self.complete_ytrain)
        self.complete_train_loader_spds_norm_1Dut = data_utils.DataLoader(self.complete_train_dataset_spds_norm_1Dut, batch_size=self.batchsize, shuffle=True, generator=generator)

        self.complete_train_spds_norm_2D = min_max_normalize(complete_train_spds, min_spd, max_spd).float()
        self.complete_train_dataset_spds_norm_2D = data_utils.TensorDataset(self.complete_train_spds_norm_2D, self.complete_ytrain)
        self.complete_train_loader_spds_norm_2D = data_utils.DataLoader(self.complete_train_dataset_spds_norm_2D, batch_size=self.batchsize, shuffle=True, generator=generator)

        # not normalized 2D covariance SPD representation available to train SPD neural network
        reg_complete_train = torch.from_numpy(np.repeat(np.expand_dims(np.eye(complete_train_spds[0].size()[1]), axis=0), repeats=complete_train_spds.size()[0], axis=0))*cov_regularization
        self.complete_train_spds_2D = (complete_train_spds+reg_complete_train).float()
        self.complete_train_dataset_spds_2D = data_utils.TensorDataset(self.complete_train_spds_2D, self.complete_ytrain)
        self.complete_train_loader_spds_2D = data_utils.DataLoader(self.complete_train_dataset_spds_2D, batch_size=self.batchsize, shuffle=True, generator=generator)

        #############
        ### VALID ###
        #############

        self.val_sps_norm_2D = val_sps.float()
        self.val_dataset_sps_norm_2D = data_utils.TensorDataset(self.val_sps_norm_2D, self.yval)
        self.val_loader_sps_norm_2D = data_utils.DataLoader(self.val_dataset_sps_norm_2D, batch_size=self.batchsize, shuffle=True, generator=generator)
        
        self.val_sps_norm_1D = val_sps.view(val_sps.size()[0],-1).float()
        self.val_dataset_sps_norm_1D = data_utils.TensorDataset(self.val_sps_norm_1D, self.yval)
        self.val_loader_norm_1D = data_utils.DataLoader(self.val_dataset_sps_norm_1D, batch_size=self.batchsize, shuffle=True, generator=generator)

        self.val_spds_norm_1Dut = min_max_normalize(val_spds[:,torch.triu_indices(nbr_pulsesperscan,nbr_pulsesperscan)[0],torch.triu_indices(nbr_pulsesperscan,nbr_pulsesperscan)[1]], min_spd, max_spd).float()
        self.val_dataset_spds_norm_1Dut = data_utils.TensorDataset(self.val_spds_norm_1Dut, self.yval)
        self.val_loader_spds_norm_1Dut = data_utils.DataLoader(self.val_dataset_spds_norm_1Dut, batch_size=self.batchsize, shuffle=True, generator=generator)

        self.val_spds_norm_2D = min_max_normalize(val_spds, min_spd, max_spd).float()
        self.val_dataset_spds_norm_2D = data_utils.TensorDataset(self.val_spds_norm_2D, self.yval)
        self.val_loader_spds_norm_2D = data_utils.DataLoader(self.val_dataset_spds_norm_2D, batch_size=self.batchsize, shuffle=True, generator=generator)

        # not normalized 2D covariance SPD representation available to train SPD neural network
        reg_val = torch.from_numpy(np.repeat(np.expand_dims(np.eye(val_spds[0].size()[1]), axis=0), repeats=val_spds.size()[0], axis=0))*cov_regularization
        self.val_spds_2D = (val_spds+reg_val).float()
        self.val_dataset_spds_2D = data_utils.TensorDataset(self.val_spds_2D, self.yval)
        self.val_loader_spds_2D = data_utils.DataLoader(self.val_dataset_spds_2D, batch_size=self.batchsize, shuffle=True, generator=generator)

        ############
        ### TEST ###
        ############

        self.test_sps_norm_2D = test_sps.float()
        self.test_dataset_sps_norm_2D = data_utils.TensorDataset(self.test_sps_norm_2D, self.ytest)
        self.test_loader_sps_norm_2D = data_utils.DataLoader(self.test_dataset_sps_norm_2D, batch_size=self.batchsize, shuffle=True, generator=generator)
        
        self.test_sps_norm_1D = test_sps.view(test_sps.size()[0],-1).float()
        self.test_dataset_sps_norm_1D = data_utils.TensorDataset(self.test_sps_norm_1D, self.ytest)
        self.test_loader_sps_norm_1D = data_utils.DataLoader(self.test_dataset_sps_norm_1D, batch_size=self.batchsize, shuffle=True, generator=generator)

        self.test_spds_norm_1Dut = min_max_normalize(test_spds[:,torch.triu_indices(nbr_pulsesperscan,nbr_pulsesperscan)[0],torch.triu_indices(nbr_pulsesperscan,nbr_pulsesperscan)[1]], min_spd, max_spd).float()
        self.test_dataset_spds_norm_1Dut = data_utils.TensorDataset(self.test_spds_norm_1Dut, self.ytest)
        self.test_loader_spds_norm_1Dut = data_utils.DataLoader(self.test_dataset_spds_norm_1Dut, batch_size=self.batchsize, shuffle=True, generator=generator)

        self.test_spds_norm_2D = min_max_normalize(test_spds, min_spd, max_spd).float()
        self.test_dataset_spds_norm_2D = data_utils.TensorDataset(self.test_spds_norm_2D, self.ytest)
        self.test_loader_spds_norm_2D = data_utils.DataLoader(self.test_dataset_spds_norm_2D, batch_size=self.batchsize, shuffle=True, generator=generator)

        # not normalized 2D covariance SPD representation available to train SPD neural network
        reg_test = torch.from_numpy(np.repeat(np.expand_dims(np.eye(test_spds[0].size()[1]), axis=0), repeats=test_spds.size()[0], axis=0))*cov_regularization
        self.test_spds_2D = (test_spds+reg_test).float()
        self.test_dataset_spds_2D = data_utils.TensorDataset(self.test_spds_2D, self.ytest)
        self.test_loader_spds_2D = data_utils.DataLoader(self.test_dataset_spds_2D, batch_size=self.batchsize, shuffle=True, generator=generator)

def get_2D_latent_representations_trainloader(dataloader, net, device, title, figure_path, figure_name, plot_centroids=False, centroids=[], supervision="AD"):
    """
    - Projects dataloader samples with neural network in 2D representations.
    - Returns matplotlib figure object.
    - trainloader dedicated function since only the trainloader has 2D labels tensors in order to handle the double labeling (class labels in first column, training sample/SSL sample/SAD sample in second column).

    TODO: delete SSL (rotated normal samples) samples from the classes plot, at the moment they are plotted with their original dataset class color (but their latent distribution is revealed by the supervision classes plot).
    """

    # ignore FutureWarning of sklearn
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        net.eval()
        complete_outputs_init = False
        with torch.no_grad():
            for data, targets in dataloader:
                inputs, labels = data.to(device), targets.to(device)
                outputs = net(inputs)
                if not complete_outputs_init:
                    complete_outputs = outputs
                    complete_class_labels = labels[:, 0]  # labels[:,0] are class labels, labels[:,1] are supervision ie. vanilla/SSL/SAD samples labels
                    complete_supervision_labels = labels[:, 1]
                    complete_outputs_init = True
                else:
                    complete_outputs = torch.cat((complete_outputs, outputs), dim=0)
                    complete_class_labels = torch.cat((complete_class_labels, labels[:, 0]), dim=0)
                    complete_supervision_labels = torch.cat((complete_supervision_labels, labels[:, 1]), dim=0)

        if plot_centroids:
            if supervision=="AD":
                valid_centroids_idx = torch.LongTensor([0]).to(device)
                centroids = torch.index_select(centroids, 1, valid_centroids_idx)
            elif supervision=="SSL":
                valid_centroids_idx = torch.LongTensor([0, 1]).to(device)
                centroids = torch.index_select(centroids, 1, valid_centroids_idx)
            elif supervision=="SAD":
                valid_centroids_idx = torch.LongTensor([0, 2]).to(device)
                centroids = torch.index_select(centroids, 1, valid_centroids_idx)
            elif supervision=="SAD+SSL":
                valid_centroids_idx = torch.LongTensor([0, 1, 2]).to(device)
                centroids = torch.index_select(centroids, 1, valid_centroids_idx)
            else:
                raise ValueError("Supervision {} not implemented !".format(supervision))
            centroids = torch.transpose(centroids, 0, 1).to(device)
            complete_outputs = torch.cat((complete_outputs, centroids), dim=0)
            complete_class_labels = torch.cat((complete_class_labels, torch.ones((centroids.size()[0],)).to(device)*5), dim=0) # 5 is the label for centroids in this function
            complete_supervision_labels = torch.cat((complete_supervision_labels,torch.ones((centroids.size()[0],)).to(device)*5), dim=0)

        complete_outputs = complete_outputs.cpu().detach().numpy()
        complete_class_labels = complete_class_labels.cpu().detach().numpy()
        complete_supervision_labels = complete_supervision_labels.cpu().detach().numpy()

        # fit projections methods using complete data, all classes mixed together
        latent_2D_TSNE = TSNE(n_components=2).fit_transform(complete_outputs)
        latent_2D_PCA = PCA(n_components=2).fit_transform(complete_outputs)

        x_cls1_TSNE = latent_2D_TSNE[complete_class_labels == 1]
        x_cls2_TSNE = latent_2D_TSNE[complete_class_labels == 2]
        x_cls3_TSNE = latent_2D_TSNE[complete_class_labels == 3]
        x_cls4_TSNE = latent_2D_TSNE[complete_class_labels == 4]
        if plot_centroids:
            x_cls5_TSNE = latent_2D_TSNE[complete_class_labels == 5]

        x_cls1_PCA = latent_2D_PCA[complete_class_labels == 1]
        x_cls2_PCA = latent_2D_PCA[complete_class_labels == 2]
        x_cls3_PCA = latent_2D_PCA[complete_class_labels == 3]
        x_cls4_PCA = latent_2D_PCA[complete_class_labels == 4]
        if plot_centroids:
            x_cls5_PCA = latent_2D_PCA[complete_class_labels == 5]

        x_supervisioncls0_TSNE = latent_2D_TSNE[complete_supervision_labels == 0]
        x_supervisioncls1_TSNE = latent_2D_TSNE[complete_supervision_labels == 1]
        x_supervisioncls2_TSNE = latent_2D_TSNE[complete_supervision_labels == 2]

        x_supervisioncls0_PCA = latent_2D_PCA[complete_supervision_labels == 0]
        x_supervisioncls1_PCA = latent_2D_PCA[complete_supervision_labels == 1]
        x_supervisioncls2_PCA = latent_2D_PCA[complete_supervision_labels == 2]

        fig, axs = plt.subplots(2, 2, figsize=(20, 10))
        fig.suptitle(title)

        axs[0, 0].scatter(x=x_cls1_TSNE[:, 0], y=x_cls1_TSNE[:, 1],c='r', alpha=0.2, label="Class 1")
        axs[0, 0].scatter(x=x_cls2_TSNE[:, 0], y=x_cls2_TSNE[:, 1],c='b', alpha=0.2, label="Class 2")
        axs[0, 0].scatter(x=x_cls3_TSNE[:, 0], y=x_cls3_TSNE[:, 1],c='g', alpha=0.2, label="Class 3")
        axs[0, 0].scatter(x=x_cls4_TSNE[:, 0], y=x_cls4_TSNE[:, 1],c='y', alpha=0.2, label="Class 4")
        if plot_centroids:
            axs[0, 0].scatter(x=x_cls5_TSNE[:, 0], y=x_cls5_TSNE[:, 1], c='k', alpha=0.99, marker="*", label="Centroid", s=100)
        axs[0, 0].title.set_text("TSNE (data classes)")
        axs[0, 0].legend()

        axs[0, 1].scatter(x=x_supervisioncls0_TSNE[:, 0], y=x_supervisioncls0_TSNE[:, 1], c='c', alpha=0.2, label="Normal training samples")
        axs[0, 1].scatter(x=x_supervisioncls1_TSNE[:, 0], y=x_supervisioncls1_TSNE[:, 1], c='m', alpha=0.2, label="SSL training samples")
        axs[0, 1].scatter(x=x_supervisioncls2_TSNE[:, 0], y=x_supervisioncls2_TSNE[:, 1], c='k', alpha=0.99, label="SAD training samples")
        if plot_centroids:
            axs[0, 1].scatter(x=x_cls5_TSNE[:, 0], y=x_cls5_TSNE[:, 1], c='k', alpha=0.99, marker="*", label="Centroid", s=100)
        axs[0, 1].title.set_text("TSNE (supervision classes)")
        axs[0, 1].legend()

        axs[1, 0].scatter(x=x_cls1_PCA[:, 0], y=x_cls1_PCA[:, 1], c='r', alpha=0.2, label="Class 1")
        axs[1, 0].scatter(x=x_cls2_PCA[:, 0], y=x_cls2_PCA[:, 1], c='b', alpha=0.2, label="Class 2")
        axs[1, 0].scatter(x=x_cls3_PCA[:, 0], y=x_cls3_PCA[:, 1], c='g', alpha=0.2, label="Class 3")
        axs[1, 0].scatter(x=x_cls4_PCA[:, 0], y=x_cls4_PCA[:, 1], c='y', alpha=0.2, label="Class 4")
        if plot_centroids:
            axs[1, 0].scatter(x=x_cls5_PCA[:, 0], y=x_cls5_PCA[:, 1], c='k', alpha=0.99, marker="*", label="Centroid", s=100)
        axs[1, 0].title.set_text("PCA (data classes)")
        axs[1, 0].legend()

        axs[1, 1].scatter(x=x_supervisioncls0_PCA[:, 0], y=x_supervisioncls0_PCA[:, 1], c='c', alpha=0.2, label="Normal training samples")
        axs[1, 1].scatter(x=x_supervisioncls1_PCA[:, 0], y=x_supervisioncls1_PCA[:, 1], c='m', alpha=0.2, label="SSL training samples")
        axs[1, 1].scatter(x=x_supervisioncls2_PCA[:, 0], y=x_supervisioncls2_PCA[:, 1], c='k', alpha=0.99, label="SAD training samples")
        if plot_centroids:
            axs[1, 1].scatter(x=x_cls5_PCA[:, 0], y=x_cls5_PCA[:, 1], c='k', alpha=0.99, marker="*", label="Centroid", s=100)
        axs[1, 1].title.set_text("PCA (supervision classes)")
        axs[1, 1].legend()

        plt.savefig(figure_path + figure_name)


def get_2D_latent_representations_testloader(dataloader, net, device, title, figure_path, figure_name, plot_centroids=False, centroids=[], supervision="AD"):
    """
    - Projects dataloader samples with neural network in 2D representations.
    - Returns matplotlib figure object.
    - testloader dedicated function since only the trainloader has 2D labels tensors in order to handle the double labeling (class labels in first column, training sample/SSL sample/SAD sample in second column).
    """

    # ignore FutureWarning of sklearn
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        net.eval()
        complete_outputs_init = False
        with torch.no_grad():
            for data, targets in dataloader:
                inputs, labels = data.to(device), targets.to(device)
                outputs = net(inputs)
                if not complete_outputs_init:
                    complete_outputs = outputs
                    complete_class_labels = labels
                    complete_outputs_init = True
                else:
                    complete_outputs = torch.cat((complete_outputs, outputs), dim=0)
                    complete_class_labels = torch.cat((complete_class_labels, labels), dim=0)

        if plot_centroids:
            if supervision=="AD":
                valid_centroids_idx = torch.LongTensor([0]).to(device)
                centroids = torch.index_select(centroids, 1, valid_centroids_idx)
            elif supervision=="SSL":
                valid_centroids_idx = torch.LongTensor([0, 1]).to(device)
                centroids = torch.index_select(centroids, 1, valid_centroids_idx)
            elif supervision=="SAD":
                valid_centroids_idx = torch.LongTensor([0, 2]).to(device)
                centroids = torch.index_select(centroids, 1, valid_centroids_idx)
            elif supervision=="SAD+SSL":
                valid_centroids_idx = torch.LongTensor([0, 1, 2]).to(device)
                centroids = torch.index_select(centroids, 1, valid_centroids_idx)
            else:
                raise ValueError("Supervision {} not implemented !".format(supervision))
            centroids = torch.transpose(centroids, 0, 1).to(device)
            complete_outputs = torch.cat((complete_outputs, centroids), dim=0)
            complete_class_labels = torch.cat((complete_class_labels, torch.ones((centroids.size()[0],)).to(device)*5), dim=0) # 5 is the label for centroids in this function

        complete_outputs = complete_outputs.cpu().detach().numpy()
        complete_class_labels = complete_class_labels.cpu().detach().numpy()

        # fit projections methods using complete data, all classes mixed together
        latent_2D_TSNE = TSNE(n_components=2).fit_transform(complete_outputs)
        latent_2D_PCA = PCA(n_components=2).fit_transform(complete_outputs)

        x_cls1_TSNE = latent_2D_TSNE[complete_class_labels == 1]
        x_cls2_TSNE = latent_2D_TSNE[complete_class_labels == 2]
        x_cls3_TSNE = latent_2D_TSNE[complete_class_labels == 3]
        x_cls4_TSNE = latent_2D_TSNE[complete_class_labels == 4]

        x_cls1_PCA = latent_2D_PCA[complete_class_labels == 1]
        x_cls2_PCA = latent_2D_PCA[complete_class_labels == 2]
        x_cls3_PCA = latent_2D_PCA[complete_class_labels == 3]
        x_cls4_PCA = latent_2D_PCA[complete_class_labels == 4]

        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle(title)

        axs[0].scatter(x=x_cls1_TSNE[:, 0], y=x_cls1_TSNE[:, 1], c='r', alpha=0.2, label="Class 1")
        axs[0].scatter(x=x_cls2_TSNE[:, 0], y=x_cls2_TSNE[:, 1], c='b', alpha=0.2, label="Class 2")
        axs[0].scatter(x=x_cls3_TSNE[:, 0], y=x_cls3_TSNE[:, 1], c='g', alpha=0.2, label="Class 3")
        axs[0].scatter(x=x_cls4_TSNE[:, 0], y=x_cls4_TSNE[:, 1], c='y', alpha=0.2, label="Class 4")
        axs[0].title.set_text("TSNE (data classes)")
        axs[0].legend()

        axs[1].scatter(x=x_cls1_PCA[:, 0], y=x_cls1_PCA[:, 1], c='r', alpha=0.2, label="Class 1")
        axs[1].scatter(x=x_cls2_PCA[:, 0], y=x_cls2_PCA[:, 1], c='b', alpha=0.2, label="Class 2")
        axs[1].scatter(x=x_cls3_PCA[:, 0], y=x_cls3_PCA[:, 1], c='g', alpha=0.2, label="Class 3")
        axs[1].scatter(x=x_cls4_PCA[:, 0], y=x_cls4_PCA[:, 1], c='y', alpha=0.2, label="Class 4")
        axs[1].title.set_text("PCA (data classes)")
        axs[1].legend()

        plt.savefig(figure_path + figure_name)
