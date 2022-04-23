import datetime
import random
import sys
from pathlib import Path
import ast
import seaborn as sns

from networks import *
from training import *

def deep_msvdd(AD_name, dataset, nbr_modes, nbr_epochs, batchsize, lr_init, lr_decay, lr_milestones,
               weight_decay, nbr_seeds, train_ratio, nbr_pulse_per_scan, nbr_targets, date_string,
               supervision="AD", loss="deep-msvdd", arch="net0", nbr_modes_SAD=1, SAD_ratio=0.01):

    print("\n {}".format(AD_name))

    classes = [1, 2, 3, 4]

    # Default device to 'cpu' if cuda is not available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    results_dicts_list = []
    results_path = './results/' + dataset + '/' + AD_name + '/' + loss + '/{}/'.format(date_string)

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

        normal_cls = np.random.choice(classes, nbr_modes, replace=False).tolist()
        outlier_cls = [cls for cls in classes if cls not in normal_cls]
        SAD_cls = np.random.choice(outlier_cls, nbr_modes_SAD, replace=False).tolist()

        radar_dataset = simulated_radar_dataset(train_ratio, normal_cls, SAD_cls, SAD_ratio, batchsize, nbr_targets, nbr_pulse_per_scan)
        radar_dataset.get_dataloaders(generator=g)

        # choose which supervision setup is to be used in terms of training samples availability (WARNING: check the selected loss in training.py to fully understand how supervised an experiment is)
        if supervision == "AD":
            # vanilla setup, corresponding to Deep SVDD original paper: only normal training samples available
            train_loader = radar_dataset.train_loader_sps_norm_2D
            x_train = radar_dataset.train_sps_norm_2D
        elif supervision=="SAD":
            train_loader = radar_dataset.train_loader_sps_norm_sad_2D
            x_train = radar_dataset.train_sps_norm_sad_2D
        else:
            raise ValueError("Supervision {} not implemented !".format(supervision))
        complete_train_loader = radar_dataset.complete_train_loader_sps_norm_2D
        val_loader = radar_dataset.val_loader_sps_norm_2D
        test_loader = radar_dataset.test_loader_sps_norm_2D

        learning_rate = lr_init
        if arch == "net0":
            net = SimuSPs_LeNet0(rep_dim=64).to(device)
        elif arch == "net1":
            net = SimuSPs_LeNet1(rep_dim=64).to(device)
        elif arch == "net2":
            net = SimuSPs_LeNet2(rep_dim=64).to(device)
        elif arch == "net3":
            net = SimuSPs_LeNet3(rep_dim=64).to(device)
        elif arch == "net4":
            net = SimuSPs_LeNet4(rep_dim=64).to(device)
        elif arch == "net5":
            net = SimuSPs_LeNet5(rep_dim=64).to(device)
        elif arch == "net6":
            net = SimuSPs_LeNet6(rep_dim=64).to(device)
        elif arch == "net7":
            net = SimuSPs_LeNet7(rep_dim=64).to(device)
        else:
            raise ValueError("Architecture {} not implemented !".format(arch))
        torch.save(net.state_dict(), results_path + 'untrained_net_seed{}_normal{}.pt'.format(seed, normal_cls))
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=False)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_decay)

        # Initialize hyperspheres centers; needs x_train and y_train to re-create dataloader with hyperspheres labels
        # start with 100 centroids as indicated in DeepMSVDD paper
        hyperspheres_centers, _ = init_centers_c_kmeans_MSVDD(device, train_loader, x_train, net, nbr_centroids=100, batch_size=batchsize, seed=seed)
        nu = 0.1

        title = AD_name + " / " + "supervision:{}".format(supervision) + " / " + loss + " / " + "SAD_cls: {}".format(SAD_cls) + " / " + "SAD_ratio: {}".format(SAD_ratio)

        # TODO: modify get_2D_latent to adapt it to deep MSVDD specific hyperspheres_centers centroids input
        # get_2D_latent_representations_trainloader(train_loader, net, device, title + " / train data before training", figure_path=results_path, figure_name="latent2Dtrain_untrained_seed{}_normal{}.png".format(seed, normal_cls), plot_centroids=True, centroids=hyperspheres_centers, supervision=supervision)
        # get_2D_latent_representations_testloader(test_loader, net, device, title + " / test data before training", figure_path=results_path, figure_name="latent2Dtest_untrained_seed{}_normal{}.png".format(seed, normal_cls), plot_centroids=True, centroids=hyperspheres_centers, supervision=supervision)

        epoch_losses, epoch_losses_radius_sqmean, epoch_losses_margin_loss, epoch_nbr_centroids, trainAUCs, valAUCs, testAUCs, test_scores, test_labels = training_deepMSVDD(train_loader,
                                                                                                                                                        complete_train_loader,
                                                                                                                                                        val_loader,
                                                                       test_loader, normal_cls, net, device, hyperspheres_centers,
                                                                       optimizer, scheduler, nbr_epochs, loss_name=loss, nu=nu)
        torch.save(net.state_dict(), results_path + 'trained_net_seed{}_normal{}.pt'.format(seed, normal_cls))

        ###################################################################

        best_AUC_epoch = int(np.argmax(valAUCs))
        train_AUC = trainAUCs[best_AUC_epoch]
        valid_AUC = valAUCs[best_AUC_epoch]
        test_AUC = testAUCs[best_AUC_epoch]

        ###################################################################

        fig, ax = plt.subplots(3, 2, figsize=(15, 15))
        ax[0, 0].plot(np.arange(len(epoch_losses)), epoch_losses, c='g', label="complete loss")
        ax[0, 0].plot(np.arange(len(epoch_losses)), epoch_losses_radius_sqmean, c='b', label="radius sqmean loss")
        ax[0, 0].plot(np.arange(len(epoch_losses)), epoch_losses_margin_loss, c='r', label="margin loss")
        ax[0, 0].set_xlabel("epoch")
        ax[0, 0].legend()
        ax[0, 0].title.set_text('Training loss')
        ax[0, 1].scatter(np.arange(nbr_epochs + 1), trainAUCs, c='g', label="train")
        ax[0, 1].scatter(np.arange(nbr_epochs + 1), valAUCs, c='b', label="valid")
        ax[0, 1].scatter(np.arange(nbr_epochs + 1), testAUCs, c='r', label="test")
        ax[0, 1].set_xlabel("epoch")
        ax[0, 1].legend()
        ax[0, 1].title.set_text('AUCs during training (max test AUC {})'.format(max(testAUCs)))
        sns.violinplot(x=test_labels[0], y=test_scores[0], ax=ax[1, 0]).set(title='AD scores before training')
        ax[1, 0].set_xlabel("class idx")
        sns.violinplot(x=test_labels[best_AUC_epoch], y=test_scores[best_AUC_epoch], ax=ax[1, 1]).set(title='AD scores after training')
        ax[1, 1].set_xlabel("class idx")
        ax[2, 0].plot(np.arange(nbr_epochs + 1), epoch_nbr_centroids)
        ax[2, 0].set_xlabel("epoch")
        ax[2, 0].title.set_text('# centroids per epoch (end {} - nu {})'.format(epoch_nbr_centroids[-1], nu))
        plt.savefig(results_path + 'seed{}_normal{}.png'.format(seed, normal_cls))

        # TODO: modify get_2D_latent to adapt it to deep MSVDD specific hyperspheres_centers centroids input
        # get_2D_latent_representations_trainloader(train_loader, net, device, title + " / train data after training", figure_path=results_path, figure_name="latent2Dtrain_trained_seed{}_normal{}.png".format(seed, normal_cls), plot_centroids=True, centroids=hyperspheres_centers, supervision=supervision)
        # get_2D_latent_representations_testloader(test_loader, net, device, title + " / test data after training", figure_path=results_path, figure_name="latent2Dtest_trained_seed{}_normal{}.png".format(seed, normal_cls), plot_centroids=True, centroids=hyperspheres_centers, supervision=supervision)

        ###################################################################

        results_dicts_list = store_comparison_results(results_dicts_list, train_AUC, valid_AUC, test_AUC, dataset,
                                                      AD_name, arch, supervision, loss, normal_cls, outlier_cls, SAD_cls,
                                                      SAD_ratio, nbr_modes, best_AUC_epoch, nbr_epochs, lr_init,
                                                      lr_milestones, batchsize, seed)

    results_df = pd.DataFrame(results_dicts_list)

    results_df = pd.concat([results_df], ignore_index=True)
    results_df_path = results_path + 'experiment_results.csv'
    results_df.to_csv(results_df_path)

    return results_df

if __name__ == '__main__':

    # no SAD or supervision input hyperparameters like in Deep SVDD and Deep RPO scripts since SAD and SSL not implemented for Deep MSVDD (yet ?)
    AD_name = str(sys.argv[1])
    dataset = str(sys.argv[2])
    nbr_modes = int(sys.argv[3])
    nbr_epochs = int(sys.argv[4])
    batchsize = int(sys.argv[5])
    lr_init = float(sys.argv[6])
    lr_decay = float(sys.argv[7])
    lr_milestones = ast.literal_eval(sys.argv[8])
    weight_decay = float(sys.argv[9])
    nbr_seeds = int(sys.argv[10])
    train_ratio = float(sys.argv[11])
    nbr_pulse_per_scan = int(sys.argv[12])
    nbr_targets = int(sys.argv[13])
    supervision = str(sys.argv[14])
    loss = str(sys.argv[15])
    arch = str(sys.argv[16])
    experiment_name = str(sys.argv[17])

    date_string = '{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.datetime.now())
    results_directory = './results/' + dataset + '/' + AD_name + '/' + loss + '/{}/'.format(date_string)
    Path(results_directory).mkdir(parents=True, exist_ok=True)
    Path('./comparison_graphs/').mkdir(parents=True, exist_ok=True)

    results_df = deep_msvdd(AD_name=AD_name, dataset=dataset, nbr_modes=nbr_modes, nbr_epochs=nbr_epochs, batchsize=batchsize,
                          lr_init=lr_init, lr_decay=lr_decay, lr_milestones=lr_milestones, weight_decay=weight_decay,
                          nbr_seeds=nbr_seeds, train_ratio=train_ratio, nbr_pulse_per_scan=nbr_pulse_per_scan,
                          nbr_targets=nbr_targets, date_string=date_string, supervision=supervision, loss=loss, arch=arch)

    # add results to last_results_df.csv for comparison graphs generation
    global_results_df_file_path = 'comparison_graphs/last_results_df_{}.csv'.format(experiment_name)
    global_results_df = pd.read_csv(global_results_df_file_path, index_col=False)
    frames = [global_results_df, results_df]
    global_results_df = pd.concat(frames, ignore_index=True)
    global_results_df.to_csv(global_results_df_file_path, index=False)