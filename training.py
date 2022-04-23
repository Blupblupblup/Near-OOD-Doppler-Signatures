from sklearn.metrics import roc_auc_score
from utils import *
from tqdm import tqdm

def get_epoch_AUC_deepSVDD(data_loader, net, c, device, normal_cls):
    scores = []
    scores_labels = []

    net.eval()
    with torch.no_grad():
        for data, targets in data_loader:
            inputs, labels = data.to(device), targets.to(device)
            outputs = net(inputs)
            batch_scores = torch.sum((outputs - c) ** 2, dim=tuple(range(1, outputs.dim())))
            scores += batch_scores.cpu().tolist()
            scores_labels += labels.cpu().tolist()

    y_score = scores
    y_ad = convert_labels(scores_labels, normal_cls)
    auc = roc_auc_score(y_ad, y_score)
    return auc, scores, scores_labels

def get_epoch_AUC_deepSVDD_ssl(data_loader, net, c, device, normal_cls):
    scores = []
    scores_labels = []

    net.eval()
    with torch.no_grad():
        for data, targets in data_loader:
            inputs, labels = data.to(device), targets.to(device)
            outputs = net(inputs)
            batch_scores = torch.sum((outputs - c[:,0]) ** 2, dim=tuple(range(1, outputs.dim())))
            scores += batch_scores.cpu().tolist()
            scores_labels += labels.cpu().tolist()

    y_score = scores
    y_ad = convert_labels(scores_labels, normal_cls)
    auc = roc_auc_score(y_ad, y_score)
    return auc, scores, scores_labels

########################################################################################################################

def score_samples_data_loader_DeepMSVDD(test_loader, device, net, hyperspheres_centers, radius):
    scores = []
    scores_labels = []
    net.eval()
    with torch.no_grad():
        for data, targets in test_loader:
            inputs, labels = data.to(device), targets.to(device)
            outputs = net(inputs)
            dist_to_centers = torch.sum((outputs.unsqueeze(1).repeat(1, hyperspheres_centers.size()[0],1) - hyperspheres_centers.unsqueeze(0).repeat(outputs.size()[0], 1, 1)) ** 2, dim=2)
            try:
                scores_per_center = torch.cat([scores_per_center, dist_to_centers], dim=0)
            except UnboundLocalError: # if scores_per_center does not exist yet, create it
                scores_per_center = dist_to_centers

            batch_scores, min_dist_idx = torch.min(dist_to_centers, dim=1)
            batch_scores -= radius[min_dist_idx] ** 2

            scores += batch_scores.cpu().tolist()
            scores_labels += labels.cpu().tolist()

    return scores, scores_labels, scores_per_center

def get_epoch_AUC_deepMSVDD(data_loader, device, net, hyperspheres_centers, radius, normal_cls):
    scores, scores_labels, scores_per_center = score_samples_data_loader_DeepMSVDD(data_loader, device, net, hyperspheres_centers, radius)
    y_ad = convert_labels(scores_labels, normal_cls)
    auc = roc_auc_score(y_ad, scores)
    return auc, scores, scores_labels, scores_per_center

########################################################################################################################

def get_epoch_AUC_deepRPO(data_loader, net, RPO, normal_cls):
    scores, scores_labels = RPO.score_samples_data_loader(data_loader, net)
    y_train_ad = convert_labels(scores_labels, normal_cls)
    auc = roc_auc_score(y_train_ad, scores)
    return auc, scores, scores_labels

def get_epoch_AUC_deepRPO_ssl(data_loader, net, RPO_ssl, normal_cls):
    scores, scores_labels = RPO_ssl.score_samples_data_loader(data_loader, net)
    y_train_ad = convert_labels(scores_labels, normal_cls)
    auc = roc_auc_score(y_train_ad, scores)
    return auc, scores, scores_labels

def training_deepMSVDD(train_loader, complete_train_loader, val_loader, test_loader, normal_cls, net, device, hyperspheres_centers, optimizer, scheduler, num_epochs, loss_name, nu=0.1):

    epoch_losses = []
    epoch_losses_radius_sqmean = []
    epoch_losses_margin_loss = []
    epoch_nbr_centroids = []
    trainAUCs = []
    valAUCs = []
    testAUCs = []
    test_scores = []
    test_labels = []

    radius = update_radius_DMSVDD(hyperspheres_centers, nu, train_loader, net, device)
    # get all AUCs and test scores before learning begins
    trainAUCs.append(get_epoch_AUC_deepMSVDD(complete_train_loader, device, net, hyperspheres_centers, radius, normal_cls)[0])
    valAUCs.append(get_epoch_AUC_deepMSVDD(val_loader, device, net, hyperspheres_centers, radius, normal_cls)[0])
    # keep all scores for every epoch for test set in order to plot best epoch scores distribution
    epoch_test_auc, epoch_test_scores, epoch_test_labels, scores_per_center = get_epoch_AUC_deepMSVDD(test_loader, device, net, hyperspheres_centers, radius, normal_cls)
    testAUCs.append(epoch_test_auc)
    test_scores.append(epoch_test_scores)
    test_labels.append(epoch_test_labels)
    epoch_nbr_centroids.append(hyperspheres_centers.size()[0])

    for epoch in tqdm(range(num_epochs)):

        running_loss = 0.0
        running_loss_radius_sqmean = 0.0
        running_loss_margin_loss = 0.0
        net.train()
        for i, (data, targets) in enumerate(train_loader, 0):
            inputs, labels = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            dist_to_centers = torch.sum((outputs.unsqueeze(1).repeat(1, hyperspheres_centers.size()[0],1) - hyperspheres_centers.unsqueeze(0).repeat(outputs.size()[0], 1, 1)) ** 2, dim=2)
            dist_to_best_center, best_center_idx = torch.min(dist_to_centers, dim=1)
            dist_to_worst_center, best_center_idx = torch.max(dist_to_centers, dim=1)
            best_center_sqradius = radius[best_center_idx] ** 2

            radius_sqmean = (1 / radius.size()[0]) * torch.sum(radius ** 2)
            margin_loss = (1 / (nu * inputs.size()[0])) * torch.sum(torch.maximum(dist_to_best_center - best_center_sqradius, torch.zeros((dist_to_best_center.size()[0],)).to(device)))

            if loss_name=="deep-msvdd":
                loss = radius_sqmean + margin_loss
            elif loss_name=="deep-msvdd-meanbest":
                loss = torch.mean(dist_to_best_center)
            elif loss_name=="deepsvdd-meanworst":
                loss = torch.mean(dist_to_worst_center) # could be interpreted as an effort to re-attach points to abandoned centroids ?
            elif loss_name == "deepmsvdd-sad": # idea was to simultaneously increase relative pressure between good and bad hyperspheres but doesn't seem to work
                loss = torch.mean(dist_to_best_center) + 1/torch.mean(dist_to_worst_center) # focus on good centroids and actively exclude the rest
            elif loss_name == "dmsvdd-pollution": # no distinction between training samples in the computation of dist_to_centers, so nothing to add to the loss for SAD samples to be taken into account
                loss = radius_sqmean + margin_loss
            elif loss_name == "dmsvdd-meanbest-pollution":
                loss = torch.mean(dist_to_best_center)
            else:
                raise ValueError("Loss {} is not implemented".format(loss_name))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_loss_radius_sqmean += radius_sqmean.item()
            running_loss_margin_loss += margin_loss.item()


        trainAUCs.append(get_epoch_AUC_deepMSVDD(complete_train_loader, device, net, hyperspheres_centers, radius, normal_cls)[0])
        valAUCs.append(get_epoch_AUC_deepMSVDD(val_loader, device, net, hyperspheres_centers, radius, normal_cls)[0])
        # keep all scores for every epoch for test set in order to plot best epoch scores distribution
        epoch_test_auc, epoch_test_scores, epoch_test_labels, scores_per_center = get_epoch_AUC_deepMSVDD(test_loader, device, net, hyperspheres_centers, radius, normal_cls)
        testAUCs.append(epoch_test_auc)
        test_scores.append(epoch_test_scores)
        test_labels.append(epoch_test_labels)
        epoch_nbr_centroids.append(hyperspheres_centers.size()[0])
        epoch_losses.append(running_loss)
        epoch_losses_radius_sqmean.append(running_loss_radius_sqmean)
        epoch_losses_margin_loss.append(running_loss_margin_loss)

        scheduler.step()

        hyperspheres_centers = filter_centers_DMSVDD(hyperspheres_centers, radius)
        radius = update_radius_DMSVDD(hyperspheres_centers, nu, train_loader, net, device)

    return epoch_losses, epoch_losses_radius_sqmean, epoch_losses_margin_loss, epoch_nbr_centroids, trainAUCs, valAUCs, testAUCs, test_scores, test_labels

def training_deepSVDD(train_loader, complete_train_loader, val_loader, test_loader, normal_cls, net, device, c, optimizer, scheduler, num_epochs, loss_name):
    epoch_losses = []
    trainAUCs = []
    valAUCs = []
    testAUCs = []
    test_scores = []
    test_labels = []

    # get all AUCs and test scores before learning begins
    trainAUCs.append(get_epoch_AUC_deepSVDD_ssl(complete_train_loader, net, c, device, normal_cls)[0])
    valAUCs.append(get_epoch_AUC_deepSVDD_ssl(val_loader, net, c, device, normal_cls)[0])
    # keep all scores for every epoch for test set in order to plot best epoch scores distribution
    epoch_test_auc, epoch_test_scores, epoch_test_labels = get_epoch_AUC_deepSVDD_ssl(test_loader, net, c, device, normal_cls)
    testAUCs.append(epoch_test_auc)
    test_scores.append(epoch_test_scores)
    test_labels.append(epoch_test_labels)

    for epoch in tqdm(range(num_epochs)):
        net.train()
        running_loss = 0.0
        for i, (data,targets) in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data.to(device), targets.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = net(inputs)
            zeros = torch.zeros((outputs.size()[0])).to(device)
            dist = torch.zeros((outputs.size()[0])).to(device)
            dist += torch.where(labels[:, 1] == 0, torch.sum((outputs - c[:, 0]) ** 2, dim=tuple(range(1, outputs.dim()))), zeros)
            if loss_name=="deep-svdd":
                pass # nothing to do, everything is already in dist
            elif loss_name=="ssldata-sslcentroid":
                dist += torch.where(labels[:, 1] == 1, torch.sum((outputs - c[:, 1]) ** 2, dim=tuple(range(1, outputs.dim()))), zeros)
            elif loss_name=="ssldata-away":
                dist += torch.where(labels[:, 1] == 1, 1/(torch.sum((outputs - c[:, 0]) ** 2, dim=tuple(range(1, outputs.dim())))), zeros) # away from normal samples centroid, and not from SSL samples centroid
            elif loss_name=="saddata-sadcentroid":
                dist += torch.where(labels[:, 1] == 2, torch.sum((outputs - c[:, 2]) ** 2, dim=tuple(range(1, outputs.dim()))), zeros)
            elif loss_name=="saddata-away":
                dist += torch.where(labels[:, 1] == 2, 1/(torch.sum((outputs - c[:, 0]) ** 2, dim=tuple(range(1, outputs.dim())))), zeros)
            elif loss_name=="ssldata-sslcentroid_saddata-away":
                dist += torch.where(labels[:, 1] == 1, torch.sum((outputs - c[:, 1]) ** 2, dim=tuple(range(1, outputs.dim()))), zeros)
                dist += torch.where(labels[:, 1] == 2, 1 / (torch.sum((outputs - c[:, 0]) ** 2, dim=tuple(range(1, outputs.dim())))), zeros)
            elif loss_name=="ssldata-away_saddata-sadcentroid":
                dist += torch.where(labels[:, 1] == 1, 1 / (torch.sum((outputs - c[:, 0]) ** 2, dim=tuple(range(1, outputs.dim())))), zeros)
                dist += torch.where(labels[:, 1] == 2, torch.sum((outputs - c[:, 2]) ** 2, dim=tuple(range(1, outputs.dim()))), zeros)
            elif loss_name=="ssldata-sslcentroid_saddata-sadcentroid":
                dist += torch.where(labels[:, 1] == 1, torch.sum((outputs - c[:, 1]) ** 2, dim=tuple(range(1, outputs.dim()))), zeros)
                dist += torch.where(labels[:, 1] == 2, torch.sum((outputs - c[:, 2]) ** 2, dim=tuple(range(1, outputs.dim()))), zeros)
            elif loss_name=="ssldata-away_saddata-away":
                dist += torch.where(labels[:, 1] == 1, 1 / (torch.sum((outputs - c[:, 0]) ** 2, dim=tuple(range(1, outputs.dim())))), zeros)
                dist += torch.where(labels[:, 1] == 2, 1 / (torch.sum((outputs - c[:, 0]) ** 2, dim=tuple(range(1, outputs.dim())))), zeros)
            elif loss_name=="saddata-pollution":
                dist += torch.where(labels[:, 1] == 2, torch.sum((outputs - c[:, 0]) ** 2, dim=tuple(range(1, outputs.dim()))), zeros) # take into account SAD samples just like normal training samples, thus simulating labeling mistakes
            elif loss_name=="ssldata-sslcentroid_saddata-pollution":
                dist += torch.where(labels[:, 1] == 1, torch.sum((outputs - c[:, 1]) ** 2, dim=tuple(range(1, outputs.dim()))), zeros)
                dist += torch.where(labels[:, 1] == 2, torch.sum((outputs - c[:, 0]) ** 2, dim=tuple(range(1, outputs.dim()))), zeros)
            elif loss_name=="ssldata-away_saddata-pollution":
                dist += torch.where(labels[:, 1] == 1, 1 / (torch.sum((outputs - c[:, 0]) ** 2, dim=tuple(range(1, outputs.dim())))), zeros)
                dist += torch.where(labels[:, 1] == 2, torch.sum((outputs - c[:, 0]) ** 2, dim=tuple(range(1, outputs.dim()))), zeros)
            # elif loss_name=="deepsvdd-ssl-pairwise-unif":
            #     dist += torch.where(labels[:, 1] == 1, torch.sum((outputs - c[:, 1]) ** 2, dim=tuple(range(1, outputs.dim()))), zeros)
            #     dist += 0.001 / torch.cdist(outputs[labels[:, 1] == 0], outputs[labels[:, 1] == 0])
            # elif loss_name=="deepsvdd-sad-pairwise-unif":
            #     dist += torch.where(labels[:, 1] == 1, 1/(torch.sum((outputs - c[:, 0]) ** 2, dim=tuple(range(1, outputs.dim())))), zeros)
            #     dist += 0.001 / torch.cdist(outputs[labels[:, 1] == 0], outputs[labels[:, 1] == 0])
            else:
                raise ValueError("Loss {} not implemented !".format(loss_name))
            loss = torch.mean(dist)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        trainAUCs.append(get_epoch_AUC_deepSVDD_ssl(complete_train_loader, net, c, device, normal_cls)[0])
        valAUCs.append(get_epoch_AUC_deepSVDD_ssl(val_loader, net, c, device, normal_cls)[0])
        # keep all scores for every epoch for test set in order to plot best epoch scores distribution
        epoch_test_auc, epoch_test_scores, epoch_test_labels = get_epoch_AUC_deepSVDD_ssl(test_loader, net, c, device, normal_cls)
        testAUCs.append(epoch_test_auc)
        test_scores.append(epoch_test_scores)
        test_labels.append(epoch_test_labels)
        epoch_losses.append(running_loss)

        scheduler.step()

    return epoch_losses, trainAUCs, valAUCs, testAUCs, test_scores, test_labels, net

def training_deepRPO(train_loader, complete_train_loader, val_loader, test_loader, normal_cls, net, device, RPO_ssl, optimizer, scheduler, num_epochs, loss_name):

    RPO_ssl.fit(train_loader, net)

    epoch_losses = []
    trainAUCs = []
    valAUCs = []
    testAUCs = []
    test_scores = []
    test_labels = []

    # get all AUCs and test scores before learning begins
    trainAUCs.append(get_epoch_AUC_deepRPO_ssl(complete_train_loader, net, RPO_ssl, normal_cls)[0])
    valAUCs.append(get_epoch_AUC_deepRPO_ssl(val_loader, net, RPO_ssl, normal_cls)[0])
    # keep all scores for every epoch for test set in order to plot best epoch scores distribution
    epoch_test_auc, epoch_test_scores, epoch_test_labels = get_epoch_AUC_deepRPO_ssl(test_loader, net, RPO_ssl, normal_cls)
    testAUCs.append(epoch_test_auc)
    test_scores.append(epoch_test_scores)
    test_labels.append(epoch_test_labels)

    for epoch in tqdm(range(num_epochs)):
        net.train()
        running_loss = 0.0
        for i, (data,targets) in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data.to(device), targets.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = net(inputs)
            zeros = torch.zeros((outputs.size()[0])).to(device)
            dist = torch.zeros((outputs.size()[0])).to(device)
            dist += torch.where(labels[:, 1] == 0, RPO_ssl.score_samples(outputs, supervision_cls=0), zeros)
            if loss_name=="deep-rpo":
                pass  # nothing to do, everything is already in dist
            elif loss_name=="ssldata-sslcentroid":
                dist += torch.where(labels[:, 1] == 1, RPO_ssl.score_samples(outputs, supervision_cls=1), zeros)
            elif loss_name=="ssldata-away":
                dist += torch.where(labels[:, 1] == 1, 1 / (RPO_ssl.score_samples(outputs, supervision_cls=0)), zeros) # away from normal samples centroid, and not from SSL samples centroid
            elif loss_name=="saddata-sadcentroid":
                dist += torch.where(labels[:, 1] == 2, RPO_ssl.score_samples(outputs, supervision_cls=2), zeros)
            elif loss_name=="saddata-away":
                dist += torch.where(labels[:, 1] == 2, 1 / (RPO_ssl.score_samples(outputs, supervision_cls=0)), zeros)
            elif loss_name=="ssldata-sslcentroid_saddata-away":
                dist += torch.where(labels[:, 1] == 1, RPO_ssl.score_samples(outputs, supervision_cls=1), zeros)
                dist += torch.where(labels[:, 1] == 2, 1 / (RPO_ssl.score_samples(outputs, supervision_cls=0)), zeros)
            elif loss_name=="ssldata-away_saddata-sadcentroid":
                dist += torch.where(labels[:, 1] == 1, 1 / (RPO_ssl.score_samples(outputs, supervision_cls=0)), zeros)
                dist += torch.where(labels[:, 1] == 2, RPO_ssl.score_samples(outputs, supervision_cls=2), zeros)
            elif loss_name=="ssldata-sslcentroid_saddata-sadcentroid":
                dist += torch.where(labels[:, 1] == 1, RPO_ssl.score_samples(outputs, supervision_cls=1), zeros)
                dist += torch.where(labels[:, 1] == 2, RPO_ssl.score_samples(outputs, supervision_cls=2), zeros)
            elif loss_name=="ssldata-away_saddata-away":
                dist += torch.where(labels[:, 1] == 1, 1 / (RPO_ssl.score_samples(outputs, supervision_cls=0)), zeros)
                dist += torch.where(labels[:, 1] == 2, 1 / (RPO_ssl.score_samples(outputs, supervision_cls=0)), zeros)
            elif loss_name=="saddata-pollution":
                dist += torch.where(labels[:, 1] == 2, RPO_ssl.score_samples(outputs, supervision_cls=0), zeros) # take into account SAD samples just like normal training samples, thus simulating labeling mistakes
            elif loss_name=="ssldata-sslcentroid_saddata-pollution":
                dist += torch.where(labels[:, 1] == 1, RPO_ssl.score_samples(outputs, supervision_cls=1), zeros)
                dist += torch.where(labels[:, 1] == 2, RPO_ssl.score_samples(outputs, supervision_cls=0), zeros)
            elif loss_name=="ssldata-away_saddata-pollution":
                dist += torch.where(labels[:, 1] == 1, 1 / (RPO_ssl.score_samples(outputs, supervision_cls=0)), zeros)
                dist += torch.where(labels[:, 1] == 2, RPO_ssl.score_samples(outputs, supervision_cls=0), zeros)
            else:
                raise ValueError("Loss {} not implemented !".format(loss_name))
            loss = torch.mean(dist)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        trainAUCs.append(get_epoch_AUC_deepRPO_ssl(complete_train_loader, net, RPO_ssl, normal_cls)[0])
        valAUCs.append(get_epoch_AUC_deepRPO_ssl(val_loader, net, RPO_ssl, normal_cls)[0])
        # keep all scores for every epoch for test set in order to plot best epoch scores distribution
        epoch_test_auc, epoch_test_scores, epoch_test_labels = get_epoch_AUC_deepRPO_ssl(test_loader, net, RPO_ssl, normal_cls)
        testAUCs.append(epoch_test_auc)
        test_scores.append(epoch_test_scores)
        test_labels.append(epoch_test_labels)
        epoch_losses.append(running_loss)

        scheduler.step()

    return epoch_losses, trainAUCs, valAUCs, testAUCs, test_scores, test_labels, net

def get_testAUC_at_max_valAUC(valAUCs, testAUCs):
    epoch_max_valAUC = np.array(valAUCs).argmax()
    testAUC_at_max_valAUC = testAUCs[epoch_max_valAUC]
    return testAUC_at_max_valAUC, epoch_max_valAUC