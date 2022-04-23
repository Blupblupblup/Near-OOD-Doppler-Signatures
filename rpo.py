import torch
from sklearn.base import OutlierMixin

# FOR SHALLOW AD (used in shallow_pca.py and shallow_tpca.py)

class Random_Projection_Outlyingness(OutlierMixin):
    # Implemented Santiago Velasco-Forero (slightly modified by Martin Bauw)
    # santiago.velasco@mines-paristech.fr
    # Main Reference:
    # - Liu, X. and Zuo, Y. (2014). Computing projection depth and its associated estimators. Statistics and Computing 24 51–63.
    # Other:
    # - Donoho, D.L. (1982). Breakdown properties of multivariate location estimators. Ph.D. qualifying paper. Department of Statistics, Harvard University.
    # - Liu, R.Y. (1992). Data depth and multivariate rank tests. In: Dodge, Y. (ed.), L1-Statistics and Related Methods, North-Holland (Amsterdam), 279–294.
    # - Velasco-Forero, S., & Angulo, J. (2012). Random projection depth for multivariate mathematical morphology. IEEE Journal of Selected Topics in Signal Processing, 6(7), 753-763.

    def __init__(self, nproj, unit_norm, device, estimator="mean"):
        self.nproj = nproj
        self.unit_norm = unit_norm
        self.device = device
        self.estimator = estimator
        super().__init__()

    def fit(self, X):
        """
        Fit Projection Depth Function.
        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency by convention.
        """
        X = torch.from_numpy(X).to(self.device)
        U = torch.randn(X.shape[1], self.nproj).to(self.device)
        U = torch.div(U, torch.norm(U, dim=0))
        Z = torch.mm(X.float(), U)
        m = torch.median(Z, dim=0)[0]
        Z = torch.abs(Z - m)
        s = torch.max(torch.median(Z, dim=0)[0], torch.ones(torch.median(Z, dim=0)[0].size()).to(self.device) * 0.0001)
        self.U = U
        self.m = m
        self.s = s
        return self

    def decision_function(self, X):
        X = torch.from_numpy(X).to(self.device)
        Z = torch.mm(X.float(), self.U)
        Z = torch.abs(Z - self.m)
        Z = torch.div(Z, self.s)
        if self.estimator == "max":
            return (torch.max(Z, dim=1)[0]).cpu()
        if self.estimator == "mean":
            return (torch.mean(Z, dim=1)).cpu()

# FOR DEEP AD (used in deep_rpo.py)

class Random_Projection_Outlyingness_deep_1D_ssl(OutlierMixin):
    # Directly inspired by the class Random_Projection_Outlyingness() as defined previously in this script

    def __init__(self, nproj, unit_norm, device, estimator="mean"):
        self.nproj = nproj
        self.unit_norm = unit_norm
        self.device = device
        self.estimator = estimator
        super().__init__()

    def fit(self, train_loader, net):
        """
        Fit Projection Depth Function.
        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency by convention.
        """

        net.eval()
        complete_outputs_init = False
        with torch.no_grad():
            for data, targets in train_loader:
                inputs, labels = data.to(self.device), targets.to(self.device)
                outputs = net(inputs)
                if not complete_outputs_init:
                    complete_outputs = outputs
                    complete_supervision_labels = labels[:,1] # labels[:,0] are class labels, labels[:,1] are supervision ie. vanilla/SSL/SAD samples labels
                    complete_outputs_init = True
                else:
                    complete_outputs = torch.cat((complete_outputs, outputs), dim=0)
                    complete_supervision_labels = torch.cat((complete_supervision_labels, labels[:,1]), dim=0)

        X0 = complete_outputs[complete_supervision_labels == 0]
        X1 = complete_outputs[complete_supervision_labels == 1]
        X2 = complete_outputs[complete_supervision_labels == 2]

        U = torch.randn(X0.shape[1], self.nproj).to(self.device) # X0.shape[1] == X1.shape[1], random projections are shared for two SSL classes
        if self.unit_norm:
            print("RPs are normalized !")
            self.U = torch.div(U, torch.norm(U, dim=0))
        else:
            print("RPs are NOT normalized !")
            self.U = U

        Z0 = torch.mm(X0.float(), self.U)
        self.m0 = torch.median(Z0, dim=0)[0]
        Z0 = torch.abs(Z0 - self.m0)
        self.s0 = torch.max(torch.median(Z0, dim=0)[0], torch.ones(torch.median(Z0, dim=0)[0].size()).to(self.device) * 0.0001) # regularize to avoid coefficients explosion

        if X1.size()[0] > 0:
            Z1 = torch.mm(X1.float(), self.U)
            self.m1 = torch.median(Z1, dim=0)[0]
            Z1 = torch.abs(Z1 - self.m1)
            self.s1 = torch.max(torch.median(Z1, dim=0)[0], torch.ones(torch.median(Z1, dim=0)[0].size()).to(self.device) * 0.0001) # regularize to avoid coefficients explosion

        if X2.size()[0] > 0:
            Z2 = torch.mm(X2.float(), self.U)
            self.m2 = torch.median(Z2, dim=0)[0]
            Z2 = torch.abs(Z2 - self.m2)
            self.s2 = torch.max(torch.median(Z2, dim=0)[0], torch.ones(torch.median(Z2, dim=0)[0].size()).to(self.device) * 0.0001)  # regularize to avoid coefficients explosion

        print(self.m0.size())

    def score_samples(self, X, supervision_cls=0):
        """
        we will only score test samples using not transformed SSL class centroid, the transformed "1" SSL class is only
        useful for training
        """
        if supervision_cls == 0: # normal training samples
            Z = torch.mm(X.float(), self.U)
            Z = torch.abs(Z - self.m0)
            Z = torch.div(Z, self.s0)
            if self.estimator == "max":
                return (torch.max(Z, dim=1)[0])
            if self.estimator == "mean":
                return (torch.mean(Z, dim=1))
        elif supervision_cls == 1: # SSL training samples, ie. normal rotated training samples
            Z = torch.mm(X.float(), self.U)
            Z = torch.abs(Z - self.m1)
            Z = torch.div(Z, self.s1)
            if self.estimator == "max":
                return (torch.max(Z, dim=1)[0])
            if self.estimator == "mean":
                return (torch.mean(Z, dim=1))
        elif supervision_cls == 2: # SSL training samples, ie. normal rotated training samples
            Z = torch.mm(X.float(), self.U)
            Z = torch.abs(Z - self.m2)
            Z = torch.div(Z, self.s2)
            if self.estimator == "max":
                return (torch.max(Z, dim=1)[0])
            if self.estimator == "mean":
                return (torch.mean(Z, dim=1))
        else:
            raise ValueError("Inexistent supervision_cls: must be either 0 (no SSL transformation) or 1 (transformed SSL class for training only) or 2 (untransformed labeled anomalies for SAD).")

    def score_samples_data_loader(self, data_loader, net):
        net.eval()
        complete_scores_init = False
        scores_labels = []
        with torch.no_grad():
            for data, targets in data_loader:
                inputs, labels = data.to(self.device), targets.to(self.device)
                outputs = net(inputs)
                scores_labels += labels.cpu().tolist()
                if not complete_scores_init:
                    complete_scores = self.score_samples(outputs.view(outputs.shape[0],-1), supervision_cls=0) # test samples are always scored using normal samples centroid and spread
                    complete_scores_init = True
                else:
                    scores = self.score_samples(outputs.view(outputs.shape[0],-1), supervision_cls=0) # test samples are always scored using normal samples centroid and spread
                    complete_scores = torch.cat((complete_scores, scores), dim=0)
        return complete_scores.cpu().detach().numpy(), scores_labels