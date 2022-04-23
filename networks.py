import torch.nn as nn
import torch.nn.functional as F

class SimuSPs_LeNet0(nn.Module):
    """
    Architecture inspired by https://github.com/lukasruff/Deep-SAD-PyTorch/blob/master/src/networks/fmnist_LeNet.py
    """
    def __init__(self, rep_dim=64):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 16, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(16, 32, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(32 * 16 * 16, 128, bias=False)
        self.bn1d1 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(128, self.rep_dim, bias=False)

    def forward(self, x):
        x = x.view(-1, 1, 64, 64)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = x.view(int(x.size(0)), -1)
        x = F.leaky_relu(self.bn1d1(self.fc1(x)))
        x = self.fc2(x)
        return x

class SimuSPs_LeNet1(nn.Module):

    def __init__(self, rep_dim=64):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv0 = nn.Conv2d(1, 16, 7, bias=False, padding=2)
        self.bn2d0 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.conv1 = nn.Conv2d(16, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 32, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(32 * 7 * 7, 128, bias=False)
        self.bn1d1 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(128, self.rep_dim, bias=False)

    def forward(self, x):
        x = x.view(-1, 1, 64, 64)
        x = self.conv0(x)
        x = self.pool(F.leaky_relu(self.bn2d0(x)))
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = x.view(int(x.size(0)), -1)
        x = F.leaky_relu(self.bn1d1(self.fc1(x)))
        x = self.fc2(x)
        return x

class SimuSPs_LeNet2(nn.Module):

    def __init__(self, rep_dim=64):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv0 = nn.Conv2d(1, 16, 7, bias=False, padding=2)
        self.bn2d0 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.conv1 = nn.Conv2d(16, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 16, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(16 * 7 * 7, 128, bias=False)
        self.bn1d1 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(128, 64, bias=False)
        self.bn1d2 = nn.BatchNorm1d(64, eps=1e-04, affine=False)
        self.fc3 = nn.Linear(64, self.rep_dim, bias=False)

    def forward(self, x):
        x = x.view(-1, 1, 64, 64)
        x = self.conv0(x)
        x = self.pool(F.leaky_relu(self.bn2d0(x)))
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = x.view(int(x.size(0)), -1)
        x = F.leaky_relu(self.bn1d1(self.fc1(x)))
        x = F.leaky_relu(self.bn1d2(self.fc2(x)))
        x = self.fc3(x)
        return x

class SimuSPs_LeNet3(nn.Module):

    def __init__(self, rep_dim=64):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv0 = nn.Conv2d(1, 16, 10, bias=False, padding=2)
        self.bn2d0 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.conv1 = nn.Conv2d(16, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 16, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(16 * 7 * 7, 128, bias=False)
        self.bn1d1 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(128, 64, bias=False)
        self.bn1d2 = nn.BatchNorm1d(64, eps=1e-04, affine=False)
        self.fc3 = nn.Linear(64, self.rep_dim, bias=False)

    def forward(self, x):
        x = x.view(-1, 1, 64, 64)
        x = self.conv0(x)
        x = self.pool(F.leaky_relu(self.bn2d0(x)))
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = x.view(int(x.size(0)), -1)
        x = F.leaky_relu(self.bn1d1(self.fc1(x)))
        x = F.leaky_relu(self.bn1d2(self.fc2(x)))
        x = self.fc3(x)
        return x

class SimuSPs_LeNet4(nn.Module):

    def __init__(self, rep_dim=64):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 16, (9,5), bias=False, padding=(4,2))
        self.bn2d1 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(16, 32, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(32 * 16 * 16, 128, bias=False)
        self.bn1d1 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(128, self.rep_dim, bias=False)

    def forward(self, x):
        x = x.view(-1, 1, 64, 64)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = x.view(int(x.size(0)), -1)
        x = F.leaky_relu(self.bn1d1(self.fc1(x)))
        x = self.fc2(x)
        return x

class SimuSPs_LeNet5(nn.Module):

    def __init__(self, rep_dim=64):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 16, (5,9), bias=False, padding=(2,4))
        self.bn2d1 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(16, 32, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(32 * 16 * 16, 128, bias=False)
        self.bn1d1 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(128, self.rep_dim, bias=False)

    def forward(self, x):
        x = x.view(-1, 1, 64, 64)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = x.view(int(x.size(0)), -1)
        x = F.leaky_relu(self.bn1d1(self.fc1(x)))
        x = self.fc2(x)
        return x

class SimuSPs_LeNet6(nn.Module):

    def __init__(self, rep_dim=64):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 16, 5, bias=False, padding=4, dilation=2)
        self.bn2d1 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(16, 32, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(32 * 16 * 16, 128, bias=False)
        self.bn1d1 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(128, self.rep_dim, bias=False)

    def forward(self, x):
        x = x.view(-1, 1, 64, 64)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = x.view(int(x.size(0)), -1)
        x = F.leaky_relu(self.bn1d1(self.fc1(x)))
        x = self.fc2(x)
        return x

class SimuSPs_LeNet7(nn.Module):

    def __init__(self, rep_dim=64):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 16, (9, 3), bias=False, padding=(4, 2))
        self.bn2d1 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(16, 32, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(32 * 16 * 16, 128, bias=False)
        self.bn1d1 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(128, self.rep_dim, bias=False)

    def forward(self, x):
        x = x.view(-1, 1, 64, 64)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = x.view(int(x.size(0)), -1)
        x = F.leaky_relu(self.bn1d1(self.fc1(x)))
        x = self.fc2(x)
        return x