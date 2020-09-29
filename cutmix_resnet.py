import resnet as RN
import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Backbone():

    def __init__(self, trained_dataset='imagenet', depth=101, num_of_class=1000, path=None):
        model = RN.ResNet(trained_dataset, depth, num_of_class)

        model = torch.nn.DataParallel(model).cuda()

        if path is not None:
            print("=> loading checkpoint '{}'".format(path))
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'".format(path))
            model.module.avgpool = Identity()
            model.module.fc = Identity()
        self.model = model
