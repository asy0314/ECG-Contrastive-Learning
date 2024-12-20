import torch.nn as nn

from models import xresnet1d


class ModelBase(nn.Module):
    def __init__(self, arch='xresnet1d18', use_mlp=True):
        super(ModelBase, self).__init__()

        model_arch = getattr(xresnet1d, arch)
        net = model_arch() # num_classes does not matter here 

        list_of_modules = list(net.children())
        self.features = nn.Sequential(*list_of_modules[:-1], list_of_modules[-1][0]) # upto AdaptiveConcatPool1d
        self.out_features = net[-1][-1].in_features
        net[0][0] = nn.Conv1d(12, 32, kernel_size=5, stride=2, padding=2)
    
        # projection MLP
        self.projection_dim = self.out_features // 4
        if use_mlp:
            self.projection_head = nn.Sequential(
                nn.Linear(self.out_features, self.out_features),
                nn.ReLU(),
                nn.Linear(self.out_features, self.projection_dim),
            )
        else:
            self.projection_head = nn.Linear(self.out_features, self.projection_dim)


    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()
        z = self.projection_head(h)
        # note: not normalized here
        return z
