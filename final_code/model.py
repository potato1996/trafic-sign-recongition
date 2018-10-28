import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

# The following DenseNet implementation is modified based on torchvision.models.DenseNet
# Please check: https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
# ------------------------------------------------------------------------------
class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        
        super(_DenseLayer, self).__init__()

        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        
        super(_DenseBlock, self).__init__()
        
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):

        super(_Transition, self).__init__()

        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):

    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=16, block_config=(16, 16, 16),
                 num_init_features=32, bn_size=4, drop_rate=0.2, num_classes=nclasses):

        super(DenseNet, self).__init__()

        # First convolution. Output Size: (num_init_features, 32, 32)
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
	    # Here we don't need the pool2d
            #('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))


        # Each denseblock
        # Sizes after each denseblock are:
        # (16x16, 8x8, 8x8)
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)

            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)



    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        # Final pooling layer: (8x8) => (1x1)
        out = F.avg_pool2d(out, kernel_size=8, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out

#------------------------------------------------------------------------------------

# Implement an STN block
# Assuming that the input size is (32x32)
class _STN(nn.Module):
    def __init__(self, num_features, drop_rate):
        super(_STN, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(num_features, 10, kernel_size=5)), # outsize(10, 28, 28)
            ('pool1', nn.MaxPool2d(2, stride=2)), # outsize(10, 14, 14)
            ('relu1', nn.ReLU(True)),
            ('norm1', nn.BatchNorm2d(10)),
            ('conv2', nn.Conv2d(10, 20, kernel_size=5)), # outsize(20, 10, 10)
            ('pool2', nn.MaxPool2d(2, stride=2)), # outsize(20, 5, 5)
            ('relu2', nn.ReLU(True)),
            ('norm2', nn.BatchNorm2d(20))
        ]))
    
        self.fc_loc = nn.Sequential(
            nn.Linear(20 * 5 * 5, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2),
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs    = self.features(x)
        xs    = xs.view(-1, 20 * 5 * 5)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid  = F.affine_grid(theta, x.size())
        x     = F.grid_sample(x, grid)
        return x


class STN_DenseNet(DenseNet):
        def __init__(self, growth_rate=16, block_config=(16, 16, 16),
                     num_init_features=32, bn_size=4, drop_rate=0.2, num_classes=nclasses):

            super(STN_DenseNet, self).__init__(growth_rate, block_config, num_init_features, 
                                               bn_size, drop_rate, num_classes)
            self.stn = _STN(3, 0.2)

        def forward(self, x):
           x = self.stn(x)
           x = super(STN_DenseNet, self).forward(x)
           return x
