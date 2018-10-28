from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=400, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--check-point', type=str, default=None, metavar='C',
                    help='resume from a checkpoint')
parser.add_argument('--use-aug-data', type=int, default=0, metavar='U',
                    help='set to 1 to use augumented dataset')
args = parser.parse_args()

torch.cuda.manual_seed(args.seed)

### Data Initialization and Loading
from data import initialize_data, val_transforms, train_transforms # data.py in the same folder

train_data_dir = '/train_images'
val_data_dir = '/val_images'

if args.use_aug_data == 1:
    train_data_dir = '/train_aug_images'
    val_data_dir = '/val_aug_images'


train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + train_data_dir,
                         transform=train_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + val_data_dir,
                         transform=val_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=1)

### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from model import Net, DenseNet, STN_DenseNet

############################# Models #########################################
# If you would like to test DenseNet with STN module, simply replace:
# model = DenseNet(...) with model = STN_DenseNet(...)
#############################################################################
# The best model I can get...achieved ~99.13% on the public leaderboard
## Please refer to aug_large.pth for the pre-trained model
# Even without **offline** preprocesing+augumentation, this model could achieve ~98.6% (my first submission :D)
# Please refer to noaug_large.pth
# This model have about ~10M parameters and consume about 15GB GPU memory during
# training, which could fit into a single TITAN V
model = DenseNet(growth_rate = 24, # K
                 block_config = (32,32,32), # (L - 4)/6
                 num_init_features = 48, # 2 * growth rate
                 bn_size = 4,
                 drop_rate = 0.1,
                 num_classes = 43)

# A smaller model, which could also achieve ~98.3% on the public leaderboard 
# with offline preprocessing+augumentation
# Please refer to aug_small.pth
# This model have about 0.9M parameters
# model = DenseNet(growth_rate = 12, # K
#                  block_config = (16,16,16), # (L - 4)/6
#                  num_init_features = 32,
#                  bn_size = 4,
#                  drop_rate = 0.1,
#                  num_classes = 43)

# A medium modelm, achieved about ~98.8% with offline preprocessing+augumentation
# I occasionally overwrite the pre-trained model for this one...
# and have no time to train it again........
# model = DenseNet(growth_rate = 16,
#                  block_config = (24, 24 ,24),
#                  num_init_features = 32,
#                  bn_size = 4,
#                  drop_rate = 0.05
#                  num_classes = 43)

# load model from a check point
if args.check_point:
        state_dict = torch.load(args.check_point)
        model.load_state_dict(state_dict)

model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6, 14, 22], gamma=0.1)

def train(epoch):
    model.train()
    train_correct = 0
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        pred = output.data.max(1, keepdim=True)[1]
        train_loss += loss.data[0]
        train_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    
    train_loss /= len(train_loader.dataset)
    print('\nTraining set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        train_loss, train_correct, len(train_loader.dataset),
        100.0 * float(train_correct) / len(train_loader.dataset)))

    scheduler.step()


def validation():
    model.eval()
    validation_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = Variable(data.cuda()), Variable(target.cuda())
            output = model(data)
            validation_loss += F.cross_entropy(output, target, size_average=False).data[0] # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100.0 * float(correct) / len(val_loader.dataset)))

for epoch in range(1, args.epochs + 1):
    train(epoch)
    validation()
    model_file = 'model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    print('\nSaved model to ' + model_file + '. You can run `python evaluate.py ' + model_file + '` to generate the Kaggle formatted csv file')
