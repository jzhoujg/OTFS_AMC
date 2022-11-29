import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import numpy as np
import scipy.io as scio
import torch

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = out + identity

        return out
class Residual_model(nn.Module):
    def __init__(self, batch_size = 128, num_channels = 1,num_frames = 34, height = 20, width = 20,device="cpu", Trans=False):

        super(Residual_model, self).__init__()
        self.features_1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=64,kernel_size= (2,128), stride=4, padding=0),
                                        nn.BatchNorm2d(64),
                                        nn.PReLU(),
        )
        self.features_2 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=32,kernel_size= (1,8),stride=8, padding=0),
                                        nn.BatchNorm2d(32),
                                        nn.PReLU(),
        )



        self.head = nn.Sequential(
                                    nn.Linear(2432,256),
                                    nn.PReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(256, 64),
                                    nn.PReLU(),
                                    nn.Dropout(0.5),
                                    nn.Linear(64,  2)
        )



        self.apply(_init_vit_weights)
        self.batchsize = batch_size
        self.num_channels = num_channels
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.device = device
        self.trans = Trans


    def ReadOFDMSymbol(self, data, in_batch=32, lengthofsymbol=2560):

        in_batch = data.size()[0]
        data = torch.squeeze(data)
        res = torch.zeros(in_batch, 1,2, lengthofsymbol)
        res[:,0,0, :] = data.real
        res[:,0,1, :] = data.imag

        
        return res



    def forward(self,x):
        x = self.ReadOFDMSymbol_otfs(x)
        x = self.features_1(x)
        x = self.features_2(x)

        x = torch.flatten(x,1)
        x = self.head(x)
        return x

def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

if __name__ == '__main__':
    data = scio.loadmat("./1.mat")
    python_y = np.array(data['sig_rec'])
    am = np.zeros([1,2560,1],dtype=complex)
    am[0,:,:] = python_y[:,:]
    am = torch.from_numpy(am)
    model = Residual_model()
    python_y = model.forward(am)
    print(python_y)
