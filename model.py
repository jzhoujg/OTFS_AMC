import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import numpy as np
import scipy.io as scio
import torch

class SE_Moudule(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1):
        super(SE_Moudule, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(1, 76), stride=stride, padding=0)
        self.linear1 = nn.Linear(32, 8)
        self.relu1 = nn.PReLU()
        self.drop = nn.Dropout(0.2)
        self.linear2 = nn.Linear(8, 32)
        self.act = nn.Sigmoid()

    def forward(self, x):
        identity = x
        inbatch = x.size()[0]
        out = self.conv1(x)
        out = torch.flatten(out,1)
        out = self.linear1(out)
        out = self.relu1(out)
        out = self.drop(out)
        out = self.linear2(out)
        out = self.act(out)
        temp = torch.zeros(inbatch,32,1,1)
        temp[:,:,0,0] = out
        out = temp * identity

        return out


class OTFS_OFDM_CNN(nn.Module):

    def __init__(self, batch_size = 128, num_channels = 1,num_frames = 34, height = 20, width = 20,device="cpu", Trans=False):


        super(OTFS_OFDM_CNN, self).__init__()
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
        self.se_module = SE_Moudule(in_channel=32,out_channel=32)
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
        x = self.ReadOFDMSymbol(x)
        x = self.features_1(x)
        x = self.features_2(x)
        x = self.se_module(x)
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
    model = OTFS_OFDM_CNN()
    python_y = model.forward(am)
    print(python_y)