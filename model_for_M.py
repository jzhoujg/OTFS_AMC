import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn

import numpy as np
import scipy.io as scio
import torch

class OTFS_OFDM_CNN(nn.Module):

    def __init__(self, batch_size = 128, num_channels = 1,num_frames = 34, height = 20, width = 20,device="cpu", Trans=False):

        super(OTFS_OFDM_CNN, self).__init__()

        self.features_1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=64,kernel_size= (1,128), stride=1, padding=0),
                                        nn.BatchNorm2d(64),
                                        nn.PReLU(),
        )

        self.features_2 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=16,kernel_size= (2,8),stride=8, padding=0),
                                        nn.BatchNorm2d(16),
                                        nn.PReLU(),
        )



        self.head = nn.Sequential(
                                    nn.Linear(4864,512),
                                    nn.PReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(512, 64),
                                    nn.PReLU(),
                                    nn.Dropout(0.5),
                                    nn.Linear(64,  5)
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

    def ReadOFDMSymbol_otfs(self, data, in_batch=32, lengthofsymbol=2560):

        in_batch = data.size()[0]
        data = torch.squeeze(data)
        data = data.view(in_batch,4,80,8)
        data = data.transpose(2,3)
        data = data.flatten()
        data = data.view(in_batch,2560)
        res = torch.zeros(in_batch, 1,2, lengthofsymbol)

        res[:,0,0, :] = data.real
        res[:,0,1, :] = data.imag


        return res


    def forward(self,x):
        x = self.ReadOFDMSymbol(x)
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
    model = OTFS_OFDM_CNN()
    python_y = model.forward(am)

    print(python_y)

