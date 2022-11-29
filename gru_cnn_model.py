import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn

import numpy as np
import scipy.io as scio
import torch

class GRU_CNN_MODEL(nn.Module):

    def __init__(self, batch_size = 128, num_channels = 1,num_frames = 34, height = 20, width = 20,device="cpu", Trans=False):

        super(GRU_CNN_MODEL, self).__init__()
        self.features_1 = nn.Sequential(            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(2, 128), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 8), stride=8,
                      padding=0),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )


        self.head_1 = nn.Sequential(
                                    nn.Linear(2048,256),
                                    nn.PReLU(),
                                    nn.Linear(256, 64),
                                    nn.PReLU(),

        )

        self.features_2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(2, 128), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 8), stride=8,
                      padding=0),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            )

        self.head_2 = nn.Sequential(
            nn.Linear(2048, 256),
            nn.PReLU(),
            nn.Linear(256, 64),
            nn.PReLU(),

        )

        self.features_3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(2, 128), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 8), stride=8,
                      padding=0),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            )

        self.head_3 = nn.Sequential(
            nn.Linear(2048, 256),
            nn.PReLU(),
            nn.Linear(256, 64),
            nn.PReLU(),
        )

        self.features_4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(2, 128), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 8), stride=8,padding=0),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            )

        self.head_4 = nn.Sequential(
            nn.Linear(2048, 256),
            nn.PReLU(),
            nn.Linear(256, 64),
            nn.PReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(512, 64),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )

        self.gru = nn.GRU(64,64,4,batch_first=True,bidirectional=True)

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
        tem1 = torch.zeros(in_batch, 1,2, 640)
        tem2 = torch.zeros(in_batch, 1,2, 640)
        tem3 = torch.zeros(in_batch, 1,2, 640)
        tem4 = torch.zeros(in_batch, 1,2, 640)
        tem1[:,0,:,:] = res[:,0,:,:640]
        tem2[:,0,:,:] = res[:,0,:,640:1280]
        tem3[:,0,:,:] = res[:,0,:,1280:1920]
        tem4[:,0,:,:] = res[:,0,:,1920:]

        return tem1,tem2,tem3,tem4


    def forward(self,x):

        x1,x2,x3,x4 = self.ReadOFDMSymbol(x)
        in_batch = x1.size()[0]
        temp = torch.zeros(in_batch,4,64)

        x1 = self.features_1(x1)
        x1 = torch.flatten(x1,1)
        x1 = self.head_1(x1)
        x2 = self.features_2(x2)
        x2 = torch.flatten(x2,1)
        x2 = self.head_2(x2)
        x3 = self.features_3(x3)
        x3 = torch.flatten(x3,1)
        x3 = self.head_3(x3)
        x4 = self.features_4(x4)
        x4 = torch.flatten(x4,1)
        x4 = self.head_4(x4)



        temp[:, 0, :] = x1
        temp[:, 1, :] = x2
        temp[:, 2, :] = x3
        temp[:, 3, :] = x4
        _, h = self.gru(temp)
        x = h.transpose(0,1)
        x = torch.flatten(x,1)
        x = self.head(x)



        # x1 = self.features_1(x1)
        # x1 = self.features_2(x1)
        # x1 = torch.flatten(x1,1)
        # x1 = self.head(x1)

        # x = self.features_2(x)
        # x = torch.flatten(x,1)
        # x = self.head(x)
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
    model = GRU_CNN_MODEL()
    python_y = model.forward(am)
    print(python_y)



