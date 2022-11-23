import numpy as np
import scipy.io as scio
import torch

def ReadOFDMSymbol(data,in_batch=1,lengthofsymbol=320):
    data = torch.from_numpy(data)
    data = torch.squeeze(data)
    res = torch.zeros(in_batch,2,lengthofsymbol)


    res[:,0,:] = data.real
    res[:,1,:] = data.imag
    return res

data = scio.loadmat("./1.mat")
python_y = np.array(data['rx_channel'])
python_y = ReadOFDMSymbol(python_y,1,258)

print(python_y)