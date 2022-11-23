import PIL.ImageShow
from torchvision import transforms
from PIL import Image
import torch

def ReadBVPmap(BVP_map, Trans=True):


    row, col = 6, 6
    BVP_frames = BVP_map[:, :, 0:20, 0:20]
    BVP_TEMP = torch.zeros(1, 1, 34, 20, 20)

    for i in range(row):
        for j in range(col):
            one_pic = BVP_map[:, :, 0 + 20 * j:20 + 20 * j, 0 + 20 * i:20 + 20 * i]
            if Trans:
                one_pic = transforms.RandomHorizontalFlip()(one_pic)
                one_pic = transforms.Resize(21)(one_pic)
                one_pic = transforms.RandomCrop((20,20))(one_pic)
                #                                transforms.CenterCrop(224),



            BVP_frames = torch.cat((BVP_frames, one_pic), dim=1)


    BVP_TEMP[:, 0, :, :, :] = BVP_frames[:, 2:36, :, :]


    return BVP_TEMP


image = Image.open('1.png')


image = transforms.ToTensor()(image)
# image = transforms.Resize(122)(image)
# image = transforms.RandomCrop((120,120))(image)
# image = transforms.ToPILImage()(image)
# PIL.ImageShow.show(image)
temp = torch.zeros((1, 1, 120, 120))
temp[0, 0, :, :] = image[:, :]

res = ReadBVPmap(temp,True)
print(res.size())

# data_transform = {
#     "train": transforms.Compose([transforms.RandomResizedCrop(224),
#                                  transforms.RandomHorizontalFlip(),
#                                  transforms.ToTensor(),
#                                  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
#     "val": transforms.Compose([transforms.Resize(256),
#                                transforms.CenterCrop(224),
#                                transforms.ToTensor(),
#                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
#


