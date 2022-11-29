import os
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from my_dataset import MyDataSet
from model import OTFS_OFDM_CNN as create_model
from utils import read_split_data, train_one_epoch, evaluate,read_test_data,test_model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    test_images_path, test_images_label = read_test_data(args.data_path) #保持创建文件夹的方式就行

    data_transform = {
        "test": transforms.Compose([
                                     transforms.ToTensor()])
    }

    # 实例化训练数据集
    test_dataset = MyDataSet(images_path=test_images_path,
                              images_class=test_images_label,
                              transform=data_transform["test"])

    # 实例化验证数据集


    batch_size = 15000
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))





    train_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=300,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw
                                    )



    model = create_model(device=device).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        print(model.load_state_dict(weights_dict, strict=False))



    test_model(model=model,
               data_loader=train_loader,
               device=device)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str,
                        default="E:\Projects\OTFS_MODULATIONS_IDENTIFICATION\OTFS_SYN\otfs_m_test")
    parser.add_argument('--model-name', default='bvpt_model', help='create model name')
    parser.add_argument('--weights', type=str, default='last.pth', help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    main(opt)


