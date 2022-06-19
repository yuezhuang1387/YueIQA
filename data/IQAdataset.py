from torch.utils.data import Dataset, DataLoader, Subset
from utils.process import RandResizeCrop, ToTensor, RandHorizontalFlip, Normalize, five_point_crop
import os
import torch
import numpy as np
import cv2
from torchvision import transforms


class IQAdataset(torch.utils.data.Dataset):
    def __init__(self, img_path, txt_file_name, transform):
        '''
        :param img_path: 数据集图片所在路径
        :param txt_file_name: txt文件所在路径
        :param transform: 图像的预处理和数据增强操作
        '''
        super(IQAdataset, self).__init__()
        self.img_path = img_path
        self.txt_file_name = txt_file_name
        self.transform = transform

        img_name_list, score_list = [], []
        with open(self.txt_file_name, 'r',encoding='utf-8') as listFile:
            for line in listFile:
                img_name, score = line.split(', ')
                # ['bagborrow___bag253___LaLianTou___Louis Vuitton---Croisette PM Shoulder Bag - FINAL SALE_752---3316263.jpg', '75.25']
                img_name_list.append(img_name)
                score_list.append(float(score))

        # reshape score_list (n,) -> (n,1)
        score_list = np.array(score_list)
        score_list = self.normalization(score_list)
        score_list = score_list.astype('float').reshape(-1, 1)

        self.data_dict = {'d_img_list': img_name_list, 'score_list': score_list}
        # print(self.data_dict['score_list'][0].shape) # [0.9] shape: (1,)

    def normalization(self, data):
        # 对分数归一化
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def __len__(self):
        return len(self.data_dict['d_img_list'])

    def __getitem__(self, idx):
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img = cv2.imread(os.path.join(self.img_path, d_img_name), cv2.IMREAD_COLOR)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255

        score = self.data_dict['score_list'][idx]
        sample = {
            'd_img_org': d_img,
            'score': score
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

if __name__ == '__main__':
    a = IQAdataset('G:\Turingdataset\IQA_Train_22_5_6_img_noChi', 'G:\图灵深视\MANIQA\data\IQAtrain_noChi.txt',
                transform=transforms.Compose(
                    [
                        RandResizeCrop(224),
                        Normalize(0.5, 0.5),
                        RandHorizontalFlip(),
                        ToTensor()
                    ]
                ))
    from tqdm import tqdm
    for i in tqdm(range(len(a))):
        a.__getitem__(i)
