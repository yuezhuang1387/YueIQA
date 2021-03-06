'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-06-22 14:36:17
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-06-22 16:42:32
FilePath: /mnt/yue/YueIQA/data/IQAdataset.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import sys
sys.path.append('./')
from utils.process import ColorJitter, RandResizeCrop, ToTensor, RandHorizontalFlip, Myrotate
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
        d_img = cv2.imread(os.path.join(self.img_path, d_img_name), cv2.IMREAD_COLOR) # (H,W,C) unit8类型
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        score = self.data_dict['score_list'][idx]
        sample = {
            'd_img_org': d_img,
            'score': score
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

if __name__ == '__main__':
    a = IQAdataset('/mnt/yue/Turingdataset/IQA_Train_22_5_6_img_noChi', '/mnt/yue/YueIQA/data/IQAtrain_noChi.txt',
                transform=transforms.Compose(
                    [
                        ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
                        RandResizeCrop(224),
                        RandHorizontalFlip(),
                        ToTensor(0.5,0.5),
                        Myrotate(angle=180,p=1)
                    ]
                ))
    from tqdm import tqdm
    for i in tqdm(range(len(a))):
        a.__getitem__(i)
