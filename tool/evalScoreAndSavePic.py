import sys
sys.path.append('./')
import os
import os.path as osp
import shutil
import numpy as np
from tqdm import tqdm
import traceback
import torch
import torch.nn as nn
from models.maniqa import MANIQA
from tabulate import tabulate
import cv2
from torchvision import transforms
from utils.process import RandResizeCrop,Normalize,ToTensor,five_point_crop
def initModel(modelfile = '/mnt/yue/YueIQA/output_distributed_5866/models/model_maniqa/epoch158.pth'):
    model = MANIQA(embed_dim=768,
                   num_outputs=1,
                   dim_mlp=768,
                   patch_size=8,
                   img_size=224,
                   window_size=4,
                   depths=[2, 2],
                   num_heads=[4, 4],
                   num_tab=2,
                   scale=0.13)
    # 1、加载训练好的模型，文件加载后属于OrderedDict
    checkpoint = torch.load(modelfile,map_location='cpu')
    # print(type(checkpoint)) # <class 'collections.OrderedDict'>
    
    # 2、设置新的OrderedDict
    from collections import OrderedDict
    d = OrderedDict()
    
    # 3、删除k中的module字段，得到新的state_dict，命名为d
    for k,v in checkpoint.items():
        k_new = k.replace('module.','') # module.vit.cls_token -> vit.cls_token
        d[k_new] = v
    
    # 4、加载新的state_dict
    model.load_state_dict(d)
    return model

def predict(net, img_path, threshold = 0.6):
    net.eval()
    img = cv2.imread(img_path,cv2.IMREAD_COLOR)
    d_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    d_img = np.array(d_img).astype('float32') / 255
    score = np.array([1])
    sample = {
        'd_img_org': d_img,
        'score': score
    }
    transform = transforms.Compose(
        [
            RandResizeCrop(224,mode='eval'),
            Normalize(0.5, 0.5),
            ToTensor()
        ])
    data = transform(sample)
    score = 0
    for i in range(5):
        x_d = data['d_img_org'].unsqueeze(0)  # torch.Size([1,C,H,W])
        x_d = five_point_crop(i, d_img=x_d, crop_size=224)
        # 截取图像左上、右上、左下、右下、中间的区域进行预测，取均值
        score += net(x_d.cuda()) # torch.Size([1])
    
    # 综合预测分数
    score = score.data.cpu().numpy()/5 # score.shape: (1,)
    return score>=threshold, score*100

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

if __name__ == '__main__':
    img_root = '/mnt/lcs/data/CUT0608'
    extra_name = '_MANIQA_5866'
    # print(os.listdir(img_root))
    iqa_model = initModel()
    iqa_model = nn.DataParallel(iqa_model)
    iqa_model.cuda()
    for object_name in os.listdir(img_root):
        root_list = [osp.join(img_root + '/' + object_name, '模糊'), osp.join(img_root + '/' + object_name, '清晰')]
        save_list = [osp.join(img_root + extra_name + '/' + object_name, '模糊'),
                     osp.join(img_root + extra_name + '/' + object_name, '清晰')]
        print(f'检测类别: {object_name}')
        for i, path in enumerate(root_list):
            os.makedirs(save_list[i], exist_ok=True)
            namelist = os.listdir(path)
            label = path.split('/')[-1]  #
            if label == '模糊':
                label = 0
            elif label == '清晰':
                label = 1
            else:
                raise Exception('评价数据集文件命名不正确！')
            for name in tqdm(namelist):  # 当names不为空时
                if name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = osp.join(path, name)
                    pred, score = predict(iqa_model,img_path=img_path)  # 预测类别，评分
                    save_path = osp.join(save_list[i],
                                         'N_' + str(score[0]) + '_' if score < 0 else str(score[0]) + '_' + name)
                    # print(pred, score)
                    shutil.copy(img_path, save_path)