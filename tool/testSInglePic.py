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
from utils.process import RandResizeCrop,ToTensor,five_point_crop
def initModel(modelfile = '/mnt/yue/YueIQA/output_distributed_5866/models/model_maniqa/epoch178.pth'):
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
    # d_img = np.array(d_img).astype('float32') / 255
    # d_img = cv2.resize(d_img,(224,224))
    score = np.array([1])
    sample = {
        'd_img_org': d_img,
        'score': score
    }
    transform = transforms.Compose(
        [
            RandResizeCrop(224,mode='eval'),
            ToTensor(0.5,0.5)
        ])
    data = transform(sample)
    score = 0
    for i in range(5):
        x_d = data['d_img_org'].unsqueeze(0)  # torch.Size([1,C,H,W])
        x_d = five_point_crop(i, d_img=x_d, crop_size=224)
        # 截取图像左上、右上、左下、右下、中间的区域进行预测，取均值
        score += net(x_d.cuda()) # torch.Size([1])
    
    # # 综合预测分数
    score = score.data.cpu().numpy()/5 # score.shape: (1,)
    # score = net(data['d_img_org'].unsqueeze(0).cuda()).data.cpu().numpy()
    return score>=threshold, score*100

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

if __name__ == '__main__':
    img_root = '/mnt/yue/Turingdataset/IQA_Train_22_5_6_img_noChi/8.jpg'
    extra_name = '_MANIQA_5866'
    # print(os.listdir(img_root))
    iqa_model = initModel()
    iqa_model = nn.DataParallel(iqa_model)
    iqa_model.cuda()
    output = predict(iqa_model,img_path=img_root)
    print(output)