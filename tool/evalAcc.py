import sys
sys.path.append('./')
import os
import os.path as osp
import shutil
import numpy as np
from tqdm import tqdm
import traceback
import torch
from models.maniqa import MANIQA
from tabulate import tabulate
import cv2
from torchvision import transforms
from utils.process import RandResizeCrop,Normalize,ToTensor,five_point_crop
def initModel(modelfile = '/mnt/yue/YueIQA/output/models/model_maniqa/epoch7.pth'):
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

if __name__ == '__main__':
    img_root = '/mnt/lcs/data/CUT0608'
    extra_name = '_MANIQA'
    # extra_name = '_result_no_pretrain' # 原始linearityIQA测得结果
    img_root += extra_name
    # print(os.listdir(img_root))
    iqa_model = initModel()
    iqa_model.cuda()

    TP = 0
    FN = 0
    FP = 0
    TN = 0
    threshold = 29

    heads_list = ['nums_clear', 'nums_unclear', 'acc', 'precision', 'clear_recall', 'unclear_recall']
    for object_name in os.listdir(img_root):
        root_list = [osp.join(img_root + '/' + object_name, '模糊'), osp.join(img_root + '/' + object_name, '清晰')]
        acc_list = [0] * 6
        temp_tp, temp_fn, temp_fp, temp_tn, acc, precision, recall, unclear_recall, nums_unclear, nums_clear = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        for i, path in enumerate(root_list):
            namelist = os.listdir(path)
            label = path.split('/')[-1]  #
            if label == '模糊':
                label = 0
            elif label == '清晰':
                label = 1
            else:
                raise Exception('数据集文件命名不正确！')
            for name in tqdm(namelist):  # 当names不为空时
                # name: 安耐晒小金瓶拒鉴图片/模糊/12.455983_瓶盖顶部___baogo1654428733461447434.jpg
                query = name.split('_')
                if query[0]=='N':
                    pred = 0
                else:
                    score = float(query[0])
                    pred = score>threshold

                if label:
                    if pred:
                        temp_tp += 1
                    else:
                        temp_fn += 1
                    nums_clear += 1  # 当前类别下清晰图像总数
                else:
                    if pred:
                        temp_fp += 1
                    else:
                        temp_tn += 1
                    nums_unclear += 1  # 当前类别下模糊图像总数

        # print(temp_tp)
        # print(temp_fn)
        # print(temp_fp)
        # print(temp_tn)
        acc = (temp_tp + temp_tn) / (temp_tp + temp_fn + temp_fp + temp_tn)
        precision = (temp_tp) / (temp_tp + temp_fp)  # 清晰图像的查准率
        clear_recall = temp_tp / (temp_tp + temp_fn)  # 清晰图像的查全率
        unclear_recall = temp_tn / (temp_tn + temp_fp)  # 模糊图像的查全率
        acc_list[0], acc_list[1], acc_list[2], acc_list[3], acc_list[4], acc_list[
            5] = nums_clear, nums_unclear, acc, precision, clear_recall, unclear_recall
        show = tabulate([acc_list], headers=heads_list, tablefmt='orgtbl')
        print(f'当前类别: {object_name}')
        print(show+'\n')