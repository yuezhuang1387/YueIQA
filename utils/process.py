import torch
import numpy as np
import cv2

def log(x, n=10):
    # 以n为底,x的对数，使用换底公式计算
    return np.log(x)/np.log(n)

def random_crop(d_img, config):
    b, c, h, w = d_img.shape
    top = np.random.randint(0, h - config.crop_size)
    left = np.random.randint(0, w - config.crop_size)
    d_img_org = crop_image(top, left, config.crop_size, img=d_img)
    return d_img_org


def crop_image(top, left, patch_size, img=None):
    tmp_img = img[:, :, top:top + patch_size, left:left + patch_size]
    return tmp_img


def five_point_crop(idx, d_img, crop_size):
    '''
    截取图像左上、右上、左下、右下、中间的区域
    '''
    b, c, h, w = d_img.shape
    if idx == 0:
        top = 0
        left = 0
    elif idx == 1:
        top = 0
        left = w - crop_size
    elif idx == 2:
        top = h - crop_size
        left = 0
    elif idx == 3:
        top = h - crop_size
        left = w - crop_size
    elif idx == 4:
        center_h = h // 2
        center_w = w // 2
        top = center_h - crop_size // 2
        left = center_w - crop_size // 2
    d_img_org = crop_image(top, left, crop_size, img=d_img)

    return d_img_org


def split_dataset_kadid10k(txt_file_name, split_seed=20):
    np.random.seed(split_seed)
    object_data = []
    with open(txt_file_name, 'r') as listFile:
        for line in listFile:
            dis, score = line.split()
            dis = dis[:-1]
            if dis[1:3] not in object_data:
                object_data.append(dis[1:3])
    
    np.random.shuffle(object_data)
    np.random.seed(20)

    l = len(object_data)
    train_name = object_data[:int(l * 0.8)]
    val_name = object_data[int(l * 0.8):]
    return train_name, val_name


def split_dataset_tid2013(txt_file_name, split_seed=20):
    np.random.seed(split_seed)
    object_data = []
    with open(txt_file_name, 'r') as listFile:
        for line in listFile:
            score, dis = line.split()
            if dis[1:3] not in object_data:
                object_data.append(dis[1:3])
    
    np.random.shuffle(object_data)
    np.random.seed(20)

    l = len(object_data)
    train_name = object_data[:int(l * 0.8)]
    val_name = object_data[int(l * 0.8):]
    return train_name, val_name


def split_dataset_live(txt_file_name, split_seed=20):
    np.random.seed(split_seed)
    object_data = []
    with open(txt_file_name, 'r') as listFile:
        for line in listFile:
            i1, i2, ref, dis, score, h, w = line.split()
            if ref[8:] not in object_data:
                object_data.append(ref[8:])
    
    np.random.shuffle(object_data)
    np.random.seed(20)

    l = len(object_data)
    train_name = object_data[:int(l * 0.8)]
    val_name = object_data[int(l * 0.8):]
    return train_name, val_name


def split_dataset_csiq(txt_file_name, split_seed=20):
    np.random.seed(split_seed)
    object_data = []
    with open(txt_file_name, 'r') as listFile:
        for line in listFile:
            dis, score= line.split()
            dis_name, dis_type, idx_img, _ = dis.split(".")
            if dis_name not in object_data:
                object_data.append(dis_name)
    
    np.random.shuffle(object_data)
    np.random.seed(20)

    l = len(object_data)
    train_name = object_data[:int(l * 0.8)]
    val_name = object_data[int(l * 0.8):]
    return train_name, val_name


class RandResizeCrop(object):
    def __init__(self, crop_size=224, mode='train'):
        self.crop_size = crop_size
        self.mode = mode
    def __call__(self, sample):
        d_img = sample['d_img_org'] # (H,W,C) (numpy)
        score = sample['score']

        # 对输入图像的尺寸进行resize操作
        h, w, c = d_img.shape
        new_h,new_w = 0,0
        if h >self.crop_size and w>self.crop_size:
            # 全部大于224,对放大倍率进行衰减，保证衰减后的放大倍率仍然大于1
            new_r_h = 1+log(h/self.crop_size)
            new_r_w = 1+log(w/self.crop_size)
            if new_r_h<new_r_w:
                new_h = int(new_r_h*self.crop_size)+1 # 向上取整，保证new_h >=225
                new_w = int(new_h*w/h) # new_w一定>=225
            else:
                new_w = int(new_r_w*self.crop_size)+1 # 向上取整，保证new_w >=225
                new_h = int(new_w*h/w) # new_h一定>=225
        else:
            # 五种情况：一个>224，一个=224
            #         一个>224，一个<224
            #         一个=224，一个=224
            #         一个=224，一个<224
            #         一个<224，一个<224

            # 把最小边放大到225
            if h<w:
                new_h = self.crop_size+1
                new_w = int(w*new_h/h)
            else:
                new_w = self.crop_size+1
                new_h = int(h*new_w/w)
        new_img = cv2.resize(d_img,(new_w,new_h))
        top = np.random.randint(0, new_h - self.crop_size)
        left = np.random.randint(0, new_w - self.crop_size)
        if self.mode == 'train':
            ret_d_img = new_img[top: top + self.crop_size, left: left + self.crop_size,:]
        elif self.mode == 'eval':
            ret_d_img = new_img
        else:
            raise Exception(f'输入mode不正确!')

        sample = {
            'd_img_org': ret_d_img, # (crop_h,crop_w,C) (numpy)
            'score': score
        }
        return sample


class Normalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, sample):
        # r_img: (H,W,C) (numpy)
        d_img = sample['d_img_org']
        score = sample['score']
        # cv2.imshow('tt',cv2.cvtColor(d_img, cv2.COLOR_RGB2BGR))
        # cv2.waitKey()
        d_img = (d_img - self.mean) / self.var
        sample = {'d_img_org': d_img, 'score': score}
        return sample


class RandHorizontalFlip(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        d_img = sample['d_img_org']
        score = sample['score']
        prob_lr = np.random.random()
        # np.fliplr needs (H,W,C) numpy
        if prob_lr > 0.5:
            d_img = np.fliplr(d_img).copy()
        
        sample = {
            'd_img_org': d_img,
            'score': score
        }
        return sample


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        d_img = sample['d_img_org'] # (H,W,C) (numpy)
        score = sample['score'] # (1,)
        d_img = np.transpose(d_img, (2, 0, 1))  # (H,W,C)->(C,H,W)
        d_img = torch.from_numpy(d_img).type(torch.FloatTensor)
        score = torch.from_numpy(score).type(torch.FloatTensor)
        sample = {
            'd_img_org': d_img, # torch.Size([C,H,W])
            'score': score # torch.Size([1])
        }
        return sample

if __name__ == '__main__':
    print(log(4,256))
    patch_size = 224
    d_img = np.random.rand(3,140,91)
    c, h, w = d_img.shape
    # print(h/224)
    print(np.random.randint(0,0))
    top = np.random.randint(0, h - patch_size)
    left = np.random.randint(0, w - patch_size)
    ret_d_img = d_img[:, top: top + patch_size, left: left + patch_size]