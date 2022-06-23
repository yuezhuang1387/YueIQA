import os
import torch
import numpy as np
import logging
import time
import torch.nn as nn
import random

from torchvision import transforms
from torch.utils.data import DataLoader
from models.maniqa import MANIQA,getMANIQA_vit_small_patch8_224_dino
from config import Config
from utils.process import RandResizeCrop, ToTensor, RandHorizontalFlip, five_point_crop, Myrotate, ColorJitter
from scipy.stats import spearmanr, pearsonr
from data.IQAdataset import IQAdataset
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_logging(config):
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    filename = os.path.join(config.log_path, config.log_file)
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode='w',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )


def train_epoch(epoch, net, criterion, optimizer, scheduler, train_loader):
    losses = []
    net.train()
    # save data for one epoch
    pred_epoch = []
    labels_epoch = []
    
    for data in tqdm(train_loader):
        x_d = data['d_img_org'].cuda() # torch.Size([N,C,H,W])
        labels = data['score'] # torch.Size([N,1])
        labels = torch.squeeze(labels).cuda()  # torch.Size([N])
        pred_d = net(x_d) # torch.Size([N])
        optimizer.zero_grad()
        loss = criterion(pred_d, labels)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()

        # save results in one epoch
        pred_batch_numpy = pred_d.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)
    
    # compute correlation coefficient
    print(pred_epoch.shape)
    print(labels_epoch.shape)

    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

    ret_loss = np.mean(losses)
    logging.info('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}'.format(epoch + 1, ret_loss, rho_s, rho_p))

    return ret_loss, rho_s, rho_p


def eval_epoch(config, epoch, net, criterion, test_loader):
    with torch.no_grad():
        losses = []
        net.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []

        for data in tqdm(test_loader):
            pred = 0
            for i in range(config.num_avg_val):
                x_d = data['d_img_org'].cuda() # torch.Size([N,C,H,W])
                labels = data['score'] # torch.Size([N,1])
                labels = torch.squeeze(labels).cuda() # torch.Size([N])
                x_d = five_point_crop(i, d_img=x_d, crop_size=config.crop_size)
                pred += net(x_d)

            pred /= config.num_avg_val
            # compute loss
            loss = criterion(torch.squeeze(pred), labels)
            losses.append(loss.item())

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)
        
        # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

        logging.info('Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}'.format(epoch + 1, np.mean(losses), rho_s, rho_p))
        return np.mean(losses), rho_s, rho_p

# 设置准备使用的GPU编号
os.environ['CUDA_VISIBLE_DEVICES'] = '2,0'

if __name__ == '__main__':
    # cpu_num = 1
    # os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    # os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    # os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    # os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    # os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    # torch.set_num_threads(cpu_num)

    setup_seed(20)

    # config file
    config = Config({
        # dataset path
        "db_name": "IQA",
        "train_dis_path": "/mnt/yue/Turingdataset/IQA_Train_22_5_6_img_noChi",
        # "val_dis_path": "/mnt/yue/Turingdataset/IQA_Train_22_5_6_img_noChi",
        "train_txt_file_name": "/mnt/yue/YueIQA/data/IQAtrain_noChi_5866.txt",
        # "val_txt_file_name": "/mnt/yue/YueIQA/data/IQAtest_noChi.txt",

        # optimization
        "batch_size": 16, # 原始：8
        "learning_rate": 0.5e-5, # 原始：1e-5,4卡推荐0.5e-5
        "weight_decay": 1e-5,
        "n_epoch": 300, # 原始：300
        "val_freq": 1,
        "T_max": 50,
        "eta_min": 0,
        "num_avg_val": 5,
        "crop_size": 224,
        "num_workers": 32, # 原始：8

        # model
        "patch_size": 8,
        "img_size": 224,
        "embed_dim": 768,
        "dim_mlp": 768,
        "num_heads": [4, 4],
        "window_size": 4,
        "depths": [2, 2],
        "num_outputs": 1,
        "num_tab": 2,
        "scale": 0.13,
        
        # load & save checkpoint
        "model_name": "model_maniqa",
        "output_path": "./output_distributed_5866",
        "snap_path": "./output_distributed_5866/models/",               # directory for saving checkpoint
        "log_path": "./output_distributed_5866/log/maniqa/",
        "log_file": ".txt",
        "tensorboard_path": "./output_distributed_5866/tensorboard/"
    })

    if not os.path.exists(config.output_path):
        os.mkdir(config.output_path)

    if not os.path.exists(config.snap_path):
        os.mkdir(config.snap_path)
    
    if not os.path.exists(config.tensorboard_path):
        os.mkdir(config.tensorboard_path)

    currentTime = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    config.snap_path += config.model_name
    config.log_file = config.model_name + '_'+currentTime+config.log_file
    config.tensorboard_path += config.model_name

    set_logging(config)
    logging.info(config)

    writer = SummaryWriter(config.tensorboard_path)

    # data load
    train_dataset = IQAdataset(
        img_path=config.train_dis_path,
        txt_file_name=config.train_txt_file_name,
        transform=transforms.Compose(
            [
                ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
                RandResizeCrop(config.crop_size),
                RandHorizontalFlip(),
                ToTensor(0.5,0.5),
                Myrotate(angle=180,p=1)
            ]
        ),
    )
    # val_dataset = IQAdataset(
    #     img_path=config.val_dis_path,
    #     txt_file_name=config.val_txt_file_name,
    #     transform=transforms.Compose(
    #         [
    #             RandResizeCrop(config.crop_size,mode='eval'), # 测试图像同样需要进行一个缩放，但此处不用crop
    #             Normalize(0.5, 0.5),
    #             ToTensor()
    #         ]
    #     ),
    # )

    logging.info('number of train scenes: {}'.format(len(train_dataset)))
    # logging.info('number of val scenes: {}'.format(len(val_dataset)))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=True
    )
    # val_loader = DataLoader(
    #     dataset=val_dataset,
    #     batch_size=1,
    #     num_workers=config.num_workers,
    #     drop_last=True,
    #     shuffle=False
    # )
    net = MANIQA(
        embed_dim=config.embed_dim,
        num_outputs=config.num_outputs,
        dim_mlp=config.dim_mlp,
        patch_size=config.patch_size,
        img_size=config.img_size,
        window_size=config.window_size,
        depths=config.depths,
        num_heads=config.num_heads,
        num_tab=config.num_tab,
        scale=config.scale
    )
    # net = nn.DataParallel(net)
    net = nn.DataParallel(net,device_ids=[0,1],output_device=0)
    net = net.cuda()

    # loss function
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)
    # 余弦退火学习率 https://zhuanlan.zhihu.com/p/336673856

    # make directory for saving weights
    if not os.path.exists(config.snap_path):
        os.mkdir(config.snap_path)

    # train & validation
    best_srocc = 0
    best_plcc = 0
    for epoch in range(0, config.n_epoch):
        start_time = time.time()
        logging.info('Running training epoch {}'.format(epoch + 1))
        loss_train, rho_s, rho_p = train_epoch(epoch, net, criterion, optimizer, scheduler, train_loader)
        # break
        writer.add_scalar("Train_loss", loss_train, epoch)
        writer.add_scalar("SRCC", rho_s, epoch)
        writer.add_scalar("PLCC", rho_p, epoch)

        net.eval()
        # 此处去掉验证集，将全部数据用于训练，保存精度最高的模型
        if rho_s>best_srocc or rho_p>best_plcc:
            best_srocc = rho_s
            best_plcc = rho_p
            # save weights
            model_name = "epoch{}".format(epoch + 1)+'.pth'
            model_save_path = os.path.join(config.snap_path, model_name)
            torch.save(net.state_dict(), model_save_path)
            logging.info('Saving weights and model of epoch{}, SRCC:{}, PLCC:{}'.format(epoch + 1, best_srocc, best_plcc))
        # 默认保存最新的模型
        torch.save(net.state_dict(), os.path.join(config.snap_path, 'new.pth'))
        logging.info('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))