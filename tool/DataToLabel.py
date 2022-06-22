import os
import json
import numpy
import PIL.Image as Image
import shutil

def compute_score(count):
    s = 0
    c = 0
    all_score = []
    for i in range(5):
        s += (i + 1) * count[i]
        c += count[i]
        for j in range(count[i]):
            all_score.append((i + 1))
    avg = s / c
    y = (avg - 1) * 99 / 4 + 1
    return '!@#$%'.join([str(y), str(numpy.std(numpy.array(all_score)))])

def move_img(src_imgpath, dest_dir):
    dstfile = os.path.join(dest_dir,src_imgpath.split('/')[-1])
    if not os.path.isfile(src_imgpath):
        print("%s not exist!"%(src_imgpath))
    else:
        if not os.path.exists(dest_dir):
            print("Construct Dir: ", dest_dir)
            os.makedirs(dest_dir)                #创建路径
        shutil.move(src_imgpath, dstfile)          #移动文件
        print("move %s -> %s"%(src_imgpath,dstfile))


if __name__ == '__main__':
    root = r'/mnt/msk/IQA/Data/Train_Data/Label_Org'         #标签地址
    img_src_root = r'/mnt/msk/IQA/Data/Train_Data/Data_Org'   #原图地址
    img_dst_root = r'/mnt/msk/IQA/Data/Train_Data/IQA_Train_22_5_6_img'       #用于训练的图片地址
    save_path = r'/mnt/msk/IQA/Data/Train_Data/IQA_Train_22_5_6_label_yue.csv'    #标签统计

    counter = dict()

    flag2idx = {
        "非常模糊": 0,
        "模糊": 1,
        "一般": 2,
        "清晰": 3,
        "非常清晰": 4
    }

    Unic_Name = ['360皮具商城___bag4755___ChanDiBiao___False---44923---ef3c462e4a36f14b0b0c70f8b155b56d',
                '360皮具商城___bag5714___SuoKouTou___False---50259---73fedd338665e1157de33d78b1b08104',
                '4.2___bag65___BAONEIPIPAI___AI5KAE98---baogo1646134780917190992',
                '4.2___bag78___TUXINGBIAO___AIMOOTAA---baogo1647400504550227436',
                'LV1___bag2040___出厂标号___0___单肩包---2---62296---4',
                'LV1___bag2259___五金-铆钉3___1___单肩包---2---74925---5',
                'LV1___bag4292___皮签-皮签2___0___手拿包---2---67905---4',
                'LV1___bag5973___五金-拉链头1___1___手提包---2---30188---3',
                'LV1___bag9412___五金-拉链头4___0___钱包---2---84639---5',
                'TUXINGBIAO___baogo1650445338236677696',
                'bagborrow___bag180___LaLianTou___Louis Vuitton---Wilshire MM Tote_936---9975585',
                'highbags___bag1361___LaLianTou___lv---Louis Vuitton Monogram Eclipse Danube Pm m43677_260---2820043',
                'highbags___bag455___PiQian___lv---Louis Vuitton Monogram Eclipse Canvas Backpack Explorer m40527_783---2081071',
                'panghu蔻驰___bag794___包内皮牌___0___True---6802448',
                '值哆少___bag265___ChanDiBiao___True---268306---df8a3e39558b77683dc41a348cbd25fe',
                '值哆少___bag54___MaoDing___False---272459---d1d4cb1a09c6e9483ecce18e42b80afa',
                '值多少___bag5069___MaoDing___路易威登---False---87137_697---7685290',
                '名品商城___bag1369___PiQian___LV男---False---顶级原单M此款GER---a92f8fd2-b64e-43c7-aef5-d1600e7c7c11---图片---46062a51-dec7-4eb0-8195-f01c863ccce0',
                '易包商城___bag44___PiQian___LV 路易威登---592733022_222---5831300',
                '耍大牌研究所鉴定神器___bag2___KEZISUOKOUTOU___False---5f4f65f03c0541754fa4d495---f02b6576996b56a72362',
                '衣二三___bag127___PiQian___Louis Vuitton---棕色牛皮压花中号手提包_877---4409914'
            ]

    img_src_dict = dict()

    #建立图片名称-地址对应字典
    print(os.walk(img_src_root))
    for path, ds, img_src_names in os.walk(img_src_root):
        print(f'path: \n{path}')
        print(f'ds: \n{ds}')
        print(f'img_src_names: \n{img_src_names}')
        # for img_src_name in img_src_names:
        #     if img_src_name.endswith('png') or img_src_name.endswith('jpg') or img_src_name.endswith('jpeg'):
        #         img_src_path = os.path.join(path, img_src_name)
        #         img = Image.open(img_src_path)
        #         if img.size[0]>=64 and img.size[1]>=64:
        #             img_name = os.path.splitext(img_src_name)[0]
        #             if img_name in Unic_Name:
        #                 pass
        #             else:
        #                 img_src_dict[img_name] = img_src_path
        #         else:
        #             print("Not Image: ", img_src_path)


    # for path, ds, names in os.walk(root):
    #     for name in names:
    #         if name.endswith('json'):
    #             img_name = os.path.splitext(name)[0]
    #             if img_name in img_src_dict:
    #                 #img_dst_path = os.path.join(img_dst_root, os.path.split(img_src_dict[img_name])[-1])
    #                 move_img(img_src_dict[img_name],img_dst_root)
    #                 if img_name not in counter:
    #                     counter[img_name] = [0] * 5

    #                 data = json.load(open(os.path.join(path,name), encoding='utf-8'))

    #                 for flag in data['flags']:
    #                     if data['flags'][flag]:
    #                         if img_name in Unic_Name:
    #                             if flag2idx[flag] == 1:
    #                                 print('Unic: ', os.path.join(path,name))
    #                         counter[img_name][flag2idx[flag]] += 1
    #                         print(os.path.join(path,name), flag)
    #         else:
    #             print(name)

    # fout = open(save_path, 'w', encoding='utf-8')
    # images = list(counter.keys())
    # images.sort()

    # for image in images:
    #     try:
    #         fout.write('!@#$%'.join([image] + list(map(str, counter[image])) + [compute_score(counter[image])]) + '\n')
    #     except(ZeroDivisionError):
    #         print("ZeroDivisionError: ", image)

    # fout.close()

