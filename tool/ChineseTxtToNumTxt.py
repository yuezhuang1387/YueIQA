# 将已经整理好的txt训练和测试文件中的图片，重命名并存到另一个文件夹中，并在data文件下生成一个新的无中文的txt目录

testpath = 'G:\图灵深视\MANIQA\data\IQAtest.txt'
old_name_list = []
new_name_list = []
score_list = []
# 1、读取图片名称为中文的txt
with open(testpath,'r',encoding='utf-8') as f:
    for i, line in enumerate(f):
        name, score = line.split(', ')
        # name: France_105929---9JLVSH017_5.jpg_0.jpg
        query = name.split('.') # query[-1]对应图片的结尾格式；jpg/png/jpeg
        old_name_list.append(name)
        new_name_list.append(str(i+5120)+'.jpg')
        score_list.append(score)
# print(new_name_list)
# print(score_list)

import cv2
import os
import numpy as np
# 2、读取中文图片保存，使用数字名保存到新的文件夹下
root_read = 'G:\Turingdataset\IQA_Train_22_5_6_img'
root_write = 'G:\Turingdataset\IQA_Train_22_5_6_img_noChi'
for i in range(len(new_name_list)):
    # img = cv2.imread(os.path.join(root_read,old_name_list[i]))
    # 当imread无法读取中文路径时，使用imdecode
    img = cv2.imdecode(np.fromfile(
        os.path.join(root_read,old_name_list[i]),
        dtype=np.uint8), 1)  # 可读取中文路径图片
    cv2.imwrite(os.path.join(root_write,new_name_list[i]),img,[int(cv2.IMWRITE_JPEG_QUALITY),100]) # 保存最高质量图像
    # break

# 3、生成新的txt文件
new_train_txt = 'G:\图灵深视\MANIQA\data\IQAtest_noChi.txt'
for i in range(len(new_name_list)):
    with open(new_train_txt,'a',encoding='utf-8') as f:
        f.write(new_name_list[i]+', '+score_list[i]) # score中包含了换行字符


