import json
import csv
if __name__ == '__main__':
    # 1、读取csv中全部数据到字典中，key为图片名(不含路径和.jpg等),value为其对应打分
    name_score_dict = dict()
    with open('G:\图灵深视\dataset\IQA_Train_22_5_6_label.csv',encoding='utf-8') as file:
        f_csv = csv.reader(file)
        # print(type(f_csv))
        i = 0
        for a in f_csv:
            # a是的len=1的list
            i += 1
            l = a[0] # str
            l = l.split('!@#$%')
            name_score_dict[l[0]] = l[-2]
        print(len(name_score_dict))

    # 2、读取json数据：https://blog.csdn.net/qzonelaji/article/details/112908004
    json_path = 'G:\图灵深视\dataset\split_22_5_6.json'
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        # <class 'dict'>,JSON文件读入到内存以后，就是一个Python中的字典。
        # 字典是支持嵌套的，
        train_data = data['train'] # train_data为len=5120的list，对应每个图片的名字(不含路径，包括.jpg)
        test_data = data['test'] # test_data为len=746的list，对应每个图片的名字(不含路径，包括.jpg)

    # 3、根据json中的名字查找csv中相应图片的分数
    # trainpath = 'G:\图灵深视\MANIQA\data\IQAtrain.txt'
    # for train_name in train_data: # 5120张图片
    #     # train_name: France_105929---9JLVSH017_5.jpg_0.jpg
    #     query = train_name.split('.')
    #     left_index = train_name.rindex('.'+query[-1]) # 从右边开始查找'.jpe'出现的第一个索引
    #     name = train_name[:left_index] # 去除'.jpe', name: France_105929---9JLVSH017_5.jpg_0
    #     score = name_score_dict[name]
    #     savetext = train_name+", "+score
    #     with open(trainpath,'a',encoding='utf-8') as f:
    #         f.write(savetext+'\n')
    testpath = 'G:\图灵深视\MANIQA\data\IQAtest.txt'
    for test_name in test_data: # 746张图片
        # train_name: France_105929---9JLVSH017_5.jpg_0.jpg
        query = test_name.split('.')
        left_index = test_name.rindex('.'+query[-1]) # 从右边开始查找'.jpe'出现的第一个索引
        name = test_name[:left_index] # 去除'.jpe', name: France_105929---9JLVSH017_5.jpg_0
        score = name_score_dict[name]
        savetext = test_name+", "+score
        with open(testpath,'a',encoding='utf-8') as f:
            f.write(savetext+'\n')
