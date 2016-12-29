# -*- coding: utf-8 -*-
"""
Created on Sat May 14 16:14:29 2016

@author: ailab_guoqi

function:
使用动态词典的方法对文本进行情感分类
主要包括预处理、情感强度词典的构建、生成情感向量、训练与测试等步骤
"""
from sensitive import file_deal
import os
from sensitive import train_test
from datetime import datetime
from sensitive import IG

def mkdir(path):
    # 引入模块
    import os
 
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
 
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os.path.exists(path)
 
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        print (path+' 创建成功')
        # 创建目录操作函数
        os.makedirs(path)
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print (path+' 目录已存在')
        return False
        
def main(root_file, width, n_fold):
    #Step 1:先对待分词的语料进行分词和去停用词处理
    data_file = root_file + '/binary'        #分词之后的语料

    #Step 2:将分词之后的文本语料集加载到语料字典中
    all_data = dict()        #存放全部的数据
    file_dict = dict()        #存放文件名与dict_label的对应关系

    '''
    all_data的格式：
    {dict_label1：[[label1,content1],[label1,content1],[label1,content1],...],
     dict_label2：[[label2,content2],[label2,content1],[label2,content1],...],
    ....
    }
    file_dict的格式：{dict_label1：filename1,dict_label2：filename2,...}
    '''

    #载入文本数据
    print ('[%s] Load data_set to all_dict...' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    cnt = 1
    for file_name in os.listdir(data_file):
        #每次加载文件内容时，都要调用load_files()
        print (file_name)
        file_dir = data_file + '/' + file_name

        all_data[cnt] = file_deal.load_files(file_dir)
    
        file_dict[cnt] = file_name
        cnt += 1

    print (file_dict)
    print ('[%s] Load data_set has done!' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    #Step3：将文本语料拆分为训练集与测试集,并用训练集训练分类器，用测试集对分类器进行测试
    print ('[%s] Start the data_set calculate...' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    avr_acc = {'D_tree':0.0, 'svm-rbf':0.0, 'LR':0.0, 'svm-linear':0.0}

    chunk_len_dict = dict()      #存放每一类语料的每一折的大小

    for dict_label in all_data:
        chunk_len_dict[dict_label] = len(all_data[dict_label]) / n_fold

    print (chunk_len_dict)

    '''构建存储实验结果的文件'''
    #先创建本次实验结果存放的文件夹
    new_dir = root_file + '/' + 'fold_' + str(n_fold) + '/' + str(width)
    mkdir(new_dir)
    #以下三个存放动态词典的结果
    DT_f = open(new_dir + '/' + 'DT_result.txt', 'wb')
    rbf_f = open(new_dir + '/' + 'rbf_result.txt', 'wb')
    linear_f = open(new_dir + '/' + 'linear_result.txt', 'wb')
    LR_f = open(new_dir + '/' + 'LR_result.txt', 'wb')


    '''构建保存标签结果的文件'''
    #以下三个存放动态词典的标签
    DT_label = open(new_dir + '/' + 'DT.label', 'wb')
    rbf_label = open(new_dir + '/' + 'rbf.label', 'wb')
    linear_label = open(new_dir + '/' + 'linear.label', 'wb')
    LR_label = open(new_dir + '/' + 'LR.label', 'wb')


    total_label = open(new_dir + '/' + 'all_test.label', 'wb')
    
    for i in range(0, n_fold):
        print ('this is ' + str(i + 1) + ' fold\n')
        train_data = dict()        #存放训练语料
        test_data = dict()         #存放测试语料

        #分离测试集与训练集
        print ('[%s] 1. Seperate the test_set from train_set:' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        for dict_label in all_data:
            if i != n_fold - 1:
                start_num = chunk_len_dict[dict_label] * i
                end_num = start_num + chunk_len_dict[dict_label]
            
                test_data [dict_label] = all_data [dict_label][start_num:end_num]
                train_data [dict_label] = all_data[dict_label][0:start_num] + all_data[dict_label][end_num:]

            else:
                start_num = chunk_len_dict[dict_label] * i
            
                test_data [dict_label] = all_data [dict_label][start_num:]
                train_data [dict_label] = all_data[dict_label][0:start_num]

        '''
        根据训练集的文本，得到情感强度字典 
        '''
        print ('[%s] 2. Construct the IG value:' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print ('Senti_value_dict...')
        character_list = IG.Get_character_list(width, train_data)

        character_dict = dict()             #特征词词典
        for character in character_list:
            character_dict[character] = 0
        
        print (len(character_dict))
    
        '''将训练集文本转换为向量形式'''
        print ('[%s] 3. Convert the train_set to vectors:' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        w_ff = open(new_dir + '/' + 'train_IG.txt', 'wb' )
        label_f = open(new_dir + '/' + 'train.label', 'wb')
   
        for dict_label in train_data:
            content = train_data [dict_label]

            for line in content:
                label = line[0]
                line_content = line[1].strip()
                
                IG_str = IG.ConVec(line_content, character_dict)

                if IG_str != '':
                    w_str = IG_str
                    w_ff.write(w_str[:-1] + '\n')
                
                    label_f.write(label + '\n')
                
        w_ff.close()
        label_f.close()
                
        '''将测试集文本转换为向量形式'''
        print ('[%s] 4. Convert the test_set to vectors:' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        w_ff = open(new_dir + '/' + 'test_IG.txt', 'wb' )
        label_f = open(new_dir + '/' + 'test.label', 'wb')
    
        for dict_label in test_data:
            content = test_data [dict_label]

            for line in content:
                label = line[0]
                line_content = line[1].strip()
                
                IG_str = IG.ConVec(line_content, character_dict)

                if IG_str != '':
                    w_str = IG_str
                    w_ff.write(w_str[:-1] + '\n')
                
                    label_f.write(label + '\n')
                
        w_ff.close()
        label_f.close()
    
        print ('[%s] 5. Training and Testing:' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
        train_test.label_save(new_dir + '/' + 'test.label', total_label)     #对每次的测试集标签进行存储

        print ('[%s] Decision Tree is construct...' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        clf = train_test.train(new_dir + '/' + 'train_IG.txt', new_dir + '/' + 'train.label', 'D_tree')
        avr_acc['D_tree'] += train_test.test(clf, new_dir + '/' + 'test_IG.txt', new_dir + '/' + 'test.label', new_dir + '/' + 'predict.txt', new_dir + '/' + 'probab_pre.txt', DT_f) / n_fold
        train_test.label_save(new_dir + '/' + 'predict.txt', DT_label)

        print ('[%s] SVM-RBF is construct...' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        clf = train_test.train(new_dir + '/' + 'train_IG.txt', new_dir + '/' + 'train.label', 'svm-rbf')
        avr_acc['svm-rbf'] += train_test.test(clf, new_dir + '/' + 'test_IG.txt', new_dir + '/' + 'test.label', new_dir + '/' + 'predict.txt', new_dir + '/' + 'probab_pre.txt', rbf_f) / n_fold
        train_test.label_save(new_dir + '/' + 'predict.txt', rbf_label)
        
        print ('[%s] SVM-linear is construct...' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        clf = train_test.train(new_dir + '/' + 'train_IG.txt', new_dir + '/' + 'train.label', 'svm-linear')
        avr_acc['svm-linear'] += train_test.test(clf, new_dir + '/' + 'test_IG.txt', new_dir + '/' + 'test.label', new_dir + '/' + 'predict.txt', new_dir + '/' + 'probab_pre.txt', linear_f) / n_fold
        train_test.label_save(new_dir + '/' + 'predict.txt', linear_label)

        print ('[%s] Logistic Regression is construct...' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        clf = train_test.train( new_dir + '/' + 'train_IG.txt', new_dir + '/' + 'train.label', 'LR')
        avr_acc['LR'] += train_test.test(clf, new_dir + '/' + 'test_IG.txt', new_dir + '/' + 'test.label', new_dir + '/' + 'predict.txt', new_dir + '/' + 'probab_pre.txt', LR_f) / n_fold
        train_test.label_save(new_dir + '/' + 'predict.txt', LR_label)

    print (avr_acc)

    DT_f.close()
    rbf_f.close()
    LR_f.close()

    DT_label.close()
    rbf_label.close()
    LR_label.close()

    total_label.close()

    print ('[%s] The total result is ' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    f = open('total_result.txt', 'w')
    f.close()

def write_result(root_file, width, n_fold):
    new_dir = root_file + '/' + 'fold_' + str(n_fold) + '/' + str(width)
    f = open(new_dir + '/' + 'total_result.txt', 'w')

    f.write('SVM-RBF(IG) result:\n')
    train_test.print_result(new_dir + '/' + 'rbf.label', new_dir + '/' + 'all_test.label', f)
    f.write('SVM-linear(IG) result:\n')
    train_test.print_result(new_dir + '/' + 'linear.label', new_dir + '/' + 'all_test.label', f)
    f.write('LR(IG) result:\n')
    train_test.print_result(new_dir + '/' + 'LR.label', new_dir + '/' + 'all_test.label', f)
    f.write('DT(IG) result:\n')
    train_test.print_result(new_dir + '/' + 'DT.label', new_dir + '/' + 'all_test.label', f)

    f.close()