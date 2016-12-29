# -*- coding: utf-8 -*-
"""
Created on Mon Nov 07 22:50:32 2016

@author: dell
"""

import jieba
from sensitive import chi
from sensitive import train_test

corpus_dir = 'resources/test1.txt'       '''原始语料的地址'''
seg_corpus = 'resources/test_seg.txt'       '''分词之后的语料地址'''

def PreProcess():
    '''对已标注的语料进行预处理：分词'''
    f = open(corpus_dir, 'r')
    w_f = open(seg_corpus, 'w')
    for line in f:
        content = line.split(',')
        # label = line[0]
        label = content[6]
        # line = line[2:-1]
        line = content[5]
        seg_list = jieba.cut(line)
        new_line = ' '.join(seg_list)
        w_f.write(label + ' ' + new_line + '\n')
    
    f.close()
    w_f.close()

def GetResult():
    '''文本分类流程：语料集拆分、特征提取、分类器训练、测试'''
    
    '''语料集载入'''
    f = open(seg_corpus, 'r')
    corpus_dict = dict()
    content_label1 = list()
    content_label2 = list()
    content_label0 = list()
    
    for line in f:
        label = line[0]
        content = line[2:-1]
        
        if label == '0':
            content_label0.append(content)
        elif label == '1':
            content_label1.append(content)
        elif label == '2':
            content_label2.append(content)
            
    corpus_dict[0] = content_label0
    corpus_dict[1] = content_label1
    corpus_dict[2] = content_label2
    
    '''语料集分拆：训练集、测试集'''
    fold = 5
    train_corpus = dict()
    test_corpus = dict()
    
    for label in corpus_dict:
        content_list = corpus_dict[label]
        train_content = list()
        test_content = list()

        cnt = 0
        for line in content_list:
            cnt += 1
            if cnt % fold == 0:
                test_content.append(line)
            else:
                train_content.append(line)
        
        train_corpus[label] = train_content
        test_corpus[label] = test_content
        
    '''特征提取'''
    width = 3000
    character_list = chi.Get_character_list(width, train_corpus)
    
    character_dict = dict()             #特征词词典
    for character in character_list:
        character_dict[character] = 0
        
    print (len(character_dict))
    
    #训练集的向量转换
    w_ff = open('train_chi.txt', 'wb' )
    label_f = open('train.label', 'wb')
    
    for dict_label in train_corpus:
        content = train_corpus [dict_label]

        for line in content:   
            chi_str = chi.ConVec(line, character_dict)

            if chi_str != '':
                w_str = chi_str
                w_ff.write(w_str[:-1] + '\n')
                label_f.write(dict_label + '\n')
                
    w_ff.close()
    label_f.close()
    
    #测试集的向量转换
    w_ff = open('test_chi.txt', 'wb' )
    label_f = open('test.label', 'wb')
    
    for dict_label in test_corpus:
        content = train_corpus [dict_label]

        for line in content:   
            chi_str = chi.ConVec(line, character_dict)

            if chi_str != '':
                w_str = chi_str
                w_ff.write(w_str[:-1] + '\n')
                label_f.write(dict_label + '\n')
                
    w_ff.close()
    label_f.close()
    
    '''分类器训练与测试'''
    print ('Decision Tree is construct...')
    clf = train_test.train('train_chi.txt', 'train.label', 'D_tree')
    avr_acc = train_test.test(clf, 'test_chi.txt', 'test.label', 'predict_dt.txt', 'probab_pre.txt')
    print ('The acc of dt is ' + str(avr_acc) )
        
    
    
