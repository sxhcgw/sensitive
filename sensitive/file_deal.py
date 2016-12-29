# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:48:34 2016

@author: ailab_guoqi
"""
import jieba.analyse

#加载语料文本的程序模块
def load_files(file_name):
    '''加载文件中的语料内容，每条语料存为[label,content]的形式，然后加入到data_list中'''
    f = open(file_name, 'r')
    data_list = list()            #每一个元素是一个列表[label,content]
    for line in f:
        tuple_list = list()
        label = line[0]
        content = line[2:-1]
        tuple_list.append(label)
        tuple_list.append(content)
        data_list.append(tuple_list)        
    f.close()
    
    return data_list
    
def load_character(file_name):
    '''加载文件中的语料内容，并按照字进行拆分；每条语料存为[label,content]的形式，然后加入到data_list中'''
    f = open(file_name, 'r')
    data_list = list()            #每一个元素是一个列表[label,content]
    for line in f:
        tuple_list = list()
        c_list = list()
        
        label = line[0]
        content = line[2:-1]
        for i in range(0, len(content)):
            c_list.append(content[i])
        new_content = ' '.join(c_list)
        
        tuple_list.append(label)
        tuple_list.append(new_content)
        data_list.append(tuple_list)        
    f.close()
    
    return data_list
    
    
def keyword_load_files(file_name, topK):
    '''加载文件中的语料内容，并抽取每条语料的关键字。每条语料存为[label,content]的形式，然后加入到data_list中'''
    '''file_name:文件地址, topK:抽取的关键字的个数'''
    f = open(file_name, 'r')
    data_list = list()            #每一个元素是一个列表[label,content]
    for line in f:
        tuple_list = list()
        label = line[0]
        content = ' '.join(jieba.analyse.extract_tags(line[2:-1], topK))
        tuple_list.append(label)
        tuple_list.append(content)
        data_list.append(tuple_list)        
    f.close()
    
    return data_list

    
