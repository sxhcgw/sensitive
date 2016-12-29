# -*- coding: utf-8 -*-
"""
Created on Fri Apr 08 15:07:19 2016

@author: ailab_guoqi
"""
#本程序为文本分类时运用CHI算法进行特征选择的实现,输入的程序应当是已经分好词的
#每一类的语料放在一个txt文件中，将所有类别的文本放在一个文件夹下

import os
import operator

#计算每一类的词频：file_root是语料集所在文件夹的地址,start_num是正文起始位置
def CalTermFreq( data_dict ):
    Term_Freq_dict_list = list()     #存储每一类别的词频字典
    
    for dict_label in data_dict:
        content = data_dict[dict_label]
        
        Term_Freq_dict = dict()
        size_of_the_category = 0               #记录该类语料的大小
        
        for line in content:
            size_of_the_category += 1
            line_content = line[1].strip()
            temp_list = set(line_content.split(' '))
            
            for term in temp_list:              #对该行的词语在字典中进行查找，找到了就+1，如果没有就创建
                if term in Term_Freq_dict:
                    Term_Freq_dict [term] += 1.0
                else:
                    if term == '' or term == ' ':
                        pass
                    else:
                        Term_Freq_dict [term] = 1.0
        
        Term_Freq_dict['size_of_the_category'] = size_of_the_category  #将该类的语料大小保存
        Term_Freq_dict_list.append(Term_Freq_dict)     #将该类词频统计字典放入列表中
        
    return Term_Freq_dict_list
    
#计算每个词在不同类下的CHI得分
#对于卡方统计公式：CHI（词语，类标号）=(AD-BC)^2）/(A+B)(C+D)
#其中，A=该词在该类中出现的频次，B=该词在其他类中的频次，C是该类中不含该词的语料数，D是其他类中不含该词的语料数
def CalCHI(data_dict):
    Term_Freq_dict_list = CalTermFreq( data_dict )
    
    CHI_list = list()     #存放CHI的值
    
    #首先计算总文档数N
    N = 0.0
    for iterm in Term_Freq_dict_list:
        N += iterm['size_of_the_category']
    
    #计算每个词的CHI
    for iterm in Term_Freq_dict_list:
        CHI_dict = dict()
        size_of_this_category = iterm['size_of_the_category']   #得到A+C
        size_of_other_category = N - size_of_this_category      #得到B+D
        
        for word, A in iterm.iteritems():
            C = size_of_this_category - A
            B = 0.0
            for iterm in Term_Freq_dict_list:
                if word in iterm:
                    B += iterm[word]
            B = B - A       #因为计算时遍历了所有的，也就将A包含在内，故要减去
            D = size_of_other_category - B
            CHI_dict[word] = (A * D - B * C) * ( A * D - B * C) / (1.0 + (A + B) * (C + D))
        
        #对CHI字典进行降序排序，得到的结果是元组列表
        CHI_sort = sorted(CHI_dict.iteritems(), key=operator.itemgetter(1), reverse=True)
        CHI_list.append(CHI_sort)
    
    #print CHI_list
    return CHI_list

#由CHI值得排序选出特征词
def Get_character_list(top_n, data_dict):
    CHI_list = CalCHI(data_dict)
    character_list = list()
    
    for CHI_dict in CHI_list:
        count = 0
        for item in CHI_dict:
            count += 1
            if count <= top_n:
                character_list.append(item[0])
    
    return character_list

def ConVec(content, character_dict):
    text = content.strip()
    set_list = set(text.split())
    
    for word in character_dict:
        character_dict[word] = 0
            
    for word in set_list:
        if word in character_dict:
            character_dict[word] = 1
                    
    text_string = ''
    for word in character_dict:
        text_string += str(character_dict[word]) + ' '
            
    return text_string

    

if __name__ == '__main__':
    file_root = 'E:/guoqi/all_vocab/test/cut'
    source_root = 'E:/guoqi/all_vocab/test'
    start_num = 2
    top_n = 2000
    character_list = Get_character_list(top_n, file_root, start_num)   #得到特征词集合
    
    character_dict = dict()             #特征词词典
    for character in character_list:
        character_dict[character] = 0
    
    label_root = source_root + '/' + 'label.txt'
    text_root = source_root + '/' + 'text.txt'
    
    label_f = open(label_root, 'w')
    text_f = open(text_root, 'w')

    for file_name in os.listdir(file_root):
        file_dir = file_root + '/' + file_name
        
        f = open(file_dir, 'r')
        
        for line in f:
            for character in character_dict:
                character_dict[character] = 0

            label = line[0]
            text = line[start_num:-1]
            set_list = set(text.split())
            
            for word in set_list:
                if word in character_dict:
                    character_dict[word] = 1
                
            label_f.write(label + '\n')
            text_string = ''
            for word in character_dict:
                text_string += str(character_dict[word]) + ' '
            
            text_string = text_string[:-1] + '\n'
            text_f.write(text_string)

                
        
    
    
