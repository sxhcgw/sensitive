# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 16:14:50 2016

@author: ailab
"""
#基于信息增益的特征提取
import math
import operator

#计算每一类的词频：file_root是语料集所在文件夹的地址,start_num是正文起始位置
def CalTermFreq(data_dict):
    Term_Freq_dict_list = list()     #存储每一类别的词频字典
    
    for dict_label in data_dict:
        content = data_dict[dict_label]
        
        Term_Freq_dict = dict()
        size_of_the_category = 0.0               #记录该类语料的大小
        
        for line in content:
            size_of_the_category += 1.0
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

#
def CalInfoGain(data_dict):
    Term_Freq_dict_list = CalTermFreq(data_dict)
    
    IG_list = list()     #存放CHI的值
    
    #首先计算总文档数N
    N = 0.0
    for iterm in Term_Freq_dict_list:
        N += iterm['size_of_the_category']
    
    #计算每个词的IG
    for iterm in Term_Freq_dict_list:
        IG_dict = dict()
        size_of_this_category = iterm['size_of_the_category']   #得到该类语料的大小
        size_of_other_category = N - size_of_this_category      #得到其他类语料的大小
        
        print (size_of_this_category)
        print (size_of_other_category)
        
        #首先计算信息熵
        lamda1 = size_of_this_category / N
        lamda2 = size_of_other_category / N
        
        print ('lamda1:' + str(lamda1))
        print ('lamda2:' + str(lamda2))
        
        Entropy = -( lamda1 * math.log ( lamda1 ) + lamda2 * math.log ( lamda2 ) )
        print (Entropy)
        
        for word, A in iterm.iteritems():
            if word == 'size_of_the_category':
                pass
            else:
                C = size_of_this_category - A
                B = 0
                for iterm in Term_Freq_dict_list:
                    if word in iterm:
                        B += iterm[word]
                B = B - A       #因为计算时遍历了所有的，也就将A包含在内，故要减去
                D = size_of_other_category - B
                
                X = C + D
                if (X - 0.0) < 0.1:
                    print ('xxx' + word + 'xxx')
                    print (X)
            
                #计算IG值
                beita1 = ( A + B ) / N
                beita2 = A / ( A + B )
                beita3 = B / ( A + B )
                beita4 = ( C + D ) / N
                beita5 = C / ( C + D )
                beita6 = D / ( C + D )

                if A == 0:
                    if D == 0:
                        IG_dict[word] = Entropy + beita1 * ( beita3 * math.log( beita3 )) + beita4 * ( beita5 * math.log( beita5 ))
                    else:
                        IG_dict[word] = Entropy + beita1 * ( beita3 * math.log( beita3 )) + beita4 * ( beita5 * math.log( beita5 ) + beita6 * math.log( beita6 ))
                elif B == 0:
                    if C == 0:
                        IG_dict[word] = Entropy + beita1 * ( beita2 * math.log( beita2 )) + beita4 * ( beita6 * math.log( beita6 ))
                    else:
                        IG_dict[word] = Entropy + beita1 * ( beita2 * math.log( beita2 )) + beita4 * ( beita5 * math.log( beita5 ) + beita6 * math.log( beita6 ))
                elif C == 0:
                    if B == 0:
                        IG_dict[word] = Entropy + beita1 * ( beita2 * math.log( beita2 )) + beita4 * ( beita6 * math.log( beita6 ))
                    else:
                        IG_dict[word] = Entropy + beita1 * ( beita2 * math.log( beita2 ) + beita3 * math.log( beita3 )) + beita4 * ( beita6 * math.log( beita6 ))
                elif D == 0:
                    if A == 0:
                        IG_dict[word] = Entropy + beita1 * ( beita3 * math.log( beita3 )) + beita4 * ( beita5 * math.log( beita5 ))
                    else:
                        IG_dict[word] = Entropy + beita1 * ( beita2 * math.log( beita2 ) + beita3 * math.log( beita3 )) + beita4 * ( beita5 * math.log( beita5 ))
                else:
                    IG_dict[word] = Entropy + beita1 * ( beita2 * math.log( beita2 ) + beita3 * math.log( beita3 )) + beita4 * ( beita5 * math.log( beita5 ) + beita6 * math.log( beita6 ))

        #对CHI字典进行降序排序，得到的结果是元组列表
        IG_sort = sorted(IG_dict.iteritems(), key=operator.itemgetter(1), reverse=True)
        IG_list.append(IG_sort)
        
    return IG_list    

#由CHI值得排序选出特征词
def Get_character_list(top_n, data_dict):
    IG_list = CalInfoGain(data_dict)
    character_list = list()
    
    for IG_dict in IG_list:
        count = 0
        for item in IG_dict:
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
    
