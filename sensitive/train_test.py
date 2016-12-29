# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 15:34:09 2016

@author: ailab
"""
#该程序用于进行实验模型的训练与测试

import numpy as np
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import metrics

def label_save( test_file, save_f ):
    f = open(test_file, 'r')
    for line in f:
        save_f.write(line)
    f.close()

def print_result(predict_label, total_label, f):
    label_array = np.loadtxt(total_label)
    predict_array = np.loadtxt(predict_label)
    acc = metrics.accuracy_score(label_array, predict_array)
    
    f.write('Accuracy: ' + str(acc * 100.000) + '%' + '\n')       
    print('Accuracy: ', str(acc * 100.000) + '%')
    
    f.write(str(metrics.classification_report(label_array, predict_array)))
    print metrics.classification_report(label_array, predict_array) 
    

def train(data_file, lable_file, train_model='D_tree'):
    ''' 训练模型 '''
    
    # choose type of classifier
    if train_model == 'svm-rbf':
        clf = SVC(C=50.0, gamma=0.01, kernel='rbf')
    elif train_model == 'svm-linear':
        clf = SVC(C=50.0, gamma=0.01, kernel='linear')
    elif train_model == 'D_tree':
        clf = tree.DecisionTreeClassifier()
    elif train_model == 'NB':
        clf = MultinomialNB()
    elif train_model == 'LR':
        clf = LogisticRegression()

    data_array = np.loadtxt(data_file)
    label_array = np.loadtxt(lable_file)

    try:
        assert(data_array.shape[0] == label_array.shape[0])
    except:
        print data_array.shape[0]
        print label_array.shape[0]

    # 训练模型
    clf.fit(data_array, label_array)

    return clf

def test(model, data_file, lable_file, predict_file, predict_probab_file):
    # load data & label
    data_array = np.loadtxt(data_file)
    label_array = np.loadtxt(lable_file)

    assert(data_array.shape[0] == label_array.shape[0])

    lable = model.predict(data_array)
    #probab = model.predict_proba(data_array)
        
    np.savetxt(predict_file, lable)
    #np.savetxt(predict_probab_file, probab)
        
    acc = metrics.accuracy_score(label_array, lable)
        
    print('Accuracy: ', str(acc * 100.000) + '%')
    
    print metrics.classification_report(label_array, lable)
    
    return acc



    
    