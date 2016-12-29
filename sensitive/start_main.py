# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 00:47:04 2016

@author: dell
"""

from sensitive import new_main
    
root_file = 'D:/graduate_research/new_2'

for i in range(5200, 7001):
    if i % 100 == 0:
        new_main.main(root_file, i, 5)
        new_main.write_result(root_file, i, 5)
        
        new_main.main(root_file, i, 10)
        new_main.write_result(root_file, i, 10)