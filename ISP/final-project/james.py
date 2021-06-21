# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 16:41:29 2021

@author: PC
"""
'''
import pandas as pd 


#url="https://github.com/purang2/deepLearning-practice/blob/main/ISP/final-project/data/features_30_sec.csv"
url=""

#raw.githubusercontent.com
c=pd.read_csv(url)

'''

import pandas as pd

url='https://raw.githubusercontent.com/purang2/deepLearning-practice/main/ISP/final-project/data/features_30_sec.csv'
#s=requests.get(url).content
c=pd.read_csv(url)

data = pd.read_csv('data/features_30_sec.csv')