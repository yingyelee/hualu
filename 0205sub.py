#!/usr/bin/env python
# coding: utf-8

# In[3]:


from pandas.io.json import json_normalize
import pandas as pd
import json
import numpy as np
import os
from pandas import DataFrame as DF
import scipy.spatial.distance as dist
import catboost as cbt
import json
from sklearn.metrics import f1_score
import time
import gc
import math
from tqdm import tqdm
from scipy import stats
from sklearn.cluster import KMeans
from six.moves import reduce
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
data_str = open(r"phase2_train.json").read()
df_train = pd.read_json(data_str,orient = 'records')
data_str_test = open(r"phase2_test.json").read()
df_test = pd.read_json(data_str_test,orient = 'records')
df_train_data = df_train.append(df_test)


# In[4]:


df_carroad=pd.read_excel(r'data.xlsx',sheet_name='车辆违法违规信息（道路交通安全，来源抄告）')
df_carill=pd.read_excel(r'data.xlsx',sheet_name='车辆违法违规信息（交通运输违法，来源抄告）')
df_caros=pd.read_excel(r'data.xlsx',sheet_name='动态监控报警信息（车辆，超速行驶）')

df_cartire=pd.read_excel(r'data.xlsx',sheet_name='动态监控报警信息（车辆，疲劳驾驶）')

df_carol=pd.read_excel(r'data.xlsx',sheet_name='动态监控上线率（企业，%）')
df_carannu=pd.read_excel(r'data.xlsx',sheet_name='运政车辆年审记录信息')

df_cotype=pd.read_excel(r'data.xlsx',sheet_name='运政业户信息')
df_cartype=pd.read_excel(r'data.xlsx',sheet_name='运政车辆信息')
df_coexam=pd.read_excel(r'data.xlsx',sheet_name='运政质量信誉考核记录')


# In[5]:


df_cartype_group=df_cartype.groupby(["业户ID","行业类别"])["车辆牌照号"].count()
df_cartype_group=df_cartype_group.unstack('行业类别')
df_cartype_group=df_cartype_group.reset_index()
df_cartype_group.columns=["业户ID","道路危险货物运输", "道路旅客运输", "道路货物运输"] 
df_cartype_group = df_cartype_group.fillna(0)


# In[6]:


df_carroad['Month'] = df_carroad['违规时间'].dt.month
df_caros = df_caros.loc[df_caros['持续点数']>0]


# In[7]:


df_carroad['month'] = df_carroad['违规时间'].dt.month
df_carroad = df_carroad.loc[df_carroad['month']<7]
df_carroad_group=df_carroad.groupby(["单位名称"])["车牌号"].count()
df_carroad_group = df_carroad_group.reset_index()
df_carroad_group.columns=["单位名称","道路违规数"] 


# In[8]:


df_carill['month'] = df_carill['违规时间'].dt.month
df_carill = df_carill.loc[df_carill['month']<7]
df_carill_group=df_carill.groupby(["单位名称"])["车牌号"].count()
df_carill_group = df_carill_group.reset_index()
df_carill_group.columns=["单位名称","交通违规数"] 


# In[9]:



df_caros['month'] = df_caros['开始时间'].str[6:7]
df_caros = df_caros.loc[df_caros['持续点数']>0]
df_caros = df_caros.loc[df_caros['最高时速(Km/h)']>60]
df_caros['level'] = np.where(df_caros['最高时速(Km/h)'] < 80, '超速低',
                      np.where((df_caros['最高时速(Km/h)'] >= 80) & (df_caros['最高时速(Km/h)'] < 100), '超速中',
                               np.where((df_caros['最高时速(Km/h)'] >= 100) & (df_caros['最高时速(Km/h)'] < 120), '超速高',
                                        np.where(df_caros['最高时速(Km/h)'] >= 120 , '超速超高','na'))))
df_caros['level_sc'] = np.where(df_caros['最高时速(Km/h)'] < 80,1,
                      np.where((df_caros['最高时速(Km/h)'] >= 80) & (df_caros['最高时速(Km/h)'] < 100),2,
                               np.where((df_caros['最高时速(Km/h)'] >= 100) & (df_caros['最高时速(Km/h)'] < 120),3,
                                        np.where(df_caros['最高时速(Km/h)'] >= 120 ,5,'na'))))
df_caros['level_sc'] = df_caros['level_sc'].astype(float)
df_caros_group=pd.merge(df_caros,df_cartype,left_on="车牌号码",right_on="车辆牌照号",how="left")
df_caros_group = df_caros_group[df_caros_group['业户ID'].notnull()]
df_caros_group_1=df_caros_group.groupby(["业户ID"])["车牌号码"].count()
df_caros_group_2=df_caros_group.groupby(["业户ID"])["车牌号码"].nunique()
df_caros_group_3=df_caros_group.groupby(["业户ID"])["持续点数"].sum()
df_caros_group_4=df_caros_group.groupby(["业户ID"])["level_sc"].sum()
df_caros_group_1 = df_caros_group_1.reset_index()
df_caros_group_2 = df_caros_group_2.reset_index()
df_caros_group_3 = df_caros_group_3.reset_index()
df_caros_group_4 = df_caros_group_4.reset_index()
df_caros_group_final=pd.merge(df_caros_group_1,df_caros_group_2,on="业户ID")
df_caros_group_final=pd.merge(df_caros_group_final,df_caros_group_3,on="业户ID")
df_caros_group_final=pd.merge(df_caros_group_final,df_caros_group_4,on="业户ID")
df_caros_group_final.columns=["业户ID","超速车牌号码总数", "超速车牌号码去重", "超速持续点数", "超速分数"] 


# In[10]:


pivot_os_co = df_caros_group.pivot_table(values='车牌号码', index='业户ID', columns=['level','month'], aggfunc='count')
pivot_os_co.columns = ['合计_'.join(col) for col in pivot_os_co.columns]
pivot_os_co= pivot_os_co.reset_index()
pivot_os_co=pivot_os_co.fillna(0)
pivot_os_co['new_os_co_1'] = 0.00
pivot_os_co['new_os_co_2'] = 0.00
pivot_os_co['new_os_co_3'] = 0.00
pivot_os_co['new_os_co_4'] = 0.00
for i, row in pivot_os_co.iterrows():
    if row.iloc[1] + row.iloc[2] + row.iloc[3] == 0:
        pivot_os_co.at[i, 'new_os_co_1'] = (row.iloc[4] + row.iloc[5] + row.iloc[6]) / 3
    else:
        pivot_os_co.at[i, 'new_os_co_1'] = (row.iloc[4] + row.iloc[5] + row.iloc[6]) / (row.iloc[1] + row.iloc[2] + row.iloc[3])
    if row.iloc[7] + row.iloc[8] + row.iloc[9] == 0:
        pivot_os_co.at[i, 'new_os_co_2'] = (row.iloc[10] + row.iloc[11] + row.iloc[12]) / 3
    else:
        pivot_os_co.at[i, 'new_os_co_2'] = (row.iloc[10] + row.iloc[11] + row.iloc[12]) / (row.iloc[7] + row.iloc[8] + row.iloc[9])
    if row.iloc[13] + row.iloc[14] + row.iloc[15] == 0:
        pivot_os_co.at[i, 'new_os_co_3'] = (row.iloc[16] + row.iloc[17] + row.iloc[18]) / 3
    else:
        pivot_os_co.at[i, 'new_os_co_3'] = (row.iloc[16] + row.iloc[17] + row.iloc[18]) / (row.iloc[13] + row.iloc[14] + row.iloc[15])

    if row.iloc[19] + row.iloc[20] + row.iloc[21] == 0:
        pivot_os_co.at[i, 'new_os_co_4'] = (row.iloc[22] + row.iloc[23] + row.iloc[24]) / 3
    else:
        pivot_os_co.at[i, 'new_os_co_4'] = (row.iloc[22] + row.iloc[23] + row.iloc[24]) / (row.iloc[19] + row.iloc[20] + row.iloc[21])
pivot_os_sum = df_caros_group.pivot_table(values='持续点数', index='业户ID', columns=['level','month'], aggfunc='sum')
pivot_os_sum.columns = ['求和_'.join(col) for col in pivot_os_sum.columns]
pivot_os_sum= pivot_os_sum.reset_index()
pivot_os_sum=pivot_os_sum.fillna(0)
pivot_os_sum['new_os_sum_1'] = 0.00
pivot_os_sum['new_os_sum_2'] = 0.00
pivot_os_sum['new_os_sum_3'] = 0.00
pivot_os_sum['new_os_sum_4'] = 0.00
for i, row in pivot_os_sum.iterrows():
    if row.iloc[1] + row.iloc[2] + row.iloc[3] == 0:
        pivot_os_sum.at[i, 'new_os_sum_1'] = (row.iloc[4] + row.iloc[5] + row.iloc[6]) / 3
    else:
        pivot_os_sum.at[i, 'new_os_sum_1'] = (row.iloc[4] + row.iloc[5] + row.iloc[6]) / (row.iloc[1] + row.iloc[2] + row.iloc[3])
    if row.iloc[7] + row.iloc[8] + row.iloc[9] == 0:
        pivot_os_sum.at[i, 'new_os_sum_2'] = (row.iloc[10] + row.iloc[11] + row.iloc[12]) / 3
    else:
        pivot_os_sum.at[i, 'new_os_sum_2'] = (row.iloc[10] + row.iloc[11] + row.iloc[12]) / (row.iloc[7] + row.iloc[8] + row.iloc[9])
    if row.iloc[13] + row.iloc[14] + row.iloc[15] == 0:
        pivot_os_sum.at[i, 'new_os_sum_3'] = (row.iloc[16] + row.iloc[17] + row.iloc[18]) / 3
    else:
        pivot_os_sum.at[i, 'new_os_sum_3'] = (row.iloc[16] + row.iloc[17] + row.iloc[18]) / (row.iloc[13] + row.iloc[14] + row.iloc[15])

    if row.iloc[19] + row.iloc[20] + row.iloc[21] == 0:
        pivot_os_sum.at[i, 'new_os_sum_4'] = (row.iloc[22] + row.iloc[23] + row.iloc[24]) / 3
    else:
        pivot_os_sum.at[i, 'new_os_sum_4'] = (row.iloc[22] + row.iloc[23] + row.iloc[24]) / (row.iloc[19] + row.iloc[20] + row.iloc[21])


# In[11]:


caros_co_table = df_caros_group.pivot_table(values='车牌号码', index='业户ID', columns='month', aggfunc='count')
caros_co_table=caros_co_table.reset_index()
caros_co_table.columns=["业户ID","1月超速车牌", "2月超速车牌", "3月超速车牌", "4月超速车牌", "5月超速车牌", "6月超速车牌"] 
caros_num_table = df_caros_group.pivot_table(values='持续点数', index='业户ID', columns='month', aggfunc='sum')
caros_num_table=caros_num_table.reset_index()
caros_num_table.columns=["业户ID","1月超速点数", "2月超速点数", "3月超速点数", "4月超速点数", "5月超速点数", "6月超速点数"] 


# In[12]:


df_cartire['month'] = df_cartire['开始时间'].str[6:7]
df_cartire = df_cartire.loc[df_cartire['持续点数']>0]
df_cartire = df_cartire.loc[df_cartire['最高时速(Km/h)']>60]
df_cartire['level'] = np.where(df_cartire['最高时速(Km/h)'] < 70, '疲劳低',
                      np.where((df_cartire['最高时速(Km/h)'] >= 70) & (df_cartire['最高时速(Km/h)'] < 80), '疲劳中',
                               np.where((df_cartire['最高时速(Km/h)'] >= 80) & (df_cartire['最高时速(Km/h)'] < 95), '疲劳高',
                                        np.where(df_cartire['最高时速(Km/h)'] >= 95 , '疲劳超高','na'))))
df_cartire['level_sc'] = np.where(df_cartire['最高时速(Km/h)'] < 65,1,
                      np.where((df_cartire['最高时速(Km/h)'] >= 65) & (df_cartire['最高时速(Km/h)'] < 80), 2,
                               np.where((df_cartire['最高时速(Km/h)'] >= 80) & (df_cartire['最高时速(Km/h)'] < 95), 3,
                                        np.where(df_cartire['最高时速(Km/h)'] >= 95 ,5,'na'))))
df_cartire['level_sc'] = df_cartire['level_sc'].astype(float)
df_cartire_group=pd.merge(df_cartire,df_cartype,left_on="车牌号码",right_on="车辆牌照号",how="left")
df_cartire_group = df_cartire_group[df_cartire_group['业户ID'].notnull()]
df_cartire_group_1=df_cartire_group.groupby(["业户ID"])["车牌号码"].count()
df_cartire_group_2=df_cartire_group.groupby(["业户ID"])["车牌号码"].nunique()
df_cartire_group_3=df_cartire_group.groupby(["业户ID"])["持续点数"].sum()
df_cartire_group_4=df_cartire_group.groupby(["业户ID"])["level_sc"].sum()
df_cartire_group_1 = df_cartire_group_1.reset_index()
df_cartire_group_2 = df_cartire_group_2.reset_index()
df_cartire_group_3 = df_cartire_group_3.reset_index()
df_cartire_group_4 = df_cartire_group_4.reset_index()
df_cartire_group_final=pd.merge(df_cartire_group_1,df_cartire_group_2,on="业户ID")
df_cartire_group_final=pd.merge(df_cartire_group_final,df_cartire_group_3,on="业户ID")
df_cartire_group_final=pd.merge(df_cartire_group_final,df_cartire_group_4,on="业户ID")
df_cartire_group_final.columns=["业户ID","疲劳车牌号码总数", "疲劳车牌号码去重", "疲劳持续点数", "疲劳分数"] 


# In[13]:


cartire_co_table = df_cartire_group.pivot_table(values='车牌号码', index='业户ID', columns='month', aggfunc='count')
cartire_co_table=cartire_co_table.reset_index()
cartire_co_table.columns=["业户ID","1月疲劳车牌", "2月疲劳车牌", "3月疲劳车牌", "4月疲劳车牌", "5月疲劳车牌", "6月疲劳车牌"] 
cartire_num_table = df_cartire_group.pivot_table(values='持续点数', index='业户ID', columns='month', aggfunc='sum')
cartire_num_table=cartire_num_table.reset_index()
cartire_num_table.columns=["业户ID","1月疲劳点数", "2月疲劳点数", "3月疲劳点数", "4月疲劳点数", "5月疲劳点数", "6月疲劳点数"] 


# In[14]:


pivot_tire_co = df_cartire_group.pivot_table(values='车牌号码', index='业户ID', columns=['level','month'], aggfunc='count')
pivot_tire_co.columns = ['合计_'.join(col) for col in pivot_tire_co.columns]
pivot_tire_co= pivot_tire_co.reset_index()
pivot_tire_co=pivot_tire_co.fillna(0)
pivot_tire_co['new_tire_co_1'] = 0
pivot_tire_co['new_tire_co_2'] = 0
pivot_tire_co['new_tire_co_3'] = 0
pivot_tire_co['new_tire_co_4'] = 0
for i, row in pivot_tire_co.iterrows():
    if row.iloc[1] + row.iloc[2] + row.iloc[3] == 0:
        pivot_tire_co.at[i, 'new_tire_co_1'] = (row.iloc[4] + row.iloc[5] + row.iloc[6]) / 3
    else:
        pivot_tire_co.at[i, 'new_tire_co_1'] = (row.iloc[4] + row.iloc[5] + row.iloc[6]) / (row.iloc[1] + row.iloc[2] + row.iloc[3])
    if row.iloc[7] + row.iloc[8] + row.iloc[9] == 0:
        pivot_tire_co.at[i, 'new_tire_co_2'] = (row.iloc[10] + row.iloc[11] + row.iloc[12]) / 3
    else:
        pivot_tire_co.at[i, 'new_tire_co_2'] = (row.iloc[10] + row.iloc[11] + row.iloc[12]) / (row.iloc[7] + row.iloc[8] + row.iloc[9])
    if row.iloc[13] + row.iloc[14] + row.iloc[15] == 0:
        pivot_tire_co.at[i, 'new_tire_co_3'] = (row.iloc[16] + row.iloc[17] + row.iloc[18]) / 3
    else:
        pivot_tire_co.at[i, 'new_tire_co_3'] = (row.iloc[16] + row.iloc[17] + row.iloc[18]) / (row.iloc[13] + row.iloc[14] + row.iloc[15])

    # Check conditions for fourth set of columns
    if row.iloc[19] + row.iloc[20] + row.iloc[21] == 0:
        pivot_tire_co.at[i, 'new_tire_co_4'] = (row.iloc[22] + row.iloc[23] + row.iloc[24]) / 3
    else:
        pivot_tire_co.at[i, 'new_tire_co_4'] = (row.iloc[22] + row.iloc[23] + row.iloc[24]) / (row.iloc[19] + row.iloc[20] + row.iloc[21])
pivot_tire_sum = df_cartire_group.pivot_table(values='持续点数', index='业户ID', columns=['level','month'], aggfunc='sum')
pivot_tire_sum.columns = ['求和_'.join(col) for col in pivot_tire_sum.columns]
pivot_tire_sum= pivot_tire_sum.reset_index()
pivot_tire_sum=pivot_tire_sum.fillna(0)
pivot_tire_sum['new_tire_sum_1'] = 0.00
pivot_tire_sum['new_tire_sum_2'] = 0.00
pivot_tire_sum['new_tire_sum_3'] = 0.00
pivot_tire_sum['new_tire_sum_4'] = 0.00
for i, row in pivot_tire_sum.iterrows():
    if row.iloc[1] + row.iloc[2] + row.iloc[3] == 0:
        pivot_tire_sum.at[i, 'new_tire_sum_1'] = (row.iloc[4] + row.iloc[5] + row.iloc[6]) / 3
    else:
        pivot_tire_sum.at[i, 'new_tire_sum_1'] = (row.iloc[4] + row.iloc[5] + row.iloc[6]) / (row.iloc[1] + row.iloc[2] + row.iloc[3])
    if row.iloc[7] + row.iloc[8] + row.iloc[9] == 0:
        pivot_tire_sum.at[i, 'new_tire_sum_2'] = (row.iloc[10] + row.iloc[11] + row.iloc[12]) / 3
    else:
        pivot_tire_sum.at[i, 'new_tire_sum_2'] = (row.iloc[10] + row.iloc[11] + row.iloc[12]) / (row.iloc[7] + row.iloc[8] + row.iloc[9])
    if row.iloc[13] + row.iloc[14] + row.iloc[15] == 0:
        pivot_tire_sum.at[i, 'new_tire_sum_3'] = (row.iloc[16] + row.iloc[17] + row.iloc[18]) / 3
    else:
        pivot_tire_sum.at[i, 'new_tire_sum_3'] = (row.iloc[16] + row.iloc[17] + row.iloc[18]) / (row.iloc[13] + row.iloc[14] + row.iloc[15])

    if row.iloc[19] + row.iloc[20] + row.iloc[21] == 0:
        pivot_tire_sum.at[i, 'new_tire_sum_4'] = (row.iloc[22] + row.iloc[23] + row.iloc[24]) / 3
    else:
        pivot_tire_sum.at[i, 'new_tire_sum_4'] = (row.iloc[22] + row.iloc[23] + row.iloc[24]) / (row.iloc[19] + row.iloc[20] + row.iloc[21])


# In[15]:


df_carannu_group=pd.merge(df_carannu,df_cartype,left_on="车辆牌照号",right_on="车辆牌照号",how="left")
df_carannu_group_1=df_carannu_group.groupby(["业户ID"])["车辆牌照号"].count()
df_carannu_group_2=df_carannu_group.groupby(["业户ID"])["车辆牌照号"].nunique()
df_carannu_group_1 = df_carannu_group_1.reset_index()
df_carannu_group_2 = df_carannu_group_2.reset_index()
df_carannu_group_final=pd.merge(df_carannu_group_1,df_carannu_group_2,on="业户ID")
df_carannu_group_final.columns=["业户ID","年审总数", "年审去重计数"] 


# In[16]:


df_coexam=df_coexam.replace(to_replace ="优良(AAA)", value =100)
df_coexam=df_coexam.replace(to_replace ="合格(AA)", value =80)
df_coexam=df_coexam.replace(to_replace ="基本合格(A)", value =60)
df_coexam=df_coexam.replace(to_replace ="不合格(B)", value =40)
df_coexam_group=df_coexam.groupby(["业户ID"])["质量信誉考核结果"].mean()


# In[17]:


data_str = open(r"phase1_train.json").read()
df_car_train = pd.read_json(data_str,orient = 'records')
df_car_test = pd.read_csv('sub0114.csv')
df_carall_data = df_car_train.append(df_car_test)
df_carall_data=pd.merge(df_cartype,df_carall_data,left_on="车辆牌照号",right_on="car_id",how="left")



# In[18]:


data_grouped = df_carall_data.groupby('业户ID')
data_agg = data_grouped.agg({'score': ['mean', lambda x: (x == 100).sum(), lambda x: x.tail(5).mean(), lambda x: (x < 85).sum(), 
                                      lambda x: sum(x<85)/len(x), 'std']})
data_agg.columns = ['score_mean', 'score_100_count', 'score_last5_mean', 'score_lt85_count', 'score_lt85_ratio','score_std']
data_agg = data_agg.reset_index()
data_agg =data_agg.fillna(0)


# In[19]:


res=[df_cotype,df_cartype_group,df_caros_group_final,df_cartire_group_final,df_carannu_group_final,
     df_coexam_group,cartire_co_table,cartire_num_table,caros_co_table,caros_num_table
    ,pivot_os_sum,pivot_os_co,pivot_tire_sum,pivot_tire_co,data_agg]
from functools import reduce 
data= reduce(lambda left,right: pd.merge(left,right,on=['业户ID'],how='left'), res)
data=pd.merge(data,df_carroad_group,left_on="企业名称",right_on="单位名称",how="left")
data=pd.merge(data,df_carill_group,left_on="企业名称",right_on="单位名称",how="left")
data=pd.merge(data,df_carol,left_on="企业名称",right_on="企业名称",how="left")
data=pd.merge(data,df_train,left_on="业户ID",right_on="company_id",how="left")


# In[20]:


data=data.drop(columns=['经营许可证号', '单位名称_x', '单位名称_y', 'company_id','企业名称','道路危险货物运输','道路旅客运输','行业类别',
                       '道路旅客运输','道路货物运输'])


# In[21]:


data['疲劳车牌环比']=(data['1月疲劳车牌']+data['2月疲劳车牌']+data['3月疲劳车牌'])/(data['4月疲劳车牌']+data['5月疲劳车牌']+data['6月疲劳车牌']+0.01)


# In[22]:


data['疲劳车牌环比'] = (data['1月疲劳车牌'] + data['2月疲劳车牌'] + data['3月疲劳车牌']
                 ) / (data['4月疲劳车牌'] + data['5月疲劳车牌'] + data['6月疲劳车牌'] + 0.01)


# In[23]:


cols_to_fill = [col for col in data.columns if col != 'score']
data[cols_to_fill] = data[cols_to_fill].fillna(0)

data['疲劳车牌环比'] = (data['1月疲劳车牌'] + data['2月疲劳车牌'] + data['3月疲劳车牌']
                 ) / (data['4月疲劳车牌'] + data['5月疲劳车牌'] + data['6月疲劳车牌'] + 0.01)
data['疲劳点数环比']=(data['1月疲劳点数']+data['2月疲劳点数']+data['3月疲劳点数']
               )/(data['4月疲劳点数']+data['5月疲劳点数']+data['6月疲劳点数']+0.01)
data['超速车牌环比']=(data['1月超速车牌']+data['2月超速车牌']+data['3月超速车牌']
               )/(data['4月超速车牌']+data['5月超速车牌']+data['6月超速车牌']+0.01)
data['超速点数环比']=(data['1月超速点数']+data['2月超速点数']+data['3月超速点数']
               )/(data['4月超速点数']+data['5月超速点数']+data['6月超速点数']+0.01)
data['疲劳车牌方差']=data[['1月疲劳车牌', '2月疲劳车牌','3月疲劳车牌','4月疲劳车牌','5月疲劳车牌','6月疲劳车牌']].var(axis=1)
data['疲劳车牌标准差']=data['疲劳车牌方差']/(data['年审去重计数']*data['年审去重计数'])
data['疲劳点数方差']=data[['1月疲劳点数', '2月疲劳点数','3月疲劳点数','4月疲劳点数','5月疲劳点数','6月疲劳点数']].var(axis=1)
data['疲劳点数标准差']=data['疲劳点数方差']/(data['年审去重计数']*data['年审去重计数'])
data['超速车牌方差']=data[['1月超速车牌', '2月超速车牌','3月超速车牌','4月超速车牌','5月超速车牌','6月超速车牌']].var(axis=1)
data['超速车牌标准差']=data['超速车牌方差']/(data['年审去重计数']*data['年审去重计数'])
data['超速点数方差']=data[['1月超速点数', '2月超速点数','3月超速点数','4月超速点数','5月超速点数','6月超速点数']].var(axis=1)
data['超速点数标准差']=data['超速点数方差']/(data['年审去重计数']*data['年审去重计数'])
data['超速总数平均']=data['超速车牌号码总数']/data['年审去重计数']
data['超速占比']=data['超速车牌号码去重']/data['年审去重计数']
data['超速点数平均']=data['超速持续点数']/data['年审去重计数']
data['疲劳总数平均']=data['疲劳车牌号码总数']/data['年审去重计数']
data['疲劳占比']=data['疲劳车牌号码去重']/data['年审去重计数']
data['疲劳点数平均']=data['疲劳持续点数']/data['年审去重计数']
data['年审平均']=data['年审总数']/data['年审去重计数']
data['年审平均']=data['年审总数']/data['年审去重计数']
data['点数合计']=data['超速持续点数']+data['疲劳持续点数']
data['上线率合计']=data['1月上线率']+data['2月上线率']+data['3月上线率']+data['4月上线率']+data['5月上线率']+data['6月上线率']
data['上线率平均']=data[['1月上线率', '2月上线率','3月上线率','4月上线率','5月上线率','6月上线率']].var(axis=1)


# In[24]:


train = data[data['score'].notnull()]
X_train = train.drop(columns=['score'])
y_train = train['score']

test = data[data['score'].isnull()]
X_test = test.drop(columns=['score'])


# In[25]:


train_w = train.copy()
train_w=train_w.drop(columns=['业户ID','score'])


# In[26]:



target = train['score']
target_log = np.log1p(target)
def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
def cv_rmse(model):
    rmse = np.sqrt(-cross_val_score(model, train_w, target_log, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)


# In[27]:


kf = KFold(n_splits=5, random_state=42, shuffle=True)

cv_scores = []
cv_std = []


# In[28]:


from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from mlxtend.regressor import StackingRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error
catb = CatBoostRegressor()
score_catb = cv_rmse(catb)
cv_scores.append(score_catb.mean())
cv_std.append(score_catb.std())


# In[29]:


X_train,X_val,y_train,y_val = train_test_split(train_w,target_log,test_size = 0.1,random_state=42)
# Cat Boost Regressor
cat = CatBoostRegressor()
cat_model = cat.fit(X_train,y_train,
                     eval_set = (X_val,y_val),
                     plot=True,
                     verbose = 0)


# In[30]:


cat_pred = cat_model.predict(X_val)
cat_score = rmse(y_val, cat_pred)


# In[31]:


from catboost import Pool
train_pool = Pool(X_train)
val_pool = Pool(X_val)


# In[32]:


grid = {'iterations': [1000,6000],
        'learning_rate': [0.05, 0.005, 0.0005],
        'depth': [4, 6, 10],
        'l2_leaf_reg': [1, 3, 5, 9]}

final_model = CatBoostRegressor()
randomized_search_result = final_model.randomized_search(grid,
                                                   X = X_train,
                                                   y= y_train,
                                                   verbose = False,
                                                   plot=True)


# In[33]:


params = {'iterations': 1000,
          'learning_rate': 0.005,
          'depth': 4,
          'l2_leaf_reg': 1,
          'eval_metric':'RMSE',
          'early_stopping_rounds': 100,
          'verbose': 100,
          'random_seed': 42
         }
         
cat_f = CatBoostRegressor(**params)
cat_model_f = cat_f.fit(X_train,y_train,
                     eval_set = (X_val,y_val),
                     plot=True,
                     verbose = False)

catf_pred = cat_model_f.predict(X_val)
catf_score = rmse(y_val, catf_pred)


# In[34]:


catf_score


# In[35]:


test_w=test.drop(['业户ID'],axis = 1)
test_id = test['业户ID']


# In[37]:


test_pred = cat_f.predict(test_w)
submission = pd.DataFrame(test_id, columns = ['业户ID'])
test_pred = np.expm1(test_pred)
submission['score'] = test_pred 
submission.columns = ['company_id','score']
submission.loc[submission['score'] > 97, 'score'] =100


# In[38]:


submission.to_json('result.json', orient = 'records')

