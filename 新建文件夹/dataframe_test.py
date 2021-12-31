#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   dataframe_test.py
@Time    :   2021/12/31 09:36:18
@Author  :   Zhang Mingming 
@Version :   1.0
'''
import pandas as pd
import numpy as np
from pandas.core.indexes.base import Index
""" 1.创建dataframe """

# 使用numpy创建
# numpy.random.randn(m,n)：是从标准正态分布中返回m行n列个样本中；
# numpy.random.rand(m,n)：是从[0,1)中随机返回m行n列个样本。
df = pd.DataFrame(np.random.randn(100, 3), columns=[
    'first', 'second', 'third'])

# 使用dict创建
df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20180901'),
                    'C': pd.Series(1, index=range(4), dtype='float'),
                    'D': np.array([3]*4, dtype='int'),
                    'key': pd.Categorical(['test', 'train', 'test2', 'train2']),
                    'E': 'foo'
                    }
                   )

# 使用list创建
df3 = pd.DataFrame([['2', '1.2', '4.2','test'], ['0', '10', '0.3','train'], [
                   '1', '5', '0','test2'], [
                   '4', '2', '1','train2']], columns=['one', 'two', 'three','key'])

# 创建时间序列,freq参考 https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
TimeSeries_Day = pd.date_range('2018-09-01', periods=100)
TimeSeries_Month = pd.date_range('2018-09-01', periods=12, freq='M')
df.set_index(TimeSeries_Day, inplace=True)  # inplace=True改变原df,False不改变

""" 2.拆分整合数据 """
""" concat """
piece1 = df.head(10)  # 0到9号数据
piece2 = df[10:20]  # 10到19号数据
piece3 = df.iloc[20:30]  # 20到29号数据
piece4 = df.iloc[30:40, :]  # 30到39号数据
piece5 = df.loc['20181011':'2018-11-29']  # 从1011到1129的数据
piece6 = df.tail(10)  # 最后10个数据

df_pieces = pd.concat([piece1, piece2, piece3, piece4,
                      piece5, piece6], axis=0)  # concat拼接
df.equals(df_pieces)  # Out:True

""" merge """
pd.merge(df2,df3,on='key') #merge左右拼接

""" append """
day_1 = df.iloc[0]
df.append(day_1,ignore_index=False) #ignore_index若为Ture则插入数据后索引将更新，否则保持原有索引值

""" 3.导出导入数据 """
#导出为Csv文件，名称及位置（默认和notebook文件同一目录下，且不导出索引，index默认为True）
df.to_csv('foo.csv',index=False) 
fileDf=pd.read_csv('foo.csv')
""" 查看数据 """
fileDf.info() #查看数据类型，以及数据缺失情况
fileDf.describe() #查看数据描述统计性信息，数据大概分布情况
fileDf.shape #数据的维度
fileDf.dtypes #各字段数据的类型

""" 4.数据筛选 """
#数据转置
fileDf.T
#按指定属性值排序
fileDf.sort_values('first',ascending=False) #按照‘secondfirst’降序排列数据
#查看某数据数值的分布
fileDf['second'].value_counts()  #查看各项数量
fileDf['second'].value_counts(normalize=True)  #查看各项占比
""" 5.数据的切片 """
fileDf.loc[0:3,['first','second']] #获取索引值为(0:3)的中'first','second'的数据
fileDf[fileDf['first']>0] #利用boolean值提取“first”大于0的数据
List1 = fileDf[fileDf['first']>0]['first'].tolist()
fileDf[~fileDf['first'].isin(List1)&(fileDf['second']<0)] #提取“first”<0，且“second”<0的所有数据,~表示isin()方法的逆函数
""" 6.数据操作Operations """
#Apply:（用于dataframe，对row或column进行操作）类似于map
df.apply(lambda x:x.mean()) #默认对每列做平均
df.apply(lambda x:x.mean(),axis=1) #取每行平均

#stats
fileDf.mean() #求均值，默认对列
fileDf.mean(1) #求每行平均,生成的Series没有原本的index了

#groupby:按照某个index分组，后面跟的函数是聚合方式,可以输入多个column在一个list中
df['NewIndex'] = np.random.randint(1,10,100) #最小1，最大10，size 100的随机整数
df.groupby(['NewIndex']).sum()
df.groupby(['NewIndex']).sum()
