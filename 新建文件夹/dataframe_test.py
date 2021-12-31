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
                    'E': pd.Categorical(['test', 'train', 'test', 'train']),
                    'F': 'foo'
                    }
                   )

# 使用list创建
df3 = pd.DataFrame([['2', '1.2', '4.2'], ['0', '10', '0.3'], [
                   '1', '5', '0']], columns=['one', 'two', 'three'])

# 创建时间序列,freq参考 https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
TimeSeries_Day = pd.date_range('2018-09-01', periods=100)
TimeSeries_Month = pd.date_range('2018-09-01', periods=12, freq='M')
df.set_index(TimeSeries_Day, inplace=True)  # inplace=True改变原df,False不改变
