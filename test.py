# -*- coding: utf-8 -*-
'''
@File    :   numpy_test.py
@Time    :   2021/12/24 14:30:55
@Author  :   Yishu Zhou 
'''

import numpy as np
import pandas as pd
import os
import sys
# sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
# from sklearn import datasets


# Start typing your code from here

if __name__ == '__main__':
    from plotly_test import DrawFigure
    # iris = datasets.load_iris()
    # # print(iris.keys())
    # df = pd.DataFrame(iris['data'])
     # test1:测试输入一些随机的numpy.ndarray
    file_path = './Output/'
    file_name = 'TEST1'
    # 创建一个500条数据，5个columns的标准dataframe作为输入
    df1 = pd.DataFrame(np.random.randn(500, 6), columns=[
                       'one', 'two', 'three', 'four', 'five', 'six'])
    # 创建实例，定义df，输出路径，文件名
    test1 = DrawFigure(df1, file_path, file_name='TEST1')
    # 普通作图，3D作图，作子图，作某列的图
    t1 = test1.plot(xaxis_name='INDEX')
    t2 = test1.plot_3dfigures(xaxis_name='INDEX', file_name='TEST2')
    t3 = test1.plot_subplots(
        xaxis_name='Index', yaxis_name='values', file_name='TEST3')
    t4 = test1.plot_one_column(
        xaxis_name='INDEX', file_name='TEST4', trace_type=4, layout_num=2)
    test1.plot(type='Histogram', file_name='TEST4_2')
    # 画图时指定不同的layout，x轴名称，图片title，trace的名字，输出文件名，是否保存，作图具体类型，等等
    test1.plot_3dfigures(xaxis_name='INDEX', file_name='TEST5', layout_num=2)
    # 在现有图上加一条trace
    test1.Add_Trace_And_plot(t1, file_name='TEST6')
    test1.Add_Trace_And_plot(t1, x=pd.Series([1 for i in range(
        500)]), y=np.random.randn(500), file_name='TEST7', trace_name='NEWTRACE')
    # 当添加3d Trace，只传入参数x时，y和z按照实例df的第二第三列算
    test1.Add_Trace_3d_And_plot(t2, x=pd.Series(
        [1 for i in range(500)]), file_name='TEST8', trace_name='NEWTRACE')
    # 子图添加，每次添加后创建了一个新的Figure对象；其他操作不改变原来的Figure对象
    t3_2 = test1.Add_Trace_To_Subplots_And_plot(t3, x=np.random.randn(
        500), y=np.random.randn(500), file_name='TEST8_2', trace_name='23')
    t3_3 = test1.Add_Trace_To_Subplots_And_plot(t3_2, x=np.random.randn(
        500), y=np.random.randn(500), mode='lines', file_name='TEST8_3')
    t3_4 = test1.Add_Trace_To_Subplots_And_plot(t3_3, x=np.random.randn(
        500), y=np.random.randn(500), mode='lines', file_name='TEST8_4')
    t3_5 = test1.Add_Trace_To_Subplots_And_plot(t3_4, x=np.random.randn(
        500), y=np.random.randn(500), mode='lines', file_name='TEST8_5')
    t3_6 = test1.Add_Trace_To_Subplots_And_plot(t3_5, x=np.random.randn(
        500), y=np.random.randn(500), mode='lines+markers', file_name='TEST8_6')
    # 更新某张Figure的layout
    test1.Update_Layout_And_plot(t2, layout_num=2, file_name='TEST9')
    # 更改子图集的Layout,第二个参数指定第几个子图
    test1.Update_Layout_for_subplots_And_plot(
        t3, 3, xaxis_title='yes', yaxis_title='no', file_name='TEST10')
    # 3d图2d图混合子图
    t11 = test1.plot_3dfigures(x=np.random.randn(500), y=np.random.randn(
        500), z=np.random.randn(500), file_name='TEST11')
    t12 = test1.Add_Trace_To_Subplots_And_plot(
        t11, z=np.random.randn(500), type='Scatter3d', file_name='TEST11_2')
    t13 = test1.Add_Trace_To_Subplots_And_plot(
        t12, z=np.random.randn(500), type='Mesh3d', file_name='TEST11_3')
    t14 = test1.Add_Trace_To_Subplots_And_plot(
        t13, z=np.random.randn(500), type='Scatter3d', file_name='TEST11_4')
    t15 = test1.Add_Trace_To_Subplots_And_plot(
        t14, z=np.random.randn(500), type='Scatter3d', file_name='TEST11_5')
    t16 = test1.Add_Trace_To_Subplots_And_plot(
        t15, z=np.random.randn(500), type='Scatter3d', file_name='TEST11_6')
    t17 = test1.Add_Trace_To_Subplots_And_plot(t16, x=np.random.randn(
        500), y=np.random.randn(500), file_name='TEST11_7')
    # 饼图的参数特殊，x是label(一般不为数值)，y是values
    t18 = test1.Add_Trace_To_Subplots_And_plot(t17, x=['a', 'b', 'c', 'd', 'e'], y=[
                                               1, 2, 3, 5, 4], type='Pie', file_name='TEST11_8')
    test1.Update_Layout_And_plot(t18, layout_num=2, file_name='TEST11_9')
    s1 = np.random.normal(100, 10, 500)
    s2 = (s1-np.array([100 for i in range(500)]))/10
    # 直方图要输入x，只输入y的情况使用的是实例里的默认数据
    hist1 = test1.plot(x=s1, type='Histogram',
                       file_name='Hist1', mode='density')
    test1.Add_Trace_To_Subplots_And_plot(
        hist1, y=s1, type='Histogram', file_name='Hist2', mode='density')
    test1.Add_Trace_3d_And_plot(t2,type='Mesh3d',file_name='TEST12')


