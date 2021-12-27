# -*- coding: utf-8 -*-
'''
@File    :   plotly_test.py
@Time    :   2021/12/22 12:44:37
@Author  :   Yishu Zhou 
'''
# Start typing your code from here

from collections import OrderedDict
import os
import pandas as pd
import numpy as np

import plotly as py
import plotly.graph_objs as go

# 创建子图使用make_subplots
from plotly.subplots import make_subplots

import plotly.io as pio
pio.templates.default = 'plotly_dark'
""" 
Default template: 'plotly_dark'
Available templates:
    ['ggplot2', 'seaborn', 'simple_white', 'plotly',
        'plotly_white', 'plotly_dark', 'presentation', 'xgridoff',
        'ygridoff', 'gridon', 'none']
"""
# setting offilne
py.offline.init_notebook_mode(connected=True)
''' 
用numpy.ndarray a,b,c生成df: pd.DataFrame([a,b,c])
用pandas.Seires a,b,c生成df: pd.concat([a,b,c],axis=1) 不注明axis则生成一长条
生成长度为500数值为0的numpy.ndarray: np.array([0 for i in range(500)])
'''


def main():
    file_path = './Output/'
    file_name = 'TEST1'

    df1 = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [7, 7, 7, 7, 7], 'c': [
        8, 8, 8, 8, 8], 'd': [9, 9, 9, 9, 9]}, index=['a', 'b', 'c', 'd', 'e'])
    df2 = pd.DataFrame(np.random.randn(500))

    df3 = df1.copy()
    df3['4th'] = [98, 12, 33, 44, 22]
    df3['5th'] = [1231, 12, 33, 4, 1]
    df3['6th'] = [12, 3, 2, 1, 6]
    df3['7th'] = [98, 512, 33, 44, 22]
    df3['8th'] = [1231, 12, 233, 4, 1]
    # df3['9th'] = [12,3,2,1,63]

    df4 = pd.DataFrame({'x': np.random.randn(
        500), 'y': np.random.randn(500), 'z': np.random.randn(500)})

    s = np.linspace(0, 2 * np.pi, 240)  # 在指定范围内返回240个等间隔数据
    t = np.linspace(0, np.pi, 240)  # 在指定范围内返回240个等间隔数据
    tGrid, sGrid = np.meshgrid(s, t)  # 表示出s和t所有交叉的点

    r = 2 + np.sin(7 * sGrid + 5 * tGrid)  # r = 2 + sin(7s+5t)
    x = r * np.cos(sGrid) * np.sin(tGrid)  # x = r*cos(s)*sin(t)
    y = r * np.sin(sGrid) * np.sin(tGrid)  # y = r*sin(s)*sin(t)
    z = r * np.cos(tGrid)                  # z = r*cos(t)。x,y,z都是240*240矩阵

    test1 = DrawFigure(df3, file_path, file_name)
    t1 = test1.plot(title_name='suceess',
                    xaxis_name='Factor', yaxis_name='values')
    test1.Add_Trace_And_plot(t1, number=2, file_name='TEST1_2')
    t2 = test1.plot_one_column(
        mode='lines', file_name='TEST2', xaxis_name='Factor')
    t2_3 = test1.plot(x=np.random.randn(500), y=np.random.randn(
        500), mode='markers', file_name='TEST2_3', xaxis_name='rx', yaxis_name='ry')
    test1.Add_Trace_And_plot(t2, number=7, trace_type=2, file_name='TEST2_4')
    test1.Add_Trace_And_plot(t2, np.random.randn(
        500), np.random.randn(500), trace_type=2, file_name='TEST2_5')
    test1.Add_Trace_And_plot(
        t2, number=1, file_name='TEST2_2', trace_name='added trace not b')
    test1.plot_subplots(type='Bar', file_name='TEST3', title_name='test title')
    test1.plot_subplots(mode='lines+markers', file_name='TEST4')
    test1.plot_subplots(type='Pie', file_name='TEST5')
    test1.plot_subplots(mode='lines+markers', file_name='TEST6')
    test1.plot_subplots(type='Histogram', file_name='TEST7')

    test2 = DrawFigure(df4, file_path, 'TEST8')
    test2.plot_3dfigures(mode='text')
    test2.plot_3dfigures(mode="markers", file_name='TEST9')
    test2.plot_3dfigures(mode="lines", file_name='TEST10')
    test2.plot_3dfigures(x, y, z, type='Surface', file_name='TEST11')
    test2.plot_3dfigures(s, t, z, type='Surface', file_name='TEST12')
    test2.plot_3dfigures(type='Mesh3d', file_name='TEST13')

    r1 = GenerateNdimArray(N=5)
    # x,y为一维数组，z可以是多维的Array。生成的是一个曲面
    test2.plot_3dfigures(df4['x'], df4['y'], r1,
                         type='Surface', file_name='TEST14')

    # 规则同上，生成的图为多张叠加
    test2.plot_3dfigures(df4['x'], df4['y'], r1, type='Mesh3d',
                         title_name='suceess', file_name='TEST15')

    df5 = pd.DataFrame({'x': np.random.randn(500), 'y': np.random.randn(500)})
    for i in range(5):
        df5[str(i)] = np.random.uniform(i, i+1, 500)
    test3 = DrawFigure(df5, file_path, 'test16')
    t16 = test3.plot_3dfigures(
        z=df5.iloc[:, 2:4], type='Mesh3d', file_name='TEST16', trace_name='do some test')
    t16_v2 = test3.Add_Trace_3d_And_plot(
        t16, df4['x'], mode="lines", file_name='TEST16_2', trace_name='add Trace to 3D figures')
    test3.plot_3dfigures(z=df5.iloc[:, 2:4],
                         type='Surface', file_name='TEST17')
    test3.plot_3dfigures(z=df5.iloc[:, 2:4], mode='lines', file_name='TEST18',
                         xaxis_name='randomX', yaxis_name='randomY', zaxis_name='randomZ*2')
    t19_v1 = test3.plot_3dfigures(
        z=df5.iloc[:, 2:4], mode='markers', file_name='TEST19')
    t19_v2 = test3.Add_Trace_3d_And_plot(
        t19_v1, df4['x'], mode="lines", file_name='TEST20')
    # print(type(b))
    t19_v3 = test3.Update_Layout_And_plot(
        t19_v2, title_name='Update', layout_num='1', file_name='TEST21',)
    t19_v4 = test3.Update_Layout_And_plot(
        t19_v2, title_name='Update Successfully', layout_num='2', xaxis_title='abc', yaxis_title='ass', file_name='TEST21_2')
    t20_v1 = test3.plot_3dfigures(
        z=df5.iloc[:, 2:3], mode='text', file_name='TEST22')
    t20_v2 = test3.Add_Trace_3d_And_plot(t20_v1, df4['x'], file_name='TEST23')
    t20_v3 = test3.Add_Trace_3d_And_plot(
        t20_v2, df4['x'], trace_type=3, file_name='TEST24')


def GenerateNdimArray(N=10, size=500):
    # 可以作为3维图Z轴的输入参数
    # N为层数，size越大每一层越复杂
    L = []
    for i in range(N):
        L.append(np.random.uniform(i, i+1, size))
    return ToArray(*tuple(L))


def ToArray(*args):
    # 可以作为3维图Z轴的输入参数
    # 这里读取的args是一个含有多个一维array的tuple
    r = np.concatenate(args).reshape(len(args), -1)
    return r


def DataFrameToArray(df):
    # 可以作为3维图Z轴的输入参数
    return df.T.values


class DrawFigure(object):
    def __init__(self, data, file_path, file_name):
        self.x = data.index
        self.df = data
        self.y1 = data.iloc[:, 0]
        self.file_path = file_path
        self.file_name = file_name
        self.columns_names = self.df.columns.tolist()

    def trace(self, type, mode, number, x=None, y=None, scalebar=True, trace_type=None, trace_name=None):
        if trace_type is not None:
            type = None
            mode = None
        if (type == 'Scatter' and mode == 'markers') or trace_type == 1:
            return self.trace_type001(number, x, y, scalebar, trace_name)
        elif (type == 'Scatter' and mode == 'lines') or trace_type == 2:
            return self.trace_type002(number, x, y, trace_name)
        elif (type == 'Bar' and mode == 'markers') or trace_type == 3:
            return self.trace_type003(number, x, y, trace_name)
        elif (type == 'Scatter' and mode == 'lines+markers') or trace_type == 4:
            return self.trace_type004(number, x, y, trace_name)
        elif (type == 'Pie') or trace_type == 5:
            return self.trace_type005(number, x, y, trace_name)
        elif (type == 'Histogram') or trace_type == 6:
            return self.trace_type006(number, x, y, trace_name, mode)

    def trace_3d(self, type, mode, x=None, y=None, z=None, scalebar=True, trace_type=None, trace_name=None):
        if trace_type is not None:
            type = None
            mode = None
        if (type == 'Scatter3d' and mode == 'text') or trace_type == 1:
            return self.trace_3d_type001(x, y, z, trace_name)
        elif (type == 'Scatter3d' and mode == 'lines') or trace_type == 2:
            return self.trace_3d_type002(x, y, z, scalebar, trace_name)
        elif (type == 'Scatter3d' and mode == 'markers') or trace_type == 3:
            return self.trace_3d_type003(x, y, z, scalebar, trace_name)
        elif (type == 'Surface') or trace_type == 4:
            return self.trace_3d_type004(x, y, z, trace_name)
        elif (type == 'Mesh3d') or trace_type == 5:
            return self.trace_3d_type005(x, y, z, scalebar, trace_name)
# note:这行以下是layout

    def layout_type1(self, title_text=None, xaxis_title=None, yaxis_title=None, zaxis_title=None):
        return go.Layout(
            title=title_text,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,

            #  legend_title_text=legend_title_text,  # 图例标题
            legend=dict(
                #  yanchor="top",  # y轴顶部
                y=0.7,
                xanchor="left",  # x轴靠左
                x=0.01,
                traceorder="reversed",
                title_font_family="Times New Roman",  # 图例标题字体
                font=dict(  # 图例字体
                                 family="Courier",
                                 size=13,
                                 #  color="red"  # 颜色：红色
                ),
                #  bgcolor="LightSteelBlue",  # 图例背景色
                bordercolor="Black",  # 图例边框颜色和宽度
                borderwidth=2,


            ),

            #  margin=dict(l=0, r=0, b=0, t=0),  # 3d图中左右下上和边缘的距离
            margin=dict(l=0, r=0, t=40, b=0),
            scene=dict(
                xaxis=dict(
                    title_text=xaxis_title,
                    # gridcolor='rgb(255, 255, 255)',
                    # zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    # backgroundcolor='rgb(230, 230,230)'
                ),
                yaxis=dict(
                    title_text=yaxis_title,
                    # gridcolor='rgb(255, 255, 255)',
                    # zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    # backgroundcolor='rgb(230, 230,230)'
                ),
                zaxis=dict(
                    title_text=zaxis_title,
                    # gridcolor='rgb(255, 255, 255)',
                    # zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    # backgroundcolor='rgb(230, 230,230)'
                ))  # 3D图的背景颜色
        )

    def layout_type2(self, title_text=None, xaxis_title=None, yaxis_title=None, zaxis_title=None):
        return go.Layout(
            title=title_text,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            template='simple_white',
            #  legend_title_text=legend_title_text,  # 图例标题
            legend=dict(
                #  yanchor="top",  # y轴顶部
                y=0.7,
                xanchor="left",  # x轴靠左
                x=0.01,
                traceorder="reversed",
                title_font_family="Times New Roman",  # 图例标题字体
                font=dict(  # 图例字体
                                 family="Courier",
                                 size=13,
                                 #  color="red"  # 颜色：红色
                ),
                #  bgcolor="LightSteelBlue",  # 图例背景色
                bordercolor="Black",  # 图例边框颜色和宽度
                borderwidth=2,


            ),

            #  margin=dict(l=0, r=0, b=0, t=0),  # 3d图中左右下上和边缘的距离
            margin=dict(l=0, r=0, t=40, b=0),
            scene=dict(
                xaxis=dict(
                    title_text=xaxis_title,
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                ),
                yaxis=dict(
                    title_text=yaxis_title,
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                ),
                zaxis=dict(
                    title_text=zaxis_title,
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                ))  # 3D图的背景颜色
        )


# note:这行以下是2d图的trace

    def trace_type001(self, number=None, x=None, y=None, scalebar=True, trace_name=None):
        """ 颜色鲜艳的散点图 """
        if x is None:
            x = self.df.index
        if y is None:
            y = self.df.iloc[:, number]
        if trace_name is not None:
            name = trace_name
        elif number is not None:
            name = '%s' % self.columns_names[number]
        else:
            name = None
        if number == 0 and scalebar:
            showscale = True
        else:
            showscale = False
        trace = go.Scatter(
            x=x,
            y=y,
            mode='markers',
            name=name,
            marker=dict(
                size=16,
                color=y,
                colorscale='Viridis',
                showscale=showscale
            )

        )
        return trace

    def trace_type002(self, number=None, x=None, y=None, trace_name=None):
        """ 普通折线图 """
        if x is None:
            x = self.df.index
        if y is None:
            y = self.df.iloc[:, number]

        if trace_name is not None:
            name = trace_name
        elif number is not None:
            name = '%s' % self.columns_names[number]
        else:
            name = None
        trace = go.Scatter(x=x,
                           y=y,
                           mode='lines',
                           #    line=dict(color='orangered'),
                           name=name)
        return trace

    def trace_type003(self, number=None, x=None, y=None, trace_name=None):
        """ 相邻柱子颜色不同的柱状图 """
        if x is None:
            x = self.df.index
        if y is None:
            y = self.df.iloc[:, number]
        if trace_name is not None:
            name = trace_name
        elif number is not None:
            name = '%s' % self.columns_names[number]
        else:
            name = None
        if number % 2 == 0:
            yanse = 'rgb(49,130,189)'
        else:
            yanse = 'rgb(204,204,204)'
        trace = go.Bar(
            x=x,
            y=y,
            name=name,
            marker=dict(
                color=yanse
            )
        )
        return trace

    def trace_type004(self, number=None, x=None, y=None, trace_name=None):
        """ 带散点的折线图 """
        if x is None:
            x = self.df.index
        if y is None:
            y = self.df.iloc[:, number]
        if trace_name is not None:
            name = trace_name
        elif number is not None:
            name = '%s' % self.columns_names[number]
        else:
            name = None
        trace = go.Scatter(x=x,
                           y=y,
                           mode='lines+markers', name=name)
        return trace

    def trace_type005(self, number=None, x=None, y=None, trace_name=None):
        """ 饼图 """
        if x is None:
            x = self.df.index
        if y is None:
            y = self.df.iloc[:, number]
        if trace_name is not None:
            name = trace_name
        elif number is not None:
            name = '%s' % self.columns_names[number]
        else:
            name = None
        PullList = []
        # MaxIndex =
        for i in range(len(y)):
            if i == np.argmax(y):
                PullList.append(0.1)
            else:
                PullList.append(0)

        trace = go.Pie(labels=x,
                       values=y,
                       domain={"x": [0, 1]},  # 饼图的位置,
                       name=name,
                       hoverinfo="label+percent+name",
                       textinfo="percent",
                       pull=PullList,  # 各个饼弹出的程度
                       hole=0.4  # 内圈大小
                       )
        return trace

    def trace_type006(self, number=None, x=None, y=None, trace_name=None, mode=' '):
        """ 直方图 """
        if x is None:
            x = self.df.iloc[:, number]
        if trace_name is not None:
            name = trace_name
        elif number is not None:
            name = '%s' % self.columns_names[number]
        else:
            name = None
        if mode == 'markers':
            mode = 'probability'
        if number is None:
            yanse = 'rgb(204,204,204)'
        elif number % 2 == 0:
            yanse = 'rgb(49,130,189)'
        else:
            yanse = 'rgb(204,204,204)'
        width = (np.max(x)-np.min(x))/50
        trace = go.Histogram(
            # text=np.arange(1, 10000, 1),
            x=x,
            name=name,
            xbins=dict(size=width),
            histfunc='count',
            # 可选参数[' ', 'percent', 'probability', 'density', 'probability density']
            histnorm=mode,
            marker=dict(color=yanse, line=dict(color='white', width=1))
        )
        return trace
# note:这行以下是3d的trace

    def trace_3d_type001(self, x, y, z, trace_name=None):
        """ 3d图 """
        if x is None:
            x = self.df.iloc[:, 0]
        if y is None:
            y = self.df.iloc[:, 1]
        if z is None:
            z = self.df.iloc[:, 2]

        trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            hoverinfo='none',
            mode='text',
            name=trace_name,
            text=self.columns_names,
            textposition='top center',
            showlegend=True,
        )
        return [trace]

    def trace_3d_type002(self, x, y, z, scalebar=True, trace_name=None):
        """ 3d线图 """
        if x is None:
            x = self.df.iloc[:, 0]
        if y is None:
            y = self.df.iloc[:, 1]
        if z is None:
            z = self.df.iloc[:, 2:].T.values
        elif type(z) == pd.core.series.Series:
            z = z.values[None]
        elif type(z) == pd.core.frame.DataFrame:
            z = DataFrameToArray(z)
        elif z.ndim != 1 and len(z) == self.df.shape[0]:
            z = z.transpose()
        elif z.ndim == 1:
            z = z[None]
        trace_set = []
        for i in range(len(z)):
            if i == 0 and scalebar:
                showscale = True
            else:
                showscale = False
            trace = go.Scatter3d(
                x=x,
                y=y,
                z=pd.Series(z[i]),
                name=trace_name,
                hoverinfo='none',
                mode='lines',
                line=dict(color=pd.Series(z[i]),
                          colorscale='electric',   # 渐变色选择
                          showscale=showscale  # 右侧显示颜色尺标
                          ),
                showlegend=True,
            )
            trace_set.append(trace)
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        colorset=['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose',
                'balance', 'blackbody', 'bluered', 'blues', 'blugrn',
                'bluyl', 'brbg', 'brwnyl', 'bugn', 'bupu', 'burg',
                'burgyl', 'cividis', 'curl', 'darkmint', 'deep',
                'delta', 'dense', 'earth', 'edge', 'electric',
                'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens',
                'greys', 'haline', 'hot', 'hsv', 'ice', 'icefire',
                'inferno', 'jet', 'magenta', 'magma', 'matter',
                'mint', 'mrybm', 'mygbm', 'oranges', 'orrd', 'oryel',
                'peach', 'phase', 'picnic', 'pinkyl', 'piyg','plasma',
                'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',
                'puor', 'purd', 'purp', 'purples', 'purpor',
                'rainbow', 'rdbu', 'rdgy', 'rdpu', 'rdylbu',
                'rdylgn', 'redor', 'reds', 'solar', 'spectral',
                'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
                'tealrose', 'tempo', 'temps', 'thermal', 'tropic',
                'turbid', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd']
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        return trace_set

    def trace_3d_type003(self, x, y, z, scalebar=True, trace_name=None):
        """ 3d图散点 """
        if x is None:
            x = self.df.iloc[:, 0]
        if y is None:
            y = self.df.iloc[:, 1]
        if z is None:
            z = self.df.iloc[:, 2:].T.values
        elif type(z) == pd.core.series.Series:
            z = z.values[None]
        elif type(z) == pd.core.frame.DataFrame:
            z = DataFrameToArray(z)
        elif z.ndim != 1 and len(z) == self.df.shape[0]:
            z = z.transpose()
        elif z.ndim == 1:
            z = z[None]

        trace_set = []
        for i in range(len(z)):
            if i == 0 and scalebar:
                showscale = True
            else:
                showscale = False
            trace = go.Scatter3d(
                x=x,
                y=y,
                z=pd.Series(z[i]),
                name=trace_name,
                hoverinfo='none',
                mode='markers',
                marker=dict(  # 标记设置
                    size=12,
                    color=pd.Series(z[i]),
                    colorscale='bluered',   # 渐变色选择
                    opacity=0.8,  # 透明度设置
                    showscale=showscale  # 右侧显示颜色尺标
                ),
                showlegend=True,
            )
            trace_set.append(trace)
        return trace_set

    def trace_3d_type004(self, x, y, z, trace_name=None):
        """ 3d自动组合多曲面成单曲面图 """
        if x is None:
            x = self.df.iloc[:, 0]
        if y is None:
            y = self.df.iloc[:, 1]
        if z is None:
            z = self.df.iloc[:, 2:].T.values

        trace = go.Surface(
            x=x,
            y=y,
            z=z,
            name=trace_name,
            colorscale='speed',   # 渐变色选择
            # opacity=0.8,  # 透明度设置
            showscale=True,  # 右侧显示颜色尺标
            showlegend=True,
            lighting=dict(ambient=0.9)
        )
        return [trace]

    def trace_3d_type005(self, x, y, z, scalebar=True, trace_name=None):
        """ 3d叠加曲面图 """
        if x is None:
            x = self.df.iloc[:, 0]
        if y is None:
            y = self.df.iloc[:, 1]
        if z is None:
            z = self.df.iloc[:, 2:].T.values
        elif type(z) == pd.core.series.Series:
            z = z.values[None]
        elif type(z) == pd.core.frame.DataFrame:
            z = DataFrameToArray(z)
        elif z.ndim != 1 and len(z) == self.df.shape[0]:
            z = z.transpose()
        elif z.ndim == 1:
            z = z[None]
        trace_set = []

        for i in range(len(z)):
            if i == 0 and scalebar:
                showscale = True
            else:
                showscale = False
            trace = go.Mesh3d(
                x=x,
                y=y,
                z=pd.Series(z[i]),
                name=trace_name,
                hoverinfo='x+y+z',
                intensity=pd.Series(z[i]),
                colorscale='Portland',   # 渐变色选择
                opacity=0.8,  # 透明度设置
                showscale=showscale,  # 右侧显示颜色尺标
                showlegend=True
            )
            trace_set.append(trace)
        return trace_set

# note:这行以下是plot行数

    def plot(self, x=None, y=None, type='Scatter', mode='markers', layout_num='1', file_path=None, file_name=None, show=False, auto_open=False, xaxis_name=None, yaxis_name=None, title_name=None, trace_type=None, trace_name=None):
        if file_path is None:
            file_path = self.file_path
        if file_name is None:
            file_name = self.file_name
        if layout_num is not None:
            layout_num = str(layout_num)
        data = []

        if x is not None or y is not None:
            data.append(self.trace(type, mode, None,
                                   x, y, trace_type=trace_type, trace_name=trace_name))

        else:
            trace = OrderedDict()
            for number in range(self.df.shape[1]):
                trace[number] = self.trace(type, mode, number)
                data.append(trace[number])

        layout = eval('self.layout_type'+layout_num +
                      '(file_name,xaxis_name,yaxis_name)')
        fig = go.Figure(data=data, layout=layout)
        if title_name is not None:
            fig.update_layout(title=title_name)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if not show:
            py.offline.plot(fig, filename=file_path+"%s.html" %
                            file_name, auto_open=auto_open)
        else:
            py.offline.iplot(fig)
        return fig

    def plot_one_column(self, col_num=0, type='Scatter', mode='markers', layout_num='1', file_path=None, file_name=None, show=False, auto_open=False, xaxis_name=None, title_name=None, trace_type=None):
        if file_path is None:
            file_path = self.file_path
        if file_name is None:
            file_name = self.file_name
        if layout_num is not None:
            layout_num = str(layout_num)
        data = []
        trace = self.trace(type, mode, col_num, trace_type=trace_type)
        data.append(trace)

        if not os.path.exists(file_path):
            os.makedirs(file_path)
        layout = eval('self.layout_type'+layout_num +
                      '(file_name,xaxis_name,self.columns_names[col_num])')

        fig = go.Figure(data=data, layout=layout)
        if title_name is not None:
            fig.update_layout(title=title_name)

        if not show:
            py.offline.plot(fig, filename=file_path+"%s.html" %
                            file_name, auto_open=auto_open)
        else:
            py.offline.iplot(fig)
        return fig

    def plot_subplots(self, type='Scatter', mode='markers', file_path=None, file_name=None, show=False, auto_open=False, title_name=None, trace_type=None, xaxis_name=None, yaxis_name=None, zaxis_name=None, layout_num='1'):
        """ 画子图 """
        if file_path is None:
            file_path = self.file_path
        if file_name is None:
            file_name = self.file_name
        if layout_num is not None:
            layout_num = str(layout_num)
        # 定义figure框架
        columns_number = int(np.ceil(np.sqrt(self.df.shape[1])))
        specs = []
        number = 0
        subplot_titles = []
        while number < self.df.shape[1]:
            line = []
            for j in range(columns_number):
                if number >= self.df.shape[1]:
                    line.append(None)
                else:
                    line.append({"type": type})
                    subplot_titles.append(self.columns_names[number])
                number += 1

            specs.append(line)

        fig = make_subplots(rows=int(np.ceil(number//columns_number)),  # 将画布分为两行
                            cols=columns_number,  # 将画布分为两列
                            subplot_titles=subplot_titles,  # 子图的标题
                            specs=specs  # 饼图子图需要定义specs
                            )

        layout = eval('self.layout_type'+layout_num +
                      '(file_name,xaxis_name, yaxis_name,zaxis_name)')
        fig.update_layout(layout)

        if title_name is not None:
            fig.update_layout(title=title_name)
        # 添加trace
        trace = OrderedDict()
        number = 0

        while number < self.df.shape[1]:
            for j in range(columns_number):
                i = number // columns_number
                trace[number] = self.trace(
                    type, mode, number, trace_type=trace_type)
                fig.append_trace(trace[number], row=i+1, col=j+1)
                number += 1
                if number >= self.df.shape[1]:
                    break

        # 画图并保存
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        if not show:
            py.offline.plot(fig, filename=file_path+"%s.html" %
                            file_name, auto_open=auto_open)
        else:
            py.offline.iplot(fig)
        return fig

    def plot_3dfigures(self, x=None, y=None, z=None, type='Scatter3d', mode='markers', layout_num='1', file_path=None, file_name=None, show=False, auto_open=False, xaxis_name=None, yaxis_name=None, zaxis_name=None, title_name=None, trace_type=None, trace_name=None):
        if file_path is None:
            file_path = self.file_path
        if file_name is None:
            file_name = self.file_name
        if layout_num is not None:
            layout_num = str(layout_num)

        data = self.trace_3d(type, mode, x, y, z, trace_name=trace_name)

        if not os.path.exists(file_path):
            os.makedirs(file_path)
        layout = eval('self.layout_type'+layout_num +
                      '(file_name,xaxis_name, yaxis_name,zaxis_name)')
        fig = go.Figure(data=data, layout=layout)
        if title_name is not None:
            fig.update_layout(title=title_name)
        if not show:
            py.offline.plot(fig, filename=file_path+"%s.html" %
                            file_name, auto_open=auto_open)
        else:
            py.offline.iplot(fig)
        return fig

    def Add_Trace_And_plot(self, fig, x=None, y=None, number=0, type='Scatter', mode='markers',  show=False, file_path=None, file_name=None, auto_open=False, trace_type=None, trace_name=None):
        if x is not None and y is not None:
            number = None
        if file_path is None:
            file_path = self.file_path
        if file_name is None:
            file_name = self.file_name
        fig.add_trace(self.trace(type, mode, number, x, y,
                                 scalebar=False, trace_type=trace_type, trace_name=trace_name))
        # 画图并保存
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if not show:
            py.offline.plot(fig, filename=file_path+"%s.html" %
                            file_name, auto_open=auto_open)
        else:
            py.offline.iplot(fig)
        return fig

    def Add_Trace_3d_And_plot(self, fig, x=None, y=None, z=None, type='Scatter3d', mode='markers',  show=False, file_path=None, file_name=None, auto_open=False, trace_type=None, trace_name=None):
        if file_path is None:
            file_path = self.file_path
        if file_name is None:
            file_name = self.file_name
        # 3d图是集合，所以要取其中一个元素才是trace object
        fig.add_trace(self.trace_3d(type, mode, x=x, y=y, z=z,
                                    scalebar=False, trace_type=trace_type, trace_name=trace_name)[0])

        # 画图并保存
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if not show:
            py.offline.plot(fig, filename=file_path+"%s.html" %
                            file_name, auto_open=auto_open)
        else:
            py.offline.iplot(fig)
        return fig

    def Add_Trace_To_Subplots_And_plot(self, fig, x=None, y=None, z=None, col_num=0, type='Scatter', mode='markers',  show=False, file_path=None, file_name=None, auto_open=False, trace_type=None, trace_name='NewTrace', xaxis_name=None, yaxis_name=None, zaxis_name=None, layout_num='1'):
        """ 画子图 """
        if file_path is None:
            file_path = self.file_path
        if file_name is None:
            file_name = self.file_name
        if layout_num is not None:
            layout_num = str(layout_num)

        # 定义figure框架
        columns_number = int(np.ceil(np.sqrt(len(fig.data)+1)))
        specs = []
        number = 0
        subplot_titles = []
        while number <= len(fig.data):
            line = []
            for j in range(columns_number):
                if number > len(fig.data):
                    line.append(None)
                elif number == len(fig.data):
                    line.append({"type": type})
                    subplot_titles.append(trace_name)
                else:
                    line.append({"type": fig.data[number].type})
                    subplot_titles.append(fig.data[number].name)
                number += 1

            specs.append(line)
        # print(len(fig.data))
        # print(specs)

        figure = make_subplots(rows=int(np.ceil(number//columns_number)),  # 将画布分为两行
                               cols=columns_number,  # 将画布分为两列
                               subplot_titles=subplot_titles,  # 子图的标题
                               specs=specs  # 饼图子图需要定义specs
                               )

        # layout = fig.layout
        layout = eval('self.layout_type'+layout_num +
                      '(file_name,xaxis_name, yaxis_name,zaxis_name)')
        figure.update_layout(layout)
        # print(int(np.ceil(number//columns_number)))
        # 添加trace

        if type == 'Scatter3d' or type == 'Mesh3d' or type == 'Surface':
            # 因为3d图的函数输出的trace是集合，所以要取一下[0]
            newtrace = self.trace_3d(type, mode, x, y, z,
                                     scalebar=False, trace_type=trace_type, trace_name=trace_name)[0]
        else:
            newtrace = self.trace(type, mode, col_num, x, y,
                                  scalebar=False, trace_type=trace_type, trace_name=trace_name)
        number = 0

        while number <= len(fig.data):
            for j in range(columns_number):
                i = number // columns_number
                if number == len(fig.data):
                    # print(number)
                    figure.append_trace(newtrace, row=i+1, col=j+1)
                    number += 1
                    break
                figure.append_trace(fig.data[number], row=i+1, col=j+1)
                number += 1

        # 画图并保存
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        if not show:
            py.offline.plot(figure, filename=file_path+"%s.html" %
                            file_name, auto_open=auto_open)
        else:
            py.offline.iplot(figure)

        return figure

    def Update_Layout_And_plot(self, fig, title_name=None, show=False, file_path=None, file_name=None, auto_open=False, layout_num='1', xaxis_title=None, yaxis_title=None, zaxis_title=None):
        if file_path is None:
            file_path = self.file_path
        if file_name is None:
            file_name = self.file_name
        if layout_num is not None:
            layout_num = str(layout_num)
        ''' layout具体参数 '''

        fig.update_layout(eval('self.layout_type'+layout_num +
                               '(title_name,xaxis_title,yaxis_title,zaxis_title)'))

        ''' 生成图片 '''

        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if not show:
            py.offline.plot(fig, filename=file_path+"%s.html" %
                            file_name, auto_open=auto_open)
        else:
            py.offline.iplot(fig)

        return fig

    def Update_Layout_for_subplots_And_plot(self, fig, subplot_num, title_name=None, file_name=None, file_path=None, show=False, auto_open=False, layout_num='1', xaxis_title=None, yaxis_title=None, zaxis_title=None):
        if file_path is None:
            file_path = self.file_path
        if file_name is None:
            file_name = self.file_name
        if layout_num is not None:
            layout_num = str(layout_num)
        xaxis = 'xaxis'+str(subplot_num)
        yaxis = 'yaxis'+str(subplot_num)
        fig["layout"][xaxis].update({"title": xaxis_title})
        fig["layout"][yaxis].update({"title": yaxis_title})
        # 画图并保存
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if not show:
            py.offline.plot(fig, filename=file_path+"%s.html" %
                            file_name, auto_open=auto_open)
        else:
            py.offline.iplot(fig)

        return fig


if __name__ == '__main__':
    main()