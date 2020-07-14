#coding:utf8
"""
对相似度进行对一些评估方法:
1. 欧式距离
2. 皮尔逊相关系数
"""

import math

import numba as nb
from numba.extending import overload
from numba.typed import List

@nb.njit
def euclidean(x:List, y:List)->float:
    """
    直接计算欧式距离
    $\sqrt{\displaystyle{\sum_{i=1}^n(x_i-y_i)^2}}$
    """
    if len(x) != len(y):
        raise ValueError("Differente Length")

    result = 0
    
    for x_i, y_i in zip(x, y):
        result += (x_i - y_i) ** 2
    return result


def euclideanPower(x:tuple, y:tuple)->float:
    """
    以效用方式表达欧式距离的结果，计算方式：
    $\frac{1}{1+\text{euclidean}}$
    这样的结果得到是 [0, 1] 的范围内，这样得到的结果更符合相似度的表示方法
    """
    distance = euclidean(tuple(x), tuple(y))
    return 1 / (1 + distance)


@nb.njit
def dot(x, y):
    """
    计算点积
    """
    if len(x) != len(y):
        raise ValueError("x and y are differente size")

    result = 0
    for x_i, y_i in zip(x, y):
        result += x_i * y_i
    
    return result

