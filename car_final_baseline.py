# @author: lily
'''
http://www.dcjingsai.com/common/bbs/topicDetails.html?tid=2464
思路：
统计车辆在星期一至星期日最常去的哪些地点，对这些地点做标记，之后去过为1，不去为0，以此作标签，把概率最高的地点推荐给选手
使用模型：
Logistics
优化方法：
1、继续细化统计车辆在1-7的哪个小时去过哪些地点，如统计距0点的分钟数或者小时数，即按小时做推荐。另外还可以针对节假日做标记
2、挖掘同时间、同地域车辆是否相似性，对节假日的推荐单独处理
3、尝试使用svm、xgb、lgb等模型，并进行参数调整
4、模型融合
'''
import pandas as pd
import numpy as np

# 评分算法
from math import radians, atan, tan, sin, acos, cos
def getDistance(latA, lonA, latB, lonB):
    ra = 6378140 # 赤道半径：m
    rb = 6356755 # 极线半径：m
    flatten = (ra - ra) / ra
    # change angle to radians
    radLatA = radians(latA) # 弧度
    radLonA = radians(lonA)
    radLatB = radians(latB)
    radLonB = radians(lonB)

    try:
        pA = atan(rb / ra * tan(radLatA))
        pB = atan(rb / ra * tan(radLatB))
        x = acos(sin(pA) * sin(pB) + cos(pB) * cos(radLonA - radLonB))
        c1 = (sin(x) - x) * (sin(pA) + sin(pB)) ** 2 / cos(x / 2) ** 2
        c2 = (sin(x) + x) * (sin(pA) - sin(pB)) ** 2 / sin(x / 2) ** 2
        dr = flatten / 7 * (c1 - c2)
        distance = ra * (x + dr)
        return distance # master
    except:
        return 0.000000000001

def f(d):
    return 1 + (1 + np.exp(-(d - 1000) / 250))

# 计算误差
def getDistanceFromDF(data):
    tmp = data[['end_lat', 'end_lon', 'predict_end_lat', 'predict_end_lon']].astype(float)
    error = [] # 设置一个空列表error
    for i in tmp.values:
        t = getDistance(i[0], i[1], i[2], i[3]) # 逐条计算误差
        error.append(t)
    print(np.sum(f(np.array(error))) / tmp.shape[0])

# 转化数据集中的日期为pandas中的datatime类型
# 生成两个新特征，出发的星期和出发的小时
def dateConvert(data, is_Train):
    print('convert string to datetime')
    data['start_time'] = pd.to_datetime(data['start_time']) # 转化开始时间
    if is_Train:
        data['end_time'] = pd.to_datetime(data['end_time'])
    data['weekday'] = data['start_time'].dt.weekday + 1 # 生成新的一列，weekday对应0-6，这里加1
    data['hour'] = data['start_time'].dt.hour
    return data
# 合并经纬度, 常用将经纬度转为geohash编码
def latitude_longitude_to_go(data, is_Train):
    tmp = data[['start_lat', 'start_lon']] # 取出出发地经纬度
    start_geohash = [] # 定义一个空列表
    for t in tmp.values: # 逐行遍历出发地经纬度
        start_geohash.append(str(round(t[0], 5)) + '' + str(round(t[1], 5))) # 将经纬度合并
    data['startGo'] = start_geohash # 生成新的一列，值为合并后的经纬度
    if is_Train:
        tmp = data[['end_lat', 'end_lon']]
        end_geohash = []
        for t in tmp.values:
            end_geohash.append(str(round(t[0], 5)) + '' + str(round(t[1], 5)))
        data['endGo'] = end_geohash # 生成新的一列，之为合并后目的地经纬度
    return data


