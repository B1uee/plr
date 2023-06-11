from matplotlib.pylab import gca, figure, plot, subplot, title, xlabel, ylabel, xlim,show
from matplotlib.lines import Line2D
import segment
import fit
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import math
from datetime import date, datetime
import numpy.linalg as LA
from scipy.interpolate import interp1d


def draw_plot(data, segments, plot_title):
    fig = figure(figsize=(16,6))
    plot(data[:,0],data[:,1],alpha=0.8,color='red',label='raw_data')
    plot(segments[:,0],segments[:,1],alpha=0.8,color='blue',label='segment_data')
    title(plot_title)
    plt.legend(loc='best')
    xlabel("time")
    ylabel("cwnd")
    plt.savefig(plot_title+'.png')

def load_data(path):
    with open(path, 'r') as f:
        file_lines = f.readlines()
    data=[]   #输入xy二维数据
    for line in file_lines:
        if line.strip()!='0.0000000 0.0000000':
            tmp=line.strip().split(' ')
            x=float(tmp[0])
            y=float(tmp[1])
            data.append([x,y])
    data=np.asarray(data)
    return data

def get_TurnPoint(data, angle_thred):
    ''' 
    :paras data:时间序列数据
    :paras angle_thred: 判定其是否为转折点的角度阈值
 
    :return turnPoints: 预选的转折点
    '''
    turnPoints = []
    for index in range(1, len(data)-1):
        # 这里存在一个隐性的 时间序列采样是否均匀 的问题 
        # 这里如果时间采样不均匀，记得将 atan() 中的数据除以对应时间差
        theta1 = math.atan((data[index][1] - data[index - 1][1])/(data[index][0] - data[index - 1][0])) * (180 / math.pi)
        theta2 = math.atan((data[index + 1][1] - data[index][1])/(data[index + 1][0] - data[index][0])) * (180 / math.pi)
        if theta1 * theta2 < 0:
            alpha = abs(theta1) + abs(theta2)
        else:
            alpha = abs(theta1 - theta2)
        if alpha > angle_thred:
            turnPoints.append(index)
    turnPoints.insert(0, 0)
    turnPoints.append(len(data)-1)
    return turnPoints

def merge_cost(data, merge_start, merge_end):
    ''' 
    本方法使用 least-squares solution to a linear matrix equation 方法进行直线拟合。
    同时获得拟合误差。
    调用 numpy.linalg.lstsq() 方法
    :param data: 时间序列数据
    :param merge_start: 合并起始点
    :param merge_end: 合并终点
 
    :return error: 合并产生的误差
    '''
    x = np.array(data[merge_start:merge_end+1, 0])
    y = np.array(data[merge_start:merge_end+1, 1])
    A = np.ones((len(x), 2), np.float64)
    A[:,0] = x
    try:
        (result, residual, rank, s) = LA.lstsq(A, y, rcond=None) #回归系数、残差平方和、自变量X的秩、X的奇异值
        error = math.sqrt(residual[0])

    except np.linalg.LinAlgError:
        error = 0.0
    except IndexError:
        error = 0.0
 
    return error, result

def bottomUp(data, turn_points, error_thred, error_estimator=merge_cost):
    ''' 
    :param data: 时间序列数据
    :param turn_points: 转折点数据
    :param error_thred: 融合过程中的误差阈值, 超过阈值后, 融合停止
    :param error_estimator: 融合过程中的误差估计方法
 
    :return segments
    '''
    index = 0
    segments = [turn_points[0]] # 加入起始的开始点
    plr_data = [] # 插值成完整时间序列，方便后续比对
    len_turnPoints = len(turn_points) - 1
    
    while index < len_turnPoints:
        start = index
        end = index + 1
        error, ratio = error_estimator(data, turn_points[start], turn_points[end])
        while error < error_thred:
            end += 1
            if not (end < len_turnPoints): 
                end = len_turnPoints
                break
            error, ratio = error_estimator(data, turn_points[start], turn_points[end])
        segments.append(turn_points[end])
        x_range = data[turn_points[start]:turn_points[end], 0]
        #new_y = ratio[0]*x_range+ratio[1]
        tmp_data = np.c_[x_range, ratio[0]*x_range+ratio[1]]
        plr_data.extend(tmp_data.tolist())
        index = end
    return segments, np.array(plr_data)

def new_segment(angle_thred, error_thred):
    #writer = SummaryWriter("logs/"+datetime.now().strftime("%m-%d-%H_%M_%S"))
    raw_data = load_data("trace/newreno/5/cwnd.txt")
    turn_points = get_TurnPoint(raw_data, angle_thred)
    segments, plr_data = bottomUp(raw_data, turn_points, error_thred, merge_cost) 
    draw_plot(raw_data, raw_data[segments], "RawData & Segments with {}_{}".format(angle_thred, error_thred))
    save_path = "test/newreno"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.savetxt(os.path.join(save_path,'5__plrdata_{}_{}.txt'.format(angle_thred, error_thred)), np.asarray(plr_data),fmt='%.8f %.6f')
    np.savetxt(os.path.join(save_path,'5_segements_{}_{}.txt'.format(angle_thred, error_thred)), np.asarray(raw_data[segments]),fmt='%.8f %.6f')
    #writer.close()

def DTWDistance(s1, s2):
    DTW={}
 
    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0
 
    for i in range(len(s1)):
        for j in range(len(s2)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
 
    return np.sqrt(DTW[len(s1)-1, len(s2)-1])

if __name__ == "__main__":
    new_segment(1e-6,1e-10)