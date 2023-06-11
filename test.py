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
    plt.savefig(plot_title)

def draw_segments(segments):
    ax = gca()
    for segment in segments:
        line = Line2D((segment[0],segment[2]),(segment[1],segment[3]))
        ax.add_line(line)


def topdownsegment(max_error=0.001):

    for alg in ['newreno']:
        #path='/home/ml4net/LJH/LJH/contrastive-predictive-cc/data/out/'+alg  
        path = os.getcwd() + "/trace/" + alg
        #writer = SummaryWriter("logs")
        files = os.listdir(path)
        for file in files:
            tmp_path = "{}/{}".format(path, file)
            tmp_files = os.listdir(tmp_path)
            for tmp_file in tmp_files:
                if tmp_file.endswith('.txt'):
                    data = load_data(os.path.join(path,file+'/'+tmp_file))
                    speeds=[]
                    segments_all=[]
                    speeds_all=[]
                    start=0
                    '''
                    while start<=len(data):
                        end=min(start+5000,len(data)-1)
                        input_data=data[start:end]
                        start+=5000
                        segments,speeds = segment.topdownsegment(input_data, fit.regression, fit.sumsquared_error, max_error,speeds=speeds)
                        if start==0:
                            segments_all=segments
                            speeds_all=speeds
                        else:
                            segments_all+=segments
                            speeds_all+=speeds
                    '''
                    #segments,speeds = segment.topdownsegment(data, fit.regression, fit.sumsquared_error, max_error,speeds=speeds)
                    #segments_all=segments
                    #speeds_all=speeds
                    segments = segment.raw_topdownsegment(data, fit.regression, fit.raw_sumsquared_error, max_error)
                    print(segment[0:5])
                    draw_plot(data, data[segments], "RawData & Segments with {}".format(max_error))
                    a = 1
                    '''
                    draw_plot(data,"Sliding window with regression")
                    plt.savefig('fig_test_1.png')
                    draw_segments(segments)
                    plt.savefig('fig_test.png')
                    '''
                    #按照输入，匹配每个值的斜率进行输出
                    start_index=0
                    output=[]
                    i=0
                    while i<len(data):
                        x=data[i][0]
                        y=data[i][1]
                        #segments[x1,y1,x2,y2]
                        segments_all=np.asarray(segments_all)
                        #segment=segments[start_index]
                        if x>=segments_all[start_index][0] and x<=segments_all[start_index][2]:
                            
                            speed_x=speeds_all[start_index][0]
                            output.append([x,speed_x])
                            #writer.add_scalar("pre_data", speed_x, x)
                            i+=1
                        else:
                            start_index+=1
                            if start_index>=len(segments_all):
                                break
                    #writer.close()
                    save_path = os.getcwd()+'/test/'+alg
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                draw_plot(data, np.array(output), "RawData & Segments with {}.png".format(max_error))
                np.savetxt(os.path.join(save_path,file+'_2.txt'),np.asarray(output),fmt='%.8f %.6f')
                #np.savetxt('/home/ml4net/LJH/LJH/PLR/out_speed/'+alg+'/'+file,np.asarray(output),fmt='%.8f %.6f')

def load_data(path):
    with open(path, 'r') as f:
        file_lines = f.readlines()
    data=[]   #输入xy二维数据
    for line in file_lines[0:2000]:
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
    x = np.arange(merge_start, merge_end+1) # +1 的原因是 np.arange 为左闭右开
    y = np.array(data[merge_start:merge_end+1])
    A = np.ones((len(x), 2), np.float64)
    A[:,0] = x
    try:
        (result, residual, rank, s) = LA.lstsq(A, y[:,1], rcond=None)
        error = math.sqrt(residual[0])

    except np.linalg.LinAlgError:
        error = 0.0
    except IndexError:
        error = 0.0
 
    return error

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
    len_turnPoints = len(turn_points) - 1
 
    while index < len_turnPoints:
        start = index
        end = index + 1
        while error_estimator(data, turn_points[start], turn_points[end]) < error_thred:
            end += 1
            if not (end < len_turnPoints): break
        segments.append(turn_points[end])
        index = end
 
    return segments

def new_segment(angle_thred, error_thred):
    #writer = SummaryWriter("logs/"+datetime.now().strftime("%m-%d-%H_%M_%S"))
    raw_data = load_data("trace/newreno/5/cwnd.txt")
    turn_points = get_TurnPoint(raw_data, angle_thred)
    segments = bottomUp(raw_data, turn_points, error_thred, merge_cost)
    f1 = interp1d(raw_data[segments][0],raw_data[segments][1],kind='linear',bounds_error=False)
    new_y= f1(raw_data[:,0])
    new_data = np.c_[raw_data[:,0], new_y]
    draw_plot(raw_data, raw_data[segments], "RawData & Segments with {}_{}".format(angle_thred, error_thred))
    save_path = "test/newreno"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #np.savetxt(os.path.join(save_path,'5_new_segment_new_data.txt'), np.asarray(new_data),fmt='%.8f %.6f')
    np.savetxt(os.path.join(save_path,'5_segments_{}_{}.txt'.format(angle_thred, error_thred)), np.asarray(raw_data[segments]),fmt='%.8f %.6f')


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
    #new_segment(1e-6,1e-12)
    topdownsegment(1e-4)
    #raw_data = load_data("trace/newreno/5/cwnd.txt")
    #segments = load_data("test/newreno/5_new_segment_new_data.txt")
    #draw_plot(raw_data, segments, "test")
    
    
    #dtw_dist = DTWDistance(raw_data,segments)
    #print(dtw_dist)

    