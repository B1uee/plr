from matplotlib.pylab import gca, figure, plot, subplot, title, xlabel, ylabel, xlim,show
from matplotlib.lines import Line2D
import segment
import fit
import matplotlib.pyplot as plt
import numpy as np
import os


def draw_plot(data,plot_title):
    plot(data[:,0],data[:,1],alpha=0.8,color='red')
    title(plot_title)
    xlabel("Samples")
    ylabel("Signal")
    xlim((0,len(data)-1))
def draw_segments(segments):
    ax = gca()
    for segment in segments:
        line = Line2D((segment[0],segment[2]),(segment[1],segment[3]))
        ax.add_line(line)


def topdownsegment(max_error=0.001):

    for alg in ['newreno']:
        #path='/home/ml4net/LJH/LJH/contrastive-predictive-cc/data/out/'+alg  
        path = os.getcwd() + "/trace/" + alg
        files = os.listdir(path)
        for file in files:
            tmp_path = "{}/{}".format(path, file)
            tmp_files = os.listdir(tmp_path)
            for tmp_file in tmp_files:
                if tmp_file.endswith('.txt'):
                    data, speeds_all, segments_all = load_data(os.path.join(path,file+'/'+tmp_file), max_error)
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
                            i+=1
                        else:
                            start_index+=1
                            if start_index>=len(segments_all):
                                break
                    save_path = os.getcwd()+'/test/'+alg
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    np.savetxt(os.path.join(save_path,file+'.txt'),np.asarray(output),fmt='%.8f %.6f')
                #np.savetxt('/home/ml4net/LJH/LJH/PLR/out_speed/'+alg+'/'+file,np.asarray(output),fmt='%.8f %.6f')

def load_data(path, max_error):
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
    speeds=[]
    segments_all=[]
    speeds_all=[]
    start=0
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
    return data, speeds_all, segments_all


if __name__ == "__main__":
    #data, _,_, = load_data()
    topdownsegment(0.0001)
    # with open("test/newreno/5.txt", 'r') as f:
    #     file_lines = f.readlines()
    # data=[]   #输入xy二维数据
    # for line in file_lines[0:2000]:
    #     if line.strip()!='0.0000000 0.0000000':
    #         tmp=line.strip().split(' ')
    #         x=float(tmp[0])
    #         y=float(tmp[1])
    #     data.append([x,y])
    # data=np.asarray(data)
    # draw_plot(data,"topdown")
    # plt.savefig('test/newreno/fig_test_1.png')
