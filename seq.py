import numpy as np
from matplotlib.pylab import gca, figure, plot, subplot, title, xlabel, ylabel, xlim,show
from matplotlib.lines import Line2D
import segment
import fit
max_error=0.005

files=['CUBIC_SEQ.tr']
dic_output={}
for file in files:
    time_last=0.000001
    dic_output[file]=[]
    f=open('example_data/'+file)
    datas=f.readlines()
    f_output=file+' '+'seq'
    for line in datas:
        if 'cwnd' not in line:
            if '+' in line:
                line=line.strip().split(' ')
                if line[2]+' '+line[3]+' '+line[-5] =='0 2 1':
                    time=float(line[1])
                    seq=int(float(line[-2]))
                    '''
                    if time_last==time:
                        time_output=time+0.0000001
                    else:
                        time_output=time
                    '''
                    time_last=time
                    dic_output[file].append([time,seq])

for key in dic_output:
    output=np.asarray(dic_output[key])
    file_out='data/train/'+key+'_seq.txt'
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=6)   #设精度
    np.savetxt(file_out,output,fmt='%.07f')
    
        
       







    


