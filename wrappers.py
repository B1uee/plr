from numpy import arange, array, ones
from numpy.linalg import lstsq
import numpy as np


# p, error = leastsquareslinefit(sequence,(x0,x1))
def leastsquareslinefit(sequence,seq_range):
    """Return the parameters and error for a least squares line fit of one segment of a sequence"""
    index_left=seq_range[0]
    index_right=seq_range[1]+1
    seg_data=np.asarray(sequence[index_left:index_right])
    x=array(seg_data[:,0])
    y=array(seg_data[:,1])
    x=np.expand_dims(x,1)   #x增加一个维度并使得值为1，表示0次项的系数为1
    bias=ones((len(x),1),float)
    x=np.concatenate((x,bias),axis=1)
    #x = arange(seq_range[0],seq_range[1]+1)
    #y = array(sequence[seq_range[0]:seq_range[1]+1])
    #A = ones((len(x),2),float)     #修改这里，改成x值而不是用间隔相同的index
    #A[:,0] = x
    (p,residuals,rank,s) = lstsq(x,y)
    try:
        error = residuals[0]
        #print(error)
    except IndexError:
        error = 0.0
        #print('indexerror')
    return (p,error)

def leastsquareslinefit_1(sequence,seq_range):
    """Return the parameters and error for a least squares line fit of one segment of a sequence"""
    x = arange(seq_range[0],seq_range[1]+1)
    y = array(sequence[seq_range[0]:seq_range[1]+1])
    A = ones((len(x),2),float)
    A[:,0] = x
    (p,residuals,rank,s) = lstsq(A,y)
    try:
        error = residuals[0]
    except IndexError:
        error = 0.0
    return (p,error)
