from wrappers import leastsquareslinefit,leastsquareslinefit_1

# compute_error functions

def sumsquared_error(sequence, segment,range_xy):
    """Return the sum of squared errors for a least squares line fit of one segment of a sequence"""
    x0,y0,x1,y1 = segment

    p, error = leastsquareslinefit(sequence,range_xy)
    return error
    
# create_segment functions

def regression(sequence, seq_range):
    """Return (x0,y0,x1,y1) of a line fit to a segment of a sequence using linear regression"""
    #print(seq_range)
    p, error = leastsquareslinefit(sequence,seq_range)   #p为斜率加bias
    x=sequence[:,0]
    index_left=seq_range[0]
    x_left=x[index_left]
    index_right=seq_range[1]
    x_right=x[index_right]
    y0 = p[0]*x_left + p[1]
    y1 = p[0]*x_right + p[1]
    return (x_left,y0,x_right,y1),p[0],(index_left,index_right)
    
def interpolate(sequence, seq_range):
    """Return (x0,y0,x1,y1) of a line fit to a segment using a simple interpolation"""
    return (seq_range[0], sequence[seq_range[0]], seq_range[1], sequence[seq_range[1]])
