
def slidingwindowsegment(sequence, create_segment, compute_error, max_error, seq_range=None):
    """
    Return a list of line segments that approximate the sequence.

    The list is computed using the sliding window technique. 

    Parameters
    ----------
    sequence : sequence to segment
    create_segment : a function of two arguments (sequence, sequence range) that returns a line segment that approximates the sequence data in the specified range
    compute_error: a function of two argments (sequence, segment) that returns the error from fitting the specified line segment to the sequence data
    max_error: the maximum allowable line segment fitting error

    """
    if not seq_range:
        seq_range = (0,len(sequence)-1)

    start = seq_range[0]
    end = start
    result_segment = create_segment(sequence,(seq_range[0],seq_range[1]))
    while end < seq_range[1]:
        end += 1
        test_segment = create_segment(sequence,(start,end))
        error = compute_error(sequence,test_segment)
        if error <= max_error:
            result_segment = test_segment
        else:
            break

    if end == seq_range[1]:
        return [result_segment]
    else:
        return [result_segment] + slidingwindowsegment(sequence, create_segment, compute_error, max_error, (end-1,seq_range[1]))
        
def bottomupsegment(sequence, create_segment, compute_error, max_error):
    """
    Return a list of line segments that approximate the sequence.
    
    The list is computed using the bottom-up technique.
    
    Parameters
    ----------
    sequence : sequence to segment
    create_segment : a function of two arguments (sequence, sequence range) that returns a line segment that approximates the sequence data in the specified range
    compute_error: a function of two argments (sequence, segment) that returns the error from fitting the specified line segment to the sequence data
    max_error: the maximum allowable line segment fitting error
    
    """
    segments = [create_segment(sequence,seq_range) for seq_range in zip(range(len(sequence))[:-1],range(len(sequence))[1:])]
    mergesegments = [create_segment(sequence,(seg1[0],seg2[2])) for seg1,seg2 in zip(segments[:-1],segments[1:])]
    mergecosts = [compute_error(sequence,segment) for segment in mergesegments]

    while min(mergecosts) < max_error:
        idx = mergecosts.index(min(mergecosts))
        segments[idx] = mergesegments[idx]
        del segments[idx+1]

        if idx > 0:
            mergesegments[idx-1] = create_segment(sequence,(segments[idx-1][0],segments[idx][2]))
            mergecosts[idx-1] = compute_error(sequence,mergesegments[idx-1])

        if idx+1 < len(mergecosts):
            mergesegments[idx+1] = create_segment(sequence,(segments[idx][0],segments[idx+1][2]))
            mergecosts[idx+1] = compute_error(sequence,mergesegments[idx])

        del mergesegments[idx]
        del mergecosts[idx]

    return segments
    
def topdownsegment(sequence, create_segment, compute_error, max_error, speeds, seq_range=None):
    """
    Return a list of line segments that approximate the sequence.
    
    The list is computed using the bottom-up technique.
    
    Parameters
    ----------
    sequence : sequence to segment
    create_segment : a function of two arguments (sequence, sequence range) that returns a line segment that approximates the sequence data in the specified range
    compute_error: a function of two argments (sequence, segment) that returns the error from fitting the specified line segment to the sequence data
    max_error: the maximum allowable line segment fitting error
    
    """
    if not seq_range:
        seq_range = (0,len(sequence)-1)

    bestlefterror,bestleftsegment = float('inf'), None
    bestrighterror,bestrightsegment = float('inf'), None
    bestidx = None
    last_speeds=None
    '''
    if seq_range[0]+1==seq_range[1]:
        bestidx=0
    '''
    for idx in range(seq_range[0]+1,seq_range[1]):
        segment_left,p_left,range_xy = create_segment(sequence,(seq_range[0],idx))   #直接输出斜率
        error_left = compute_error(sequence,segment_left,range_xy)
        segment_right,p_right,range_xy = create_segment(sequence,(idx,seq_range[1]))
        error_right = compute_error(sequence, segment_right,range_xy)
        if error_left + error_right < bestlefterror + bestrighterror:
            bestlefterror, bestrighterror = error_left, error_right
            bestleftsegment, bestrightsegment = segment_left, segment_right
            bestidx = idx
            best_left_speed=p_left
            best_ringt_speed=p_right
            
    
    if bestlefterror <= max_error:
        leftsegs = [bestleftsegment]
        speeds.append([best_left_speed])
    else:
        leftsegs,speeds_out = topdownsegment(sequence, create_segment, compute_error, max_error, seq_range=(seq_range[0],bestidx),speeds=speeds)
        speeds=speeds_out

    if bestrighterror <= max_error:
        rightsegs = [bestrightsegment]
        speeds.append([best_ringt_speed])
    else:
        rightsegs,speeds_out = topdownsegment(sequence, create_segment, compute_error, max_error, seq_range=(bestidx,seq_range[1]),speeds=speeds)
        speeds=speeds_out    
        #sequence, create_segment, compute_error, max_error, speeds, seq_range=None  
    
    a=leftsegs+rightsegs
    #print(a)    
    return leftsegs+rightsegs,speeds

