import numpy as np 

def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """

    def get_p2_norm(x_row,y_row):
        diff_row = np.abs(x_row - y_row)
        power_row = np.square(diff_row)
        sum_row = np.sum(power_row)
        return np.sqrt(sum_row)

    M = len(X)
    N = len(Y)

    result = []
    for j in range(M):
        for k in range(N):
            result.append(get_p2_norm(X[j,:], Y[k,:]))

    result = np.array(result)
    return np.reshape(result, (M,N))


def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    def get_p1_norm(x_row,y_row):
        diff_row = np.abs(x_row - y_row)
        sum_row = np.sum(diff_row)
        return sum_row

    M = len(X)
    N = len(Y)

    result = []
    for j in range(M):
        for k in range(N):
            result.append(get_p1_norm(X[j,:], Y[k,:]))

    result = np.array(result)
    return np.reshape(result, (M,N))

