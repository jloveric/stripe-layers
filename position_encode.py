
def position_encode(q, coord):
    """
    Args :
        Each node has a different coordinate
        q : Input features tensor [batch, channels, nodes] which
        is assumed to be in the range [-1, 1]
        coord : Input of coordinates, where the coordinate 
        is a tensor [batch, nodes]
    """

    qout = q.reshape(q.shape[0], q.shape[1], -1)
    coord_mod = coord.reshape(coord.shape[0], -1)
    for i in range(q.shape[1]):
        qout[:, i, :] = 0.5*(1+q[:, i, :])+coord_mod

    return qout.reshape(*q.shape)
