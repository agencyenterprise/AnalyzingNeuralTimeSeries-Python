import numpy as np
from scipy.special import legendre
from numpy.linalg import inv

# Define the laplacian_perrinX function in Python
def laplacian_perrinX(data, x, y, z, leg_order=None, smoothing=1e-5):
    numelectrodes = len(x)
    
    # Set default parameters for +/- 100 electrodes
    if numelectrodes > 100:
        m = 3
        leg_order = 40 if leg_order is None else leg_order
    else:
        m = 4
        leg_order = 20 if leg_order is None else leg_order

    # Scale XYZ coordinates to unit sphere
    maxrad = np.max(np.sqrt(x**2 + y**2 + z**2))
    x = x / maxrad
    y = y / maxrad
    z = z / maxrad
    
    # Initialize G, H, and cosdist matrices
    G = np.zeros((numelectrodes, numelectrodes))
    H = np.zeros((numelectrodes, numelectrodes))
    cosdist = np.zeros((numelectrodes, numelectrodes))
    
    # Compute cosdist matrix
    for i in range(numelectrodes):
        for j in range(i + 1, numelectrodes):
            cosdist[i, j] = 1 - (((x[i] - x[j])**2 + (y[i] - y[j])**2 + (z[i] - z[j])**2) / 2)
    cosdist = cosdist + cosdist.T + np.eye(numelectrodes)
    
    # Compute Legendre polynomial
    legpoly = np.zeros((leg_order, numelectrodes, numelectrodes))
    for ni in range(leg_order):
        temp = legendre(ni+1)(cosdist)
        legpoly[ni, :, :] = temp
    
    # Precompute electrode-independent variables
    twoN1 = 2 * (np.arange(1, leg_order + 1)) + 1
    gdenom = (np.arange(1, leg_order + 1) * (np.arange(1, leg_order + 1) + 1))**m
    hdenom = (np.arange(1, leg_order + 1) * (np.arange(1, leg_order + 1) + 1))**(m - 1)
    
    # Compute G and H matrices
    for i in range(numelectrodes):
        for j in range(i, numelectrodes):
            g = np.sum(twoN1 * legpoly[:, i, j] / gdenom)
            h = -np.sum(twoN1 * legpoly[:, i, j] / hdenom)
            G[i, j] = g / (4 * np.pi)
            H[i, j] = -h / (4 * np.pi)
    
    # Mirror matrix
    G = G + G.T
    H = H + H.T
    
    # Correct for diagonal-double
    G = G - np.eye(numelectrodes) * G[0, 0] / 2
    H = H - np.eye(numelectrodes) * H[0, 0] / 2
    
    # Reshape data to electrodes X time/trials
    orig_data_size = data.shape
    data = np.reshape(data, (orig_data_size[0], -1), 'F')
    
    # Add smoothing constant to diagonal
    Gs = G + np.eye(numelectrodes) * smoothing
    
    # Compute C matrix
    Gsinv = inv(Gs)
    GsinvS = np.sum(Gsinv, axis=1)
    dataGs = data.T @ Gsinv
    C = dataGs - (np.sum(dataGs, axis=1) / np.sum(GsinvS))[:, None] * GsinvS
    
    # Compute surface Laplacian (and reshape to original data size)
    surf_lap = np.reshape((C @ H.T).T, orig_data_size, 'F')
    
    return surf_lap, G, H