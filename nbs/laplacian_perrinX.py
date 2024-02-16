import numpy as np
from scipy.special import legendre
from numpy.linalg import inv

# Define the laplacian_perrinX function in Python
def laplacian_perrinX(data, x, y, z, leg_order=None, smoothing=1e-5):
    """
    Compute surface Laplacian of EEG data using the Perrin et al. (1989) method.

    Parameters:
    data (ndarray): EEG data array where the first dimension corresponds to electrodes.
                    The data can be N-Dimensional.
    x (ndarray): 1D array of x coordinates of electrode positions.
    y (ndarray): 1D array of y coordinates of electrode positions.
    z (ndarray): 1D array of z coordinates of electrode positions.
    leg_order (int, optional): Order of Legendre polynomial. Default is 20 for <=100 electrodes
                               and 40 for >100 electrodes.
    smoothing (float, optional): Smoothing parameter (lambda) for G matrix. Default is 1e-5.

    Returns:
    surf_lap (ndarray): The surface Laplacian of the input EEG data.
    G (ndarray): G matrix used in the computation.
    H (ndarray): H matrix used in the computation.

    Notes:
    - The input coordinates (x, y, z) should be scaled to the unit sphere.
    - The EEG data should be organized with electrodes along the first dimension.
    - The smoothing parameter controls the flexibility of the spline interpolation.
    - The function automatically handles multiple trials and time points in the EEG data.

    Original MATLAB function by Mike X Cohen
    Python translation by AE Studio (and ChatGPT)
    """
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