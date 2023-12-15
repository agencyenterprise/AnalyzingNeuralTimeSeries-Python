import numpy as np
from scipy.linalg import cholesky, inv

def armorf(x, Nr, Nl, p):
    """
    Estimate autoregressive (AR) model parameters using the Levinson-Wiggins-Robinson (LWR) algorithm.

    Parameters:
    x (ndarray): A 2D array where each row represents a variable's time series.
    Nr (int): The number of realizations (trials).
    Nl (int): The length of each realization (trial).
    p (int): The order of the AR model.

    Returns:
    coeff (ndarray): Coefficient matrix of the AR model.
    E (ndarray): Final prediction error covariance matrix (noise covariance of the AR model).
    kr (list): List of reflection coefficients (parcor coefficients).

    The function is adapted from the MATLAB function armorf, which is part of the BSMART toolbox.
    Original MATLAB function by Yonghong Chen
    Python translation by AE Studio (and ChatGPT)
    """
    L, N = x.shape
    R0 = np.zeros((L, L))
    pf = R0.copy()
    pb = R0.copy()
    pfb = R0.copy()
    ap = np.zeros((L, L, p+1))
    bp = np.zeros((L, L, p+1))
    En = R0.copy()
    
    for i in range(Nr):
        En += x[:, (i*Nl):(i*Nl+Nl)] @ x[:, (i*Nl):(i*Nl+Nl)].T
        ap[:, :, 0] += x[:, (i*Nl+1):(i*Nl+Nl)] @ x[:, (i*Nl+1):(i*Nl+Nl)].T
        bp[:, :, 0] += x[:, (i*Nl):(i*Nl+Nl-1)] @ x[:, (i*Nl):(i*Nl+Nl-1)].T
    
    ap[:, :, 0] = inv(cholesky(ap[:, :, 0] / Nr * (Nl-1)).T)
    bp[:, :, 0] = inv(cholesky(bp[:, :, 0] / Nr * (Nl-1)).T)
    
    for i in range(Nr):
        efp = ap[:, :, 0] @ x[:, (i*Nl+1):(i*Nl+Nl)]
        ebp = bp[:, :, 0] @ x[:, (i*Nl):(i*Nl+Nl-1)]
        pf += efp @ efp.T
        pb += ebp @ ebp.T
        pfb += efp @ ebp.T
    
    En = cholesky(En / N).T
    
    # Initial output variables
    coeff = []  # Coefficient matrices of the AR model
    kr = []  # Reflection coefficients
    
    for m in range(p):
        ck = inv(cholesky(pf).T) @ pfb @ inv(cholesky(pb))
        kr.append(ck)
        ef = np.eye(L) - ck @ ck.T
        eb = np.eye(L) - ck.T @ ck
        
        En = En @ cholesky(ef).T
        E = (ef + eb) / 2
        
        a = np.zeros((L, L, m+2))
        b = np.zeros((L, L, m+2))
        ap = np.concatenate((ap, np.zeros((L, L, 1))), axis=2)
        bp = np.concatenate((bp, np.zeros((L, L, 1))), axis=2)
        pf = np.zeros((L, L))
        pb = np.zeros((L, L))
        pfb = np.zeros((L, L))
        
        for i in range(m+2):
            a[:, :, i] = inv(cholesky(ef).T) @ (ap[:, :, i] - ck @ bp[:, :, m+1-i])
            b[:, :, i] = inv(cholesky(eb).T) @ (bp[:, :, i] - ck.T @ ap[:, :, m+1-i])
        
        for k in range(Nr):
            efp = np.zeros((L, Nl-m-2))
            ebp = np.zeros((L, Nl-m-2))
            for i in range(m+2):
                k1=m+2-i+k*Nl
                k2=Nl-i+k*Nl
                efp += a[:, :, i] @ x[:, k1:k2]
                ebp += b[:, :, m+1-i] @ x[:, k1-1:k2-1]
            pf += efp @ efp.T
            pb += ebp @ ebp.T
            pfb += efp @ ebp.T
        
        ap = a
        bp = b
    
    for j in range(p):
        coeff.append(inv(a[:, :, 0]) @ a[:, :, j+1])
    
    coeff = np.concatenate(coeff, axis=1)
    
    return -coeff, En @ En.T, kr