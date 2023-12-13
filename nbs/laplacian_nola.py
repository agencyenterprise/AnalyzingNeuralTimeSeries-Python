import numpy as np
from numpy.linalg import pinv

def laplacian_nola(x, y, z, data, smoothing=100):
    """
    Compute surface Laplacian of EEG data via New Orleans method.
    
    Parameters:
    x, y, z : x, y, z coordinates of electrode positions
    data : EEG data (can be N-D, but first dimension must be electrodes)
    smoothing : smoothing parameter (default: 100)
    
    Returns:
    surf_lap : the surface Laplacian (second spatial derivative)
    """
    n = len(x)
    
    # Budge zero values
    x[x == 0] = 0.001
    y[y == 0] = 0.001
    z[z == 0] = 0.001
    
    # Compute K
    k = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s = x[i] - x[j]
            t = y[i] - y[j]
            r = z[i] - z[j]
            str = s**2 + t**2 + r**2
            k[i, j] = ((str + smoothing)**2) * np.log(str + smoothing)
    k += k.T
    kinv = pinv(k)
    
    # Compute E and A
    e = np.vstack((np.ones(n), x, y, x**2, x*y, y**2, z, z*x, z*y, z**2)).T
    ke = kinv @ e
    a = e.T @ ke
    ainv = pinv(a)
    
    # Compute Laplacian over data
    orig_data_size = data.shape
    data = np.reshape(data, (orig_data_size[0], -1), 'F')
    surf_lap = np.zeros_like(data)
    
    for ti in range(data.shape[1]):
        kv = kinv @ data[:, ti]
        ev = e.T @ kv
        
        q = ainv @ ev

        eq = e @ q
        keq = kinv @ eq

        p = kv - keq
        
        # Compute Laplacian
        hxy = np.hypot(x, y)
        r = np.hypot(hxy, z)
        el = np.arctan2(z, hxy)
        az = np.arctan2(y, x)

        el = np.pi / 2 - el
        
        # Trig functions
        st = np.sin(el)
        ct = np.cos(el)
        sp = np.sin(az)
        cp = np.cos(az)
        
        uuxyz = (2 * q[3] + 2 * q[5] + 2 * q[9] - 
                 (2 * st * (q[1] * cp + q[2] * sp) / r + 
                  2 * q[6] * ct / r + 
                  6 * st**2 * (q[3] * cp**2 + q[5] * sp**2 + q[4] * sp * cp) + 
                  6 * st * ct * (q[7] * cp + q[8] * sp) + 
                  6 * q[9] * ct**2))
        
        ttcomp = np.zeros_like(st)
        rrcomp = np.zeros_like(st)
        
        for j in range(n):
            a = r[j] * (st * cp - np.sin(el[j]) * np.cos(az[j]))
            b = r[j] * (st * sp - np.sin(el[j]) * np.sin(az[j]))
            c = r[j] * (ct - np.cos(el[j]))
            
            str = a**2 + b**2 + c**2
            strw = str + smoothing**2
            
            comterm = 4 * str / strw - (str / strw)**2 + 2 * np.log(strw)
            comterm2 = 2 * (2 * str * np.log(strw) + (str**2) / strw)
            
            tcomp = 3 * comterm2 + 4 * str * comterm
            dr = 2 * (a * st * cp + b * st * sp + c * ct)
            
            rcomp = dr * comterm2 + 2 * r[j] * comterm2 / 2 + r[j] * (dr**2) * comterm
            ttcomp += p[j] * tcomp
            rrcomp += p[j] * rcomp / r[j]
        
        surf_lap[:, ti] = -(ttcomp + uuxyz - rrcomp)
    
    surf_lap = np.reshape(surf_lap, orig_data_size, 'F')
    return surf_lap