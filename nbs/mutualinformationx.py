import numpy as np
from scipy.stats import iqr

def mutualinformationx(x, y, fd_bins=None, permtest=False):
    if len(x) != len(y):
        raise ValueError('X and Y must have equal length')
    
    # Vectorize in the case of matrices
    x = x.flatten()
    y = y.flatten()
    
    # Determine the optimal number of bins for each variable
    if fd_bins is None:
        n = len(x)
        maxmin_range_x = np.max(x) - np.min(x)
        fd_bins_x = np.ceil(maxmin_range_x / (2.0 * iqr(x) * n ** (-1/3)))
        
        n = len(y)
        maxmin_range_y = np.max(y) - np.min(y)
        fd_bins_y = np.ceil(maxmin_range_y / (2.0 * iqr(y) * n ** (-1/3)))
        
        # Use the average
        fd_bins = int(np.ceil((fd_bins_x + fd_bins_y) / 2))
    
    # Bin data
    edges_x = np.linspace(np.min(x), np.max(x), fd_bins + 1)
    hist_x, _ = np.histogram(x, edges_x)
    
    edges_y = np.linspace(np.min(y), np.max(y), fd_bins + 1)
    hist_y, _ = np.histogram(y, edges_y)
    
    # Compute entropies
    prob_x = hist_x / np.sum(hist_x)
    prob_y = hist_y / np.sum(hist_y)
    
    entropy_x = -np.sum(prob_x * np.log2(prob_x + np.finfo(float).eps))
    entropy_y = -np.sum(prob_y * np.log2(prob_y + np.finfo(float).eps))
    
    # Compute joint probabilities
    joint_hist, _, _ = np.histogram2d(x, y, [edges_x, edges_y])
    joint_prob = joint_hist / np.sum(joint_hist)
    
    entropy_joint = -np.sum(joint_prob * np.log2(joint_prob + np.finfo(float).eps))
    
    # Mutual information
    mi = entropy_x + entropy_y - entropy_joint
    
    # Optional permutation testing
    if permtest:
        n_permutes = 500
        perm_mi = np.zeros(n_permutes)
        
        for permi in range(n_permutes):
            # Shuffle bins
            binbreak = np.random.choice(np.round(n * 0.8).astype(int), 1)[0] + np.round(n * 0.1).astype(int)
            bins2_perm = np.roll(y, binbreak)
            
            # Compute joint probabilities for permuted data
            joint_hist_perm, _, _ = np.histogram2d(x, bins2_perm, [edges_x, edges_y])
            joint_prob_perm = joint_hist_perm / np.sum(joint_hist_perm)
            
            entropy_joint_perm = -np.sum(joint_prob_perm * np.log2(joint_prob_perm + np.finfo(float).eps))
            
            # Mutual information for permuted data
            perm_mi[permi] = entropy_x + entropy_y - entropy_joint_perm
        
        # Convert MI to Z-scores
        mi = (mi - np.mean(perm_mi)) / np.std(perm_mi)
    
    return mi, np.array([entropy_x, entropy_y, entropy_joint]), fd_bins