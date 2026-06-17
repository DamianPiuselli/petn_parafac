"""
Baseline Chemometric Calibration and Alignment Algorithms for Chromatography.
Implements:
1. Correlation Optimized Warping (COW)
2. COW-PARAFAC
3. Multivariate Curve Resolution-Alternative Least Squares (MCR-ALS)
"""
import numpy as np
import scipy.optimize
import tensorly as tl
from tensorly.decomposition import non_negative_parafac

def pearson_correlation(x, y):
    """Calculates the Pearson correlation coefficient between two vectors."""
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_dev = x - x_mean
    y_dev = y - y_mean
    var_x = np.sum(x_dev**2)
    var_y = np.sum(y_dev**2)
    if var_x == 0 or var_y == 0:
        return 0.0
    return np.sum(x_dev * y_dev) / np.sqrt(var_x * var_y)

def cow_1d(T, P, N_seg=10, slack=3):
    """
    Finds the optimal warping path aligning profile P to reference T using dynamic programming.
    
    Args:
        T: Reference profile of length J (1D array)
        P: Sample profile to align of length J (1D array)
        N_seg: Number of segments
        slack: Maximum slack (warping tolerance)
        
    Returns:
        v_opt: List of optimal segment boundaries in the sample profile of length N_seg + 1
    """
    J = len(T)
    # Define reference segment boundaries
    u = [int(round(s * (J - 1) / N_seg)) for s in range(N_seg + 1)]
    L = [u[s] - u[s-1] for s in range(1, N_seg + 1)]
    
    # Define bounds on segment lengths in sample
    l_min = [max(1, L[s-1] - slack) for s in range(1, N_seg + 1)]
    l_max = [L[s-1] + slack for s in range(1, N_seg + 1)]
    
    # Forward bounds for reachable states
    min_fwd = [0] * (N_seg + 1)
    max_fwd = [0] * (N_seg + 1)
    for s in range(1, N_seg + 1):
        min_fwd[s] = min_fwd[s-1] + l_min[s-1]
        max_fwd[s] = max_fwd[s-1] + l_max[s-1]
        
    # Backward bounds for reachable states
    min_bwd = [0] * (N_seg + 1)
    max_bwd = [0] * (N_seg + 1)
    min_bwd[N_seg] = J - 1
    max_bwd[N_seg] = J - 1
    for s in range(N_seg - 1, -1, -1):
        min_bwd[s] = min_bwd[s+1] - l_max[s]
        max_bwd[s] = max_bwd[s+1] - l_min[s]
        
    # v_bounds[s] is (v_start, v_end) for the s-th boundary in the sample
    v_bounds = []
    for s in range(N_seg + 1):
        v_start = max(min_fwd[s], min_bwd[s])
        v_end = min(max_fwd[s], max_bwd[s])
        v_bounds.append((v_start, v_end))
        
    # Dynamic programming tables
    dp = np.full((N_seg + 1, J), -np.inf)
    parent = np.zeros((N_seg + 1, J), dtype=int)
    dp[0, 0] = 0.0
    
    for s in range(1, N_seg + 1):
        v_prev_start, v_prev_end = v_bounds[s-1]
        v_curr_start, v_curr_end = v_bounds[s]
        n_ref = u[s] - u[s-1] + 1
        T_seg = T[u[s-1] : u[s] + 1]
        
        for v in range(v_curr_start, v_curr_end + 1):
            for v_prev in range(v_prev_start, v_prev_end + 1):
                l = v - v_prev
                if l_min[s-1] <= l <= l_max[s-1]:
                    if dp[s-1, v_prev] > -np.inf:
                        P_seg = P[v_prev : v + 1]
                        xp = np.linspace(0, 1, len(P_seg))
                        x_new = np.linspace(0, 1, n_ref)
                        P_seg_interp = np.interp(x_new, xp, P_seg)
                        
                        val = pearson_correlation(P_seg_interp, T_seg)
                        score = dp[s-1, v_prev] + val
                        if score > dp[s, v]:
                            dp[s, v] = score
                            parent[s, v] = v_prev
                            
    v_opt = [0] * (N_seg + 1)
    v_opt[N_seg] = J - 1
    for s in range(N_seg, 0, -1):
        v_opt[s-1] = parent[s, v_opt[s]]
        
    return v_opt

def correlation_optimized_warping(X, N_seg=10, slack=3, ref_idx=None):
    """
    Applies Correlation Optimized Warping (COW) to align a 3D chromatographic tensor.
    Aligns the time dimension (axis 1) of each sample relative to a reference profile.
    
    Args:
        X: numpy array of shape (I, J, K) where I is samples, J is time points, K is spectral channels.
        N_seg: number of segments for dynamic programming.
        slack: maximum warping slack.
        ref_idx: index of the sample to use as reference. If None, selects the sample with the
                 highest average TIC correlation with all other samples.
                 
    Returns:
        X_aligned: aligned numpy array of shape (I, J, K)
        warping_paths: list of optimal warping boundaries (v_opt) for each sample.
    """
    I, J, K = X.shape
    tics = np.sum(X, axis=2) # Total Ion Chromatograms (I, J)
    
    if ref_idx is None:
        mean_corrs = []
        for i in range(I):
            corrs = []
            for j in range(I):
                if i != j:
                    corrs.append(pearson_correlation(tics[i], tics[j]))
            mean_corrs.append(np.mean(corrs))
        ref_idx = np.argmax(mean_corrs)
        
    T = tics[ref_idx]
    
    X_aligned = np.zeros_like(X)
    warping_paths = []
    u = [int(round(s * (J - 1) / N_seg)) for s in range(N_seg + 1)]
    
    for i in range(I):
        if i == ref_idx:
            X_aligned[i] = X[i]
            warping_paths.append(u)
            continue
            
        P = tics[i]
        v_opt = cow_1d(T, P, N_seg=N_seg, slack=slack)
        warping_paths.append(v_opt)
        
        # Build warping index mapping
        warped_indices = np.zeros(J)
        for s in range(1, N_seg + 1):
            u_start = u[s-1]
            u_end = u[s]
            v_start = v_opt[s-1]
            v_end = v_opt[s]
            t = np.arange(u_start, u_end + 1)
            if u_end > u_start:
                warped_indices[u_start:u_end + 1] = v_start + (t - u_start) / (u_end - u_start) * (v_end - v_start)
            else:
                warped_indices[u_start:u_end + 1] = v_start
                
        # Interpolate each spectral channel
        for k in range(K):
            X_aligned[i, :, k] = np.interp(warped_indices, np.arange(J), X[i, :, k])
            
    return X_aligned, warping_paths

class COWPARAFAC:
    """
    COW-PARAFAC Baseline.
    Pre-aligns raw chromatograms using COW and then runs standard non-negative PARAFAC.
    """
    def __init__(self, num_components=3, N_seg=10, slack=3):
        self.num_components = num_components
        self.N_seg = N_seg
        self.slack = slack
        
        self.A_ = None  # Scores (I, R)
        self.B_ = None  # Chromatographic profiles (J, R)
        self.C_ = None  # Spectral profiles (K, R)
        self.X_aligned_ = None
        
    def fit(self, X):
        """
        Fits COW-PARAFAC on the 3D chromatographic tensor X of shape (I, J, K).
        """
        X_aligned, _ = correlation_optimized_warping(X, N_seg=self.N_seg, slack=self.slack)
        self.X_aligned_ = X_aligned
        
        tensor = tl.tensor(X_aligned)
        weights, factors = non_negative_parafac(
            tensor,
            rank=self.num_components,
            init='random',
            n_iter_max=500,
            tol=1e-6,
            random_state=42
        )
        
        A = tl.to_numpy(factors[0])
        B = tl.to_numpy(factors[1])
        C = tl.to_numpy(factors[2])
        w = tl.to_numpy(weights)
        
        # Absorb weights into A
        A = A * w[np.newaxis, :]
        
        # Normalize B and C to unit L2 norm, transferring scale to A
        for r in range(self.num_components):
            norm_b = np.linalg.norm(B[:, r])
            norm_c = np.linalg.norm(C[:, r])
            if norm_b > 0:
                B[:, r] /= norm_b
                A[:, r] *= norm_b
            if norm_c > 0:
                C[:, r] /= norm_c
                A[:, r] *= norm_c
                
        self.A_ = A
        self.B_ = B
        self.C_ = C
        return self

class MCRALS:
    """
    Multivariate Curve Resolution-Alternative Least Squares (MCR-ALS) Baseline.
    Resolves the bilinear system X_concat = C * S^T with non-negativity constraints.
    Does not restrict sample chromatograms to be aligned.
    """
    def __init__(self, num_components=3, max_iter=100, tol=1e-5):
        self.num_components = num_components
        self.max_iter = max_iter
        self.tol = tol
        
        self.A_ = None  # Scores (I, R)
        self.B_ = None  # Chromatographic profiles (J, R) - canonical/average
        self.C_ = None  # Spectral profiles (K, R)
        self.B_samples_ = None  # Sample-specific profiles (I, J, R)
        
    def fit(self, X):
        """
        Fits MCR-ALS on the 3D chromatographic tensor X of shape (I, J, K).
        """
        I, J, K = X.shape
        R = self.num_components
        
        # Concatenate X vertically along the time dimension: shape (I * J, K)
        D = X.reshape(I * J, K)
        N = I * J
        
        # Initialize S (spectral profiles, K x R) using SVD
        U, s, Vt = np.linalg.svd(D, full_matrices=False)
        S = np.abs(Vt[:R].T)
        
        C = np.zeros((N, R))
        
        # Alternating Least Squares Loop
        prev_err = float('inf')
        for iteration in range(self.max_iter):
            # Update C
            for n in range(N):
                C[n, :], _ = scipy.optimize.nnls(S, D[n, :])
                
            # Update S
            for k in range(K):
                S[k, :], _ = scipy.optimize.nnls(C, D[:, k])
                
            # Check convergence
            reconstruction = C @ S.T
            err = np.linalg.norm(D - reconstruction) / np.linalg.norm(D)
            if abs(prev_err - err) < self.tol:
                break
            prev_err = err
            
        C_samples = C.reshape(I, J, R)
        
        # Decompose resolved concentrations and shapes
        A = np.zeros((I, R))
        B_samples = np.zeros((I, J, R))
        
        for i in range(I):
            for r in range(R):
                area = np.sum(C_samples[i, :, r])
                A[i, r] = area
                if area > 0:
                    B_samples[i, :, r] = C_samples[i, :, r] / area
                else:
                    B_samples[i, :, r] = C_samples[i, :, r]
                    
        # Average normalized profiles to get canonical chromatography profiles
        B = np.mean(B_samples, axis=0)
        
        C_spec = S.copy()
        
        # Scale resolution
        for r in range(R):
            norm_b = np.linalg.norm(B[:, r])
            norm_c = np.linalg.norm(C_spec[:, r])
            if norm_b > 0:
                B[:, r] /= norm_b
                A[:, r] *= norm_b
            if norm_c > 0:
                C_spec[:, r] /= norm_c
                A[:, r] *= norm_c
                
            if norm_b > 0:
                B_samples[:, :, r] /= norm_b
                
        self.A_ = A
        self.B_ = B
        self.C_ = C_spec
        self.B_samples_ = B_samples
        return self
