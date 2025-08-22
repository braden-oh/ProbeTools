import os
import h5py
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import pymc as pm
from scipy.special import erf
import pytensor.tensor as pt

"""
Useful functions in this directory:

extract_MPV(filepath, lowerVoltage=0, plot=False)
- filepath: string containing a path to an .hdf5 file
- lowerVoltage: integer minimum voltage to consider; Gaussian is fit only to data above this voltage
- plot: boolean indicating whether to display plots of raw data and derivative/Gaussian fit with outliers marked

process_RPA_directory(folder, lowerVoltage=0, plot=False)
- folder: string containing a path to a directory containing .hdf5 files with "RPA" in the name
- lowerVoltage: integer minimum voltage to pass through to extract_MPV
- plot: boolean flag to pass through to extract_MPV

"""


def gauss(x, amp, center, width):
    return amp * np.exp(-((x - center) ** 2) / (2 * width ** 2))

def residual_outliers(y, window_length=31, polyorder=2, threshold=3.0):
    """
    Flags outliers based on deviation from a Savitzky-Golay smoothed curve.

    Parameters:
    - y: 1D array of raw signal data
    - window_length: size of the S-G smoothing window (odd integer)
    - polyorder: degree of polynomial to use in smoothing
    - threshold: number of standard deviations to use for cutoff

    Returns:
    - mask: boolean array where True = outlier
    """
    if window_length >= len(y):
        window_length = len(y) - 1 if len(y) % 2 == 0 else len(y)
    if window_length % 2 == 0:
        window_length += 1  # must be odd

    smoothed = savgol_filter(y, window_length=window_length, polyorder=polyorder, mode='interp')
    residuals = y - smoothed
    sigma = np.std(residuals)

    return np.abs(residuals) < threshold * sigma

def movmedian_outliers(y, window_size=25, threshold=4):
    """
    Replicates MATLAB's isoutlier(y, 'movmedian', window_size).
    Flags values that deviate more than `threshold` * MAD from local median.

    Parameters:
    - y: input 1D array
    - window_size: size of the moving window (should be odd)
    - threshold: multiple of MAD to use as outlier criterion

    Returns:
    - mask: boolean array where True indicates an outlier
    """
    if window_size % 2 == 0:
        window_size += 1  # ensure window is odd

    median_y = median_filter(y, size=window_size, mode='nearest')
    deviation = np.abs(y - median_y)
    mad = np.median(deviation)

    # Scale MAD to be consistent with standard deviation for normal data
    scaled_mad = 1.4826 * mad

    return deviation < threshold * scaled_mad

# Define the Gaussian function (with baseline, c)
def gaussian(x, a, x0, sigma, c):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + c

def build_savename(filepath):
    file = filepath.split('/')[-1]
    _, bias = RPA_extract_metadata(filepath)
    angle = file.split('_')[1]
    if file.split('_')[-1][1] == '.':
        savename = 'RPA ' + angle + ' ' + str(bias) + 'V ' + file.split('_')[-1][0] + ' '
    else:
        savename = 'RPA ' + angle + ' ' + str(bias) + 'V '
    return savename


#==============================================================================================================
#==============================================================================================================

def regression_gaussfit(x, y, bias, savepath=None, center_guess=None):
    # Initial guesses: [amplitude, center, width, baseline]
    amp_guess = np.max(y)  # negative peak expected
    if center_guess is None:
        center_guess = np.max(x) * 0.8
        if center_guess > bias:
            center_guess = bias-1
    sigma_guess = (x[-1] - x[0]) / 10
    baseline_guess = np.median(y)
    p0 = [amp_guess, center_guess, sigma_guess, baseline_guess]
    # Perform the curve fit and covariance computation
    try:
        # [amplitude, center, width, baseline]
        lower_bounds = [0, -np.inf, 0, -np.inf]
        upper_bounds = [np.inf, bias, np.inf, np.inf]
        # Combine into a tuple for the 'bounds' argument
        param_bounds = (lower_bounds, upper_bounds)
        p_opt, p_cov = curve_fit(gaussian, x, y, p0=p0, bounds=param_bounds)
        amp, x0, sig, base = p_opt
        x0_err = np.sqrt(p_cov[1,1])
        x0_lower = x0-x0_err
        x0_upper = x0+x0_err
    except:
        print('No fit found.  Zeroing.')
        return 0, 0, 0

    # Plot the fit and save to external directory ---------------------------
    if savepath is not None:
        xfit = np.linspace(x.min(), x.max(), 500)
        yfit = gaussian(xfit, amp, x0, sig, base)
        plt.figure()
        plt.plot(x, y, 'k.', label='data')
        plt.plot(xfit, yfit, 'r-', label='Posterior mean fit')
        plt.axvline(x0, color='k', linestyle='--', label=f'{x0:.2f} V (95% CI: {x0_lower:.2f} to {x0_upper:.2f} V)')
        plt.axvspan(x0_lower, x0_upper, color='gray', alpha=0.25, label='Covariance uncertainty band')
        plt.xlabel('Voltage [V]')
        plt.ylabel('-dI/dV')
        plt.legend()
        plt.savefig(savepath, dpi=300)
        plt.close()

    return x0, x0-x0_err, x0+x0_err

def bayesian_gaussfit_mcmc(x, y, bias, savepath=None):
    # 1. Scale x and y for numerically stable inference ---------------------------
    xmean, xstd = np.mean(x), np.std(x)
    ymean, ystd = np.mean(y), np.std(y) if np.std(y) != 0 else 1.0

    x_scaled = (x - xmean) / xstd
    bias_scaled = (bias - xmean) / xstd
    y_scaled = (y - ymean) / ystd
    if bias_scaled <= min(x_scaled):
        print(f'bias value {bias} must be greater than minimum x value [{min(x)}.  Returning zeros.')
        return 0, 0, 0

    # 2. Estimate noise for likelihood term ---------------------------------------
    # If unknown, just use empirical std:
    yerr = np.std(y_scaled) if np.std(y_scaled) != 0 else 1.0

    # 3. PyMC Bayesian Gaussian Fit -----------------------------------------------
    with pm.Model() as model:
        # Scaled priors
        a = pm.HalfNormal('a', sigma=np.max(y_scaled) or 1.0)
        x0 = pm.Uniform('x0', lower=min(x_scaled), upper=bias_scaled)
        sigma = pm.HalfNormal('sigma', sigma=1.0)
        c = pm.Normal('c', mu=np.median(y_scaled), sigma=2*np.abs(np.median(y_scaled)) + 1e-12)

        # Gaussian model
        mu = a * pm.math.exp(-(x_scaled - x0) ** 2 / (2 * sigma ** 2)) + c

        # Likelihood
        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=yerr, observed=y_scaled)
        
        # MCMC sampling
        trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.99, progressbar=True, random_seed=42)

    # 4. Extract mean and credible intervals (unscale!) ---------------------------
    a_mean = np.mean(trace.posterior['a'].values)
    x0_mean_scaled = np.mean(trace.posterior['x0'].values)
    sigma_mean_scaled = np.mean(trace.posterior['sigma'].values)
    c_mean = np.mean(trace.posterior['c'].values)

    # For 95% credible interval
    x0_samples_scaled = trace.posterior['x0'].values.flatten()
    x0_lower_scaled, x0_upper_scaled = np.percentile(x0_samples_scaled, [2.5, 97.5])

    # Unscale back to your input units:
    x0_mean = x0_mean_scaled * xstd + xmean
    x0_lower = x0_lower_scaled * xstd + xmean
    x0_upper = x0_upper_scaled * xstd + xmean

    # 5. Plot the fit and save to external directory ---------------------------
    if savepath is not None:
        xfit = np.linspace(x.min(), x.max(), 500)
        yfit = gaussian(xfit, a_mean * ystd, x0_mean, sigma_mean_scaled * xstd, c_mean * ystd + ymean)
        plt.figure()
        plt.plot(x, y, 'k.', label='data')
        plt.plot(xfit, yfit, 'r-', label='Posterior mean fit')
        plt.axvline(x0_mean, color='k', linestyle='--', label=f'{x0_mean:.2f} V (95% CI: {x0_lower:.2f} to {x0_upper:.2f} V)')
        plt.axvspan(x0_lower, x0_upper, color='gray', alpha=0.25, label='95% credible interval')
        plt.xlabel('Voltage [V]')
        plt.ylabel('-dI/dV')
        plt.legend()
        plt.savefig(savepath, dpi=300)
        plt.close()

    return x0_mean, x0_lower, x0_upper
    

def clean_RPA_data(V_raw, I_raw):
    # --- Begin MAD outlier filter ---
    window = 21  # odd, tune as needed
    halfwin = window // 2

    def clean(arr):
        outlier_indices = []
        for i in range(len(arr)):
            start = max(0, i - halfwin)
            end = min(len(arr), i + halfwin + 1)
            local_window = arr[start:end]
            local_median = np.median(local_window)
            local_mad = np.median(np.abs(local_window - local_median))
            if local_mad == 0:
                continue
            if np.abs(arr[i] - local_median) > 5 * local_mad:
                outlier_indices.append(i)
        outlier_indices = np.array(outlier_indices)
        # Return the keep_indicies array
        return np.setdiff1d(np.arange(len(arr)), outlier_indices)
    
    # Clean the data over the y-axis data
    keep_indices = clean(I_raw)
    V_clean = V_raw[keep_indices]
    I_clean = I_raw[keep_indices]
    # Clean the data over the x-axis data
    keep_indices = clean(V_clean)
    V_clean = V_clean[keep_indices]
    I_clean = I_clean[keep_indices]
    # --- End MAD outlier filter ---
    # Remove *duplicate* voltages (keeping the first instance)
    _, unique_indices = np.unique(V_clean, return_index=True)
    V = V_clean[sorted(unique_indices)]
    I = I_clean[sorted(unique_indices)]
    return V, I


def extract_MPV_dIdV_fit(filepath, fit_lower=True, bayesian=True):
    file = filepath.split('/')[-1]
    savepath = filepath[:-len(file)] + 'Figures/RPA'
    _, bias = RPA_extract_metadata(filepath)
    savename = os.path.join(savepath, file[:-5])
    # Load HDF5 data
    with h5py.File(filepath, 'r') as f:
        data = f['/Data'][:]
    # Extract raw data
    V_raw = data[:,1]
    I_raw = data[:,2]
    # Clean the data
    V, I = clean_RPA_data(V_raw, I_raw)
    # Smooth the data before computing the gradient
    I_smooth = savgol_filter(I, window_length=11, polyorder=5)
    # Compute the derivative at each point via gradient (rather than diff)
    dIdV = -np.gradient(I_smooth, V)
    # (Try to) clean outliers out of the derivative data (needs better tuning)
    V, dIdV = clean_RPA_data(V, dIdV)
    # --- Separate by high energy population and low energy population WITH APPLICABLE LIMITS ---
    thresh = 40
    #thresh = 30
    # For the low range, let's simply fit from 3V up to the threshold voltage
    V_low = V[3:thresh]
    dIdV_low = dIdV[3:thresh]
    # For the high range, let's fit over a window
    if bias - 90 >= thresh: start = bias - 90
    else: start = thresh
    stop = bias + 10
    V_high = V[start:stop]
    dIdV_high = dIdV[start:stop]
    # --- Perform the high energy fit --- 
    if bias > thresh:   # Only do a high energy fit if we have a high energy plate
        if bayesian: mpv, lower, upper = bayesian_gaussfit_mcmc(V_high, dIdV_high, bias, savename+' high.png')
        else: mpv, lower, upper = regression_gaussfit(V_high, dIdV_high, bias, savename+' high reg.png')
        idx = np.abs(V - mpv).argmin()    # Find the MPV index in the full cleaned dataset
        mpv_cur = I_smooth[idx]      # Pull the corresponding smoothed current value
        data = [mpv, lower, upper, mpv_cur]
    else:
        data = [0, 0, 0, 0]
    # --- Perform the low energy fit ---
    if fit_lower:
        if bayesian: mpv, lower, upper = bayesian_gaussfit_mcmc(V_low, dIdV_low, bias, savename+' low.png')
        else: mpv, lower, upper = regression_gaussfit(V_low, dIdV_low, bias, savename+' low reg.png', center_guess=15)
        data = data + [mpv, lower, upper]
    else:
        data = data + [0, 0, 0]
    
    return data


def extract_MPV(filepath, savepath=None, bias=None, maxbias=None):
    # Short circuit out if we're at a zero bias condition
    # (resolve this later by allowing a single erf fit)
    if bias == 0:
        print('Ignored file.  Continuing...')
        return list(np.zeros(12))

    with h5py.File(filepath, 'r') as f:
        data = f['/Data'][:]
    # Extract raw data
    V_raw = data[:,1]
    I_raw = data[:,2]
    # Cut out data above a max threshold
    if maxbias:
        mask = V_raw < maxbias
        V_raw = V_raw[mask]
        I_raw = I_raw[mask]
    # --- 1. Clean the data ---
    x, y = clean_RPA_data(V_raw, I_raw)

    # --- 2. Normalize the data ---
    y_min = np.min(y)
    y_max = np.max(y)
    y_norm = (y - y_min) / (y_max - y_min)

    # --- 3. Define the double error function we'll fit to the data ---
    def double_error_func_tensor(x, A1, x01, sigma1, A2, x02, sigma2, C):
        term1 = A1/2 * (1 - pt.erf((x - x01)/(np.sqrt(2)*sigma1)))
        term2 = A2/2 * (1 - pt.erf((x - x02)/(np.sqrt(2)*sigma2)))
        return term1 + term2 + C

    def double_error_func(x, A1, x01, sigma1, A2, x02, sigma2, C):
        term1 = A1/2 * (1 - erf((x - x01)/(np.sqrt(2)*sigma1)))
        term2 = A2/2 * (1 - erf((x - x02)/(np.sqrt(2)*sigma2)))
        return term1 + term2 + C

    # --- 4. Set up the priors ---
    with pm.Model() as model:
        # Low energy priors
        A1 = pm.HalfNormal("A1", sigma=1)
        x01 = pm.Uniform("x01", lower=min(x), upper=50)
        sigma1 = pm.HalfNormal("sigma1", sigma=20)
        # High energy priors
        A2 = pm.HalfNormal("A2", sigma=1)
        if bias is not None:  x02 = pm.Uniform("x02", lower=30, upper=float(bias)) # Center cannot be higher than bias
        else: x02 = pm.Uniform("x02", lower=30, upper=max(x))   # If bias is not given, search the whole range
        #x02 = pm.Uniform("x02", lower=30, upper=biasfloat)
        sigma2 = pm.HalfNormal("sigma2", sigma=20)
        # Vertical offset parameter
        C = pm.Normal("C", mu=0, sigma=0.1)
        
        mu = double_error_func_tensor(x, A1, x01, sigma1, A2, x02, sigma2, C)
        
        sigma_noise = pm.HalfNormal("sigma_noise", sigma=0.05)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma_noise, observed=y_norm)
        
        # --- 5. Run the MCMC ---
        #print(model.initial_point())
        trace = pm.sample(2000, tune=1000, target_accept=0.95, return_inferencedata=True)

    # --- 6. Unscale the model ---
    # Use median or mean posterior values for prediction
    A1_m = trace.posterior['A1'].mean().item()
    x01_m = trace.posterior['x01'].mean().item()
    sigma1_m = trace.posterior['sigma1'].mean().item()
    A2_m = trace.posterior['A2'].mean().item()
    x02_m = trace.posterior['x02'].mean().item()
    sigma2_m = trace.posterior['sigma2'].mean().item()
    C_m = trace.posterior['C'].mean().item()
    # Model in normalized space:
    model_fit_norm = double_error_func(x, A1_m, x01_m, sigma1_m, A2_m, x02_m, sigma2_m, C_m)
    # Unscale to original units:
    model_fit = model_fit_norm * (y_max - y_min) + y_min

    # --- 7. Plot for comparison ---
    if savepath is not None:
        plt.figure()
        plt.plot(x, y, 'b.', label='Raw Data')
        plt.plot(x, model_fit, label='Posterior Fit', linewidth=1)
        plt.xlabel('Voltage [V]')
        plt.ylabel('Current Current [A]')
        plt.legend()
        plt.savefig(savepath, dpi=300)
        plt.close()

    # --- 8. Do statistics and retrieve current values ---
    x01_samples = trace.posterior["x01"].values.flatten()
    x01_l, x01_u = np.percentile(x01_samples,[2.5,97.5])    # Compute 95% confidence interval bounds

    x01_m_cur = y[np.abs(x - x01_m).argmin()]    # Mean current
    x01_l_cur = y[np.abs(x - x01_u).argmin()]    # Lower current bound (from upper MPV)
    x01_u_cur = y[np.abs(x - x01_l).argmin()]    # Upper current bound (from lower MPV)

    x02_samples = trace.posterior["x02"].values.flatten()
    x02_l, x02_u = np.percentile(x02_samples,[2.5,97.5])    # Compute 95% confidence interval bounds

    #x02_m_cur = y[np.abs(x - x02_m).argmin()]    # Mean current
    idx = np.abs(x - x02_m).argmin()    # Index of mean current
    x02_m_cur = np.mean(y[idx-5:idx+5])
    #x02_l_cur = y[np.abs(x - x02_u).argmin()]    # Lower current bound (from upper MPV)
    idx = np.abs(x - x02_u).argmin()    # Index of mean current
    x02_l_cur = np.mean(y[idx-5:idx+5])
    #x02_u_cur = y[np.abs(x - x02_l).argmin()]    # Upper current bound (from lower MPV)
    idx = np.abs(x - x02_l).argmin()    # Index of mean current
    x02_u_cur = np.mean(y[idx-5:idx+5])

    print(f"V01 (low energy center): mean={x01_m:.2f} V, 95% CI=({x01_l}, {x01_u})")
    print(f"V02 (high energy center): mean={x02_m:.2f} V, 95% CI=({x02_l}, {x02_u})")

    return [x01_m, x01_l, x01_u, x01_m_cur, x01_l_cur, x01_u_cur, 
            x02_m, x02_l, x02_u, x02_m_cur, x02_l_cur, x02_u_cur]


def RPA_extract_metadata(file):
    parts = file.split('_')
    bias = int(parts[2].split('V')[0])
    
    split_angle = parts[1][:-3].split('-')
    if len(split_angle) == 1:
        angle = int(split_angle[0])
    elif len(split_angle) == 2:
        angle = int(split_angle[0]) + int(split_angle[1])/10**(len(split_angle[1]))
    else:
        Exception('Unexpected format for angle in file ' + file)
    return angle, bias


# === Batch processing ===
def RPA_construct_database(dir_path, fitall=True, maxbias=None):

    RPA_df = pd.DataFrame(columns=['Filename', 'Angle', 'Bias', 
                                   'Low MPV', 'Low MPV Lower', 'Low MPV Upper', 
                                   'Low MPV Current', 'Low MPV Current Lower', 'Low MPV Current Upper',
                                   'High MPV', 'High MPV Lower', 'High MPV Upper', 
                                   'High MPV Current', 'High MPV Current Lower', 'High MPV Current Upper',])

    print('-----')
    print('Constructing database...')

    if not fitall:
        try:
            RPA_df = pd.read_csv(os.path.join(dir_path, 'RPA_MPVs.csv'))
            print('Found existing RPA_MPVs file.')
        except:
            print('No RPA database file found.')

    # Obtain a list of all .txt file names 
    files = os.listdir(dir_path)
    files = [f for f in files if f.endswith('.hdf5') and 'RPA' in f]
    files.sort()

    # Identify the figure save directory
    figpath = os.path.join(dir_path, 'Figures/RPA')
    os.makedirs(figpath, exist_ok=True)

    for file in files:
        file = file.split('/')[-1]
        print('Processing ' + file)
        full_path = os.path.join(dir_path, file)
        angle, bias = RPA_extract_metadata(full_path)
        savepath = os.path.join(figpath, file.split('.')[0]+'.png')

        if not fitall and RPA_df['Filename'].str.contains(file).any():
            print('Already processed ' + file + '. Skipping.')
        else:
            # Obtain MPVs & currents for the upper and lower ranges (with plots being generated)
            RPA_df.loc[len(RPA_df)] = [file, angle, bias] + extract_MPV(full_path, savepath, bias, maxbias)

    RPA_df.to_csv(os.path.join(dir_path, "RPA_MPVs.csv"), index=False)
    print('Saved RPA database to ' + os.path.join(dir_path, "RPA_MPVs.csv"))

# Example usage:
if __name__ == "__main__":
    """
    folder = '/Users/braden/Documents/Beam Catcher/2025 Spring Test Campaign/2025-07 Processing Folder/Varied Bias at 27 deg'
    #folder = '/Users/braden/Documents/Beam Catcher/2025 Spring Test Campaign/2025-05 Reprocessing Folder'
    #path = '/Users/braden/Documents/Beam Catcher/2025 Spring Test Campaign/2025-07 Processing Folder/Mobile RPA at 27 deg'
    RPA_construct_database(folder, fitall=False)
    #extract_MPV_bayesian(os.path.join(folder, 'RPA_24deg_150V_bias.hdf5'))

    #print(extract_MPV_bayesian(os.path.join(folder, 'FixedRPA_26-68deg_150V_1.hdf5')))

    #path = '/Users/braden/Documents/Beam Catcher/2025 Spring Test Campaign/2025-07 Processing Folder/Varied Bias at 27 deg'
    #os.chdir(path)
    #os.getcwd()
    #print(extract_MPV('FixedRPA_26-68deg_150V_1.hdf5'))

    """
    path = '/Users/braden/Documents/Beam Catcher/2025 Spring Test Campaign/2025-07 Processing Folder/Varied Bias at 27 deg'
    os.chdir(path)
    print(os.getcwd())
    RPA_construct_database(path, fitall=False, maxbias=150)
    
