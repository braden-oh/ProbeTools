
import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pymc as pm
from scipy.special import erf
import pytensor.tensor as pt

filepath = '/Users/braden/Documents/ProbeTools/Testbed/Refactored RPA'

os.chdir(filepath)
os.getcwd()

def extract_metadata(f):
    parts = f.split('_')
    prop = parts[0]
    sccm = float(parts[1].split('sccm')[0])
    current = float(parts[2].split('A')[0])
    # Extract the type of probe
    probe = parts[3].split('.csv')[0]
    if 'RPA' in probe: probe = 'RPA'
    elif 'LP' in probe: probe = 'Langmuir'
    else: raise ValueError('Unrecognized probe type in file name ' + f)
    # Return the metadata
    return [f, prop, sccm, current, probe]

def parse_directory(path=None, savelog=True):
    # --- Navigate to the target directory ---
    if path: os.chdir(path)
    print("Parsing '" + os.getcwd() + "'")
    # -- Create the results directory if it doesn't already exist ---
    savedir = os.path.join(os.getcwd(), 'Results')
    os.makedirs(savedir, exist_ok=True)
    # --- Parse any csv file in the target directory ---
    csvs = [f for f in os.listdir() if '.csv' in f]
    metadata_table = []
    for file in csvs:
        metadata_table.append(extract_metadata(file))
    print('Parsed %i files.' % len(csvs))
    # --- Package the metadata into a pandas dataframe ---
    cols = ['Filename', 'Propellant', 'Flow [sccm]', 'Discharge Current [A]', 'Probe']
    df = pd.DataFrame(metadata_table, columns=cols)
    # --- Save the metadata to the results directory ---
    if savelog:
        df.to_csv(os.path.join(savedir, 'File Metadata.csv'))
        print("Saved metadata table to '/Results/File Metadata.csv'")
    return df
        
df = parse_directory()



def MAD_outlier_filter(V_raw, I_raw, window=21):
    # Note: Window must be an odd number.
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
    
    # --- Clean the data over the y-axis data ---
    keep_indices = clean(I_raw)
    V_clean = V_raw[keep_indices]
    I_clean = I_raw[keep_indices]
    # --- Clean the data over the x-axis data ---
    keep_indices = clean(V_clean)
    V_clean = V_clean[keep_indices]
    I_clean = I_clean[keep_indices]
    # --- Remove *duplicate* voltages (keeping the first instance) ---
    _, unique_indices = np.unique(V_clean, return_index=True)
    V = V_clean[sorted(unique_indices)]
    I = I_clean[sorted(unique_indices)]
    return V, I

def fit_dual_erf(x, y, savepath=None, bias=None, max_x=None):
    # --- 1. Cut out data above a max threshold ---
    if max_x:
        mask = x < max_x
        x = x[mask]
        y = y[mask]
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

def fit_erf(x, y, savepath=None, bias=None, max_x=None):
    # --- 1. Cut out data above a max threshold ---
    if max_x:
        mask = x < max_x
        x = x[mask]
        y = y[mask]
    # --- 2. Normalize the data ---
    y_min = np.min(y)
    y_max = np.max(y)
    y_norm = (y - y_min) / (y_max - y_min)

    # --- 3. Define the single error function we'll fit to the data ---
    def single_error_func_tensor(x, A, x0, sigma, C):
        return A/2 * (1 - pt.erf((x - x0)/(np.sqrt(2)*sigma))) + C

    def single_error_func(x, A, x0, sigma, C):
        return A/2 * (1 - erf((x - x0)/(np.sqrt(2)*sigma))) + C

    # --- 4. Set up the priors ---
    with pm.Model() as model:
        # Gaussian Priors
        A = pm.HalfNormal("A1", sigma=1)
        x0 = pm.Uniform("x01", lower=min(x), upper=50)
        sigma = pm.HalfNormal("sigma1", sigma=20)
        # Vertical offset parameter
        C = pm.Normal("C", mu=0, sigma=0.1)
        
        mu = single_error_func_tensor(x, A, x0, sigma, C)
        
        sigma_noise = pm.HalfNormal("sigma_noise", sigma=0.05)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma_noise, observed=y_norm)
        
        # --- 5. Run the MCMC ---
        trace = pm.sample(2000, tune=1000, target_accept=0.95, return_inferencedata=True)

    # --- 6. Unscale the model ---
    # Use median or mean posterior values for prediction
    A_m = trace.posterior['A1'].mean().item()
    x0_m = trace.posterior['x01'].mean().item()
    sigma_m = trace.posterior['sigma1'].mean().item()
    C_m = trace.posterior['C'].mean().item()
    # Model in normalized space:
    model_fit_norm = single_error_func(x, A_m, x0_m, sigma_m, C_m)
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
    x0_samples = trace.posterior["x0"].values.flatten()
    x0_l, x0_u = np.percentile(x0_samples,[2.5,97.5])    # Compute 95% confidence interval bounds

    x0_m_cur = y[np.abs(x - x0_m).argmin()]    # Mean current
    x0_l_cur = y[np.abs(x - x0_u).argmin()]   # Lower current bound (from upper MPV)
    x0_u_cur = y[np.abs(x - x0_l).argmin()]   # Upper current bound (from lower MPV)

    idx = np.abs(x - x0_m).argmin()    # Index of mean current
    x0_m_cur = np.mean(y[idx-5:idx+5])
    idx = np.abs(x - x0_u).argmin()    # Index of lower current bound (from upper MPV)
    x0_l_cur = np.mean(y[idx-5:idx+5])
    idx = np.abs(x - x0_l).argmin()    # Index of upper current bound (from lower MPV)
    x0_u_cur = np.mean(y[idx-5:idx+5])

    print(f"V01 (low energy center): mean={x0_m:.2f} V, 95% CI=({x0_l}, {x0_u})")

    fit_params = [A_m, x0_m, sigma_m, C_m]
    return [x0_m, x0_l, x0_u, x0_m_cur, x0_l_cur, x0_u_cur, fit_params]

row = df.iloc[2]
RPA_df = pd.read_csv(row['Filename'], delimiter='\t')
V = RPA_df['Bias Voltage (V)']
I = RPA_df['Probe Current (A)']
x, y = MAD_outlier_filter(V, I)