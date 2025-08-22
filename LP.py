
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import h5py
import pymc as pm


def LP_extract_metadata(file):
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


def process_LP(filepath, savepath=None, l_thresh=0.06, r_thresh=0.3, lead_trim=0):
    # 1. Initial data loading and splitting ---------------------------
    with h5py.File(filepath, 'r') as f:
        data = f['/Data'][:]
    # Extract raw data
    V_raw = data[:,2]
    I_raw = -data[:,1]
    # Obtain floating voltage
    i = np.abs(I_raw).argmin() + 1
    Vf = V_raw[i]
    # Extract the I-V trace above Vf and shift up by the ion saturation current
    i = i + lead_trim   # Shift the data we fit to if the user asks for it
    V = V_raw[i:]
    logI = np.log(I_raw[i:] - min(I_raw))    # Log of current data with ion saturation subtrated out
    # Extract the left side data for line fitting
    V_left = V[:int(len(V)*l_thresh)]
    logI_left = logI[:int(len(V)*l_thresh)]
    # Extract the right side data for line fitting
    V_right = V[-int(len(V) * r_thresh):]
    logI_right = logI[-int(len(V) * r_thresh):]

    # 2. PyMC Bayesian Line Fits ---------------------------
    with pm.Model() as model:
        # Left branch priors
        m1 = pm.Normal("m1", mu=0, sigma=2)
        b1 = pm.Normal("b1", mu=0, sigma=4)
        sigma1 = pm.HalfNormal("sigma1", 1)
        pm.Normal("left_obs", mu=m1*V_left + b1, sigma=sigma1, observed=logI_left)
        # Right branch priors
        m2 = pm.Normal("m2", mu=0, sigma=2)
        b2 = pm.Normal("b2", mu=0, sigma=4)
        sigma2 = pm.HalfNormal("sigma2", 1)
        pm.Normal("right_obs", mu=m2*V_right + b2, sigma=sigma2, observed=logI_right)
        # Run the MCMC
        trace = pm.sample(1500, tune=1000, target_accept=0.92, random_seed=42)

    # 3. Extract plasma potential and electron temperature ---------------------------
    m1_samps = trace.posterior['m1'].values.flatten()
    b1_samps = trace.posterior['b1'].values.flatten()
    m2_samps = trace.posterior['m2'].values.flatten()
    b2_samps = trace.posterior['b2'].values.flatten()
    Vplasma_samps = (b2_samps - b1_samps) / (m1_samps - m2_samps)
    Vp_mean = np.mean(Vplasma_samps)
    Vp_lower, Vp_upper = np.percentile(Vplasma_samps, [2.5, 97.5])  # 95% confidence interval
    # Te from left branch (classic log-linear LP slope):
    Te_samps = 1.0 / m1_samps
    Te_mean = np.mean(Te_samps)
    Te_lower, Te_upper = np.percentile(Te_samps, [16,84])  # 95% confidence interval
    # Print telemetry
    print(f"Vplasma: {Vp_mean:.2f} V [{Vp_lower:.2f}, {Vp_upper:.2f}] V")
    print(f"T_e: {Te_mean:.2f} eV [{Te_lower:.2f}, {Te_upper:.2f}] eV")
    
    # 4. Save a plot showing the fit --------------------------- 
    if savepath is not None:
        # Plot the raw data, labeling points used for fitting
        plt.plot(V, logI, 'k.', label='Data')
        plt.plot(V_left, logI_left, 'rx', label='Left fitted points')
        plt.plot(V_right, logI_right, 'gx', label='Right fitted points')
        # Plot the left side fit line
        Vleft = np.linspace(min(V_left), Vp_mean+2, 50)
        plt.plot(Vleft, m1_samps.mean()*Vleft + b1_samps.mean(), 'b-', lw=1, label='Ion-line fit')
        # Plot the right side fit line
        Vright = np.linspace(Vp_mean-2, max(V_right), 50)
        plt.plot(Vright, m2_samps.mean()*Vright + b2_samps.mean(), 'b-', lw=1, label='Electron-line fit')
        # Plot the intersection point (plasma potential)
        plt.axvline(Vp_mean, color='k', linestyle='--', label='Vplasma')
        plt.xlabel("Voltage [V]")
        plt.ylabel("log(Current)")
        plt.legend()
        plt.title(filepath.split('/')[-1])
        plt.tight_layout()
        plt.savefig(savepath, dpi=300)
        plt.close()

    return [Vf, Vp_mean, Vp_lower, Vp_upper, Te_mean, Te_lower, Te_upper]


def LP_construct_database(dir_path, fitall=True, lead_trim=0):
    LP_df = pd.DataFrame(columns=['Filename', 'Angle', 'Bias', 'V Float',
                                   'V Plasma', 'V Plasma Lower', 'V Plasma Upper', 
                                   'Te', 'Te Lower', 'Te Upper'])

    if not fitall:
        try:
            LP_df = pd.read_csv(os.path.join(dir_path, "LP_database.csv"))
            print('Found existing LP_database file.')
        except:
            print('No LP database file found.')

    # Obtain a list of all .txt file names 
    files = os.listdir(dir_path)
    files = [f for f in files if f.endswith('.hdf5') and 'LP' in f]
    files.sort()

    # Identify the figure save directory
    figpath = os.path.join(dir_path, 'Figures/LP')
    os.makedirs(figpath, exist_ok=True)

    for file in files:
        file = file.split('/')[-1]
        print('Processing ' + file)
        full_path = os.path.join(dir_path, file)
        angle, bias = LP_extract_metadata(full_path)
        savepath = os.path.join(figpath, file.split('.')[0]+'.png')

        if not fitall and LP_df['Filename'].str.contains(file).any():
            print('Already processed ' + file + '. Skipping.')
        else:
            # Obtain LP parameters for the upper and lower ranges (with plots being generated)
            LP_df.loc[len(LP_df)] = [file, angle, bias] + process_LP(full_path, savepath, lead_trim=lead_trim)

    LP_df.to_csv(os.path.join(dir_path, "LP_database.csv"), index=False)
    print('Saved LP database to ' + os.path.join(dir_path, "LP_database.csv"))


if __name__ == "__main__":
    #folder = '/Users/braden/Documents/Beam Catcher/2025 Spring Test Campaign/2025-07 Processing Folder/Varied Bias at 27 deg'
    path = '/Users/braden/Documents/Beam Catcher/2025 Spring Test Campaign/2025-08 Processing Folder'
    os.chdir(path)
    print(os.getcwd())
    LP_construct_database(path, fitall=False, lead_trim=9)