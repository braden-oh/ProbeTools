
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
import h5py

def clean_FP_data(enc_raw, I_raw):
    # Convert inputs to numpy arrays
    I_raw = np.asarray(I_raw, dtype=np.float64)
    enc_raw = np.asarray(I_raw, dtype=np.float64)
    # Step 1: Smooth encoder positions with linear interpolation
    enc = np.linspace(enc_raw[0], enc_raw[-1], len(enc_raw))
    # --- Begin MAD outlier filter ---
    window = 21  # odd, tune as needed
    halfwin = window // 2
    outlier_indices = []
    for i in range(len(I_raw)):
        start = max(0, i - halfwin)
        end = min(len(I_raw), i + halfwin + 1)
        local_window = I_raw[start:end]
        local_median = np.median(local_window)
        local_mad = np.median(np.abs(local_window - local_median))
        if local_mad == 0:
            continue
        if np.abs(I_raw[i] - local_median) > 5 * local_mad:
            outlier_indices.append(i)
    # Now create cleaned data arrays, omitting outliers
    outlier_indices = np.array(outlier_indices)
    keep_indices = np.setdiff1d(np.arange(len(I_raw)), outlier_indices)
    
    enc_clean = enc[keep_indices]
    I_clean = I_raw[keep_indices]

    # --- End MAD outlier filter ---
    # Apply Savitzky-Golay smoothing to the cleaned current
    I_smooth = savgol_filter(I_clean, window_length=21, polyorder=3)
    return enc_clean, I_clean, I_smooth

def clean_arm_data(raw_encoder, raw_current):
    # Convert inputs to numpy arrays
    current = np.asarray(raw_current, dtype=np.float64)
    encoder = np.asarray(raw_encoder, dtype=np.float64)
    # Step 1: Smooth encoder positions with linear interpolation
    smoothed_encoder = np.linspace(encoder[0], encoder[-1], len(encoder))
    # Step 2: Remove outliers based on a sliding window average
    window_size = 20
    current_avg = uniform_filter1d(current, size=window_size)
    # Identify outliers: current exceeds 10x local average
    outlier_mask = np.abs(current) > 10 * np.abs(current_avg)
    # Filter out outliers
    cleaned_current = current[~outlier_mask]
    cleaned_encoder = smoothed_encoder[~outlier_mask]
    # Apply Savitzky-Golay smoothing to the cleaned current
    smoothed_current = savgol_filter(cleaned_current, window_length=21, polyorder=3)
    return cleaned_encoder, cleaned_current, smoothed_current


def read_FP_csv(file):
    # Read data out of the csv
    data = pd.read_csv(file)
    raw_current = np.array(data['Probe current, A - Plot 0'])
    raw_encoder = np.array(data['Encoder position - Plot 1'])
    #enc, I, I_smooth = clean_FP_data(raw_encoder, raw_current)
    enc, I, I_smooth = clean_arm_data(raw_encoder, raw_current)
    return enc, I, I_smooth

def extract_FP_metadata_OLD(file):
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

def extract_FP_metadata(file):
    # Extract the bias
    bias = int(file.split('_')[2][:-1])
    # Extract the cant angle
    split_angle = file.split('_')[1][:-3].split('-')
    if len(split_angle) == 1:
        angle = int(split_angle[0])
    elif len(split_angle) == 2:
        angle = int(split_angle[0]) + int(split_angle[1])/10**(len(split_angle[1]))
    else:
        Exception('Unexpected format for angle in file ' + file)
    
    try:
        # Extract the C2G voltage
        C2G_parts = file.split('-')[-1][:-5].split('_')
        whole = C2G_parts[0]
        decimal = C2G_parts[1]
        if whole[0] == 'p': 
            whole = float(whole[1:])
            sign = 1
        else: 
            whole = float(whole)
            sign = -1
        decimal = float(decimal) / 10**len(decimal)
        C2G = (whole + decimal) * sign
        return angle, bias, C2G
    except:
        print('No C2G voltage could be read.  Returning angle and bias.')
        return angle, bias

def load_FP_sweep(file):
    # Load the dataset
    data = h5py.File(file, 'r')
    Vs = data['Data'][:][1:,1]
    Is = data['Data'][:][1:,2]
    # Extract the bias and C2G voltage
    angle, bias, C2G = extract_FP_metadata(file)
    return [angle, bias, C2G, Vs, Is]

def correct_FP_Vs(Vs_raw, Is, C2G, C2G_baseline):
    # Correct the voltages
    Vs = Vs_raw + (C2G - C2G_baseline)
    # Calculate the average and std current from 40-60V
    i0 = np.argmin(np.abs(Vs - 40))
    i1 = np.argmin(np.abs(Vs - 60))
    I_sat = np.mean(Is[i0:i1+1])
    I_std = np.std(Is[i0:i1+1])
    # Return the quantities of interest (including corrected voltages)
    return Vs, I_sat, I_std

def FP_construct_database(dir_path, save=True):
     # Obtain a list of all .txt file names 
    files = os.listdir(dir_path)
    files = [f for f in files if f.endswith('.hdf5') and 'FP' in f]
    files.sort()

    # Identify the figure save directory
    figpath = os.path.join(dir_path, 'Figures/FP')
    os.makedirs(figpath, exist_ok=True)

    # Construct the initial dataframe
    df = pd.DataFrame(columns=['Filename', 'Angle', 'Bias', 'C2G', 'Voltage', 'Current'])

    for file in files:
        # Slice the actual file name out of the path
        file = file.split('/')[-1]
        df.loc[len(df)] = [file] + load_FP_sweep(file)
        
    df = df.sort_values(by='Bias').reset_index(drop=True)
    
    if save:
        df.to_csv(os.path.join(dir_path, 'FP_database.csv'), index=False)
        print('Saved FP database to ' + os.path.join(dir_path, "FP_database.csv"))

    return df


if __name__=="__main__":
    path = '/Users/braden/Documents/Beam Catcher/2025 Spring Test Campaign/2025-07 Processing Folder/Varied Angle at 150V'
    os.chdir(path)
    print(os.getcwd())

    df = FP_construct_database(path)

    """
    # Obtain a list of all .txt file names 
    files = os.listdir(path)
    files = [f for f in files if '.csv' in f and 'RPA' in f]
    files.sort()
    print(files)

    df = pd.DataFrame(columns=['Filename', 'Angle', 'Bias', 'Encoder', 'Current', 'Smoothed'])

    for file in files:
        angle, bias = extract_FP_metadata(file)
        enc, I, I_smooth = read_FP_csv(file)
        df.loc[len(df)] = [file, angle, bias, enc, I, I_smooth]

    filtered = df[df['Bias'] == 150]
    plt.figure()
    for i, row in filtered.iterrows():
        plt.plot(row['Encoder'], row['Smoothed'], alpha=0.5, label='%.i deg' % int(np.round(row['Angle'])))
    plt.legend()
    plt.show()
    """
