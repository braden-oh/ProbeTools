
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import os
from sklearn.metrics import r2_score
from datetime import datetime
import pandas as pd
import ruptures as rpt
import pymc as pm
from sklearn.linear_model import RANSACRegressor, LinearRegression

def scan_dir(dir_path, extension, keyword):
    # Obtain a list of all .txt file names 
    files = os.listdir(dir_path)
    files = [f for f in files if f.endswith('.'+extension) and keyword in f]
    files.sort()
    return files

# Function to convert QCM frequency to graphite film thickness
# Link to QCM manual: https://www.inficon.com/media/4262/download/XTC-3-Thin-Film-Deposition-Controller-Operating-Manual-(English).pdf?v=1&inline=true&language=en
def freq_to_thickness(freq_measured, freq_0):
    N_AT = 166100   # [Hz-cm] "frequency constant of AT cut quartz" - see page 8-1
    d_q = 2.649     # [g/cm^3] density of quartz crystal - see page 7-3
    d_f = 2.250     # [g/cm^3] density of graphite film - see appendix A-2
    Z = 3.260       # [ ] Z ratio of graphite - see appendix A-2
    #F_q = 6.045e6   # [Hz] freq of oscillation for bare quartz crystal - see page 7-4
    F_q = freq_0        # [Hz] freq of oscillation at start of test
    F_c = freq_measured # [Hz] freq of oscillation with film (composite) - see page 7-4

    # Z-Match Technique - see page 8-5
    # Direct calculation for film thickness in units of [cm]
    T_f = N_AT*d_q / (np.pi * d_f * F_c * Z) * np.arctan(Z * np.tan(np.pi*(F_q - F_c) / F_q) )
    return T_f * 1e5    # Convert from [cm] to kilo-Angstroms


def load_QCM(path, bins=5):
    # Load the data from the file
    raw_data = pd.read_csv(path, skiprows=1)
    # Create a holding matrix for our data.  We'll format as: time, freq, thickness
    data = np.zeros((raw_data.shape[0], 3))
    # Assume we have a file formatted in the form:
    # 12:12:32,Rate,0.000,Thickness,0.041,State Time,28736,Raw Frequency,5972807.258
    time_strings = raw_data.iloc[:,0]    # Extract the first column (time strings)
    time_objects = np.array([datetime.strptime(t, "%H:%M:%S") for t in time_strings])   # Convert to datetime objects
    t0 = time_objects[0]    # Get the initial time
    data[:,0] = np.array([(t - t0).total_seconds() for t in time_objects])    # Compute time differences in seconds
    data[:,1] = raw_data.iloc[:,8]                               # Record raw frequency
    data[:,2] = freq_to_thickness(data[:,1], data[0,1])     # Calculate thickness [kAng]

    times = np.transpose(data[:,0])  # [s]
    freqs = np.transpose(data[:,1])  # [Hz]
    thics = np.transpose(data[:,2])  # [kAng]

    # Correct the midnight rollover in times
    if np.any(times < 0):
        times[times < 0] += 86400

    # Bin the adjacent measurements
    def average_chunks(arr, n):
        arr = np.asarray(arr)
        # Indices for grouping
        num_chunks = int(np.ceil(len(arr) / n))
        means = []
        for i in range(num_chunks):
            chunk = arr[i*n : (i+1)*n]
            means.append(chunk.mean())
        return np.array(means)

    times = average_chunks(times, bins)
    freqs = average_chunks(freqs, bins)
    thics = average_chunks(thics, bins)

    return times, freqs, thics  # [s] [Hz] [kAng]


def fit_line(data, make_plot=False):
    # Perform a line fit (first degree polynomial fit)
    coefficients = np.polyfit(data[:,0], data[:,2], 1)
    slope = coefficients[0]     # [kAng/sec]
    y_int = coefficients[1]     # [kAng]
    # Plot the line and data if the user wants
    if make_plot:
        plt.figure()
        plt.plot(data[:,0], data[:,2])
        plt.plot(data[:,0], coefficients[0] * data[:,0] + coefficients[1])
        plt.title('QCM Thickness Over Time')
        plt.xlabel('Time Elapsed [s]')
        plt.ylabel('Film Thickness [kiloAngstroms]')
    # Calculate the R^2 score for the line to evaluate 'goodness' of fit
    r2 = r2_score(data[:,2], coefficients[0] * data[:,0] + coefficients[1])

    rate = slope * 60 * 60 * 1000 / 10    # Convert from [kAng/sec] -> [micron/kHr]
    return slope, y_int, r2, rate


def bayesian_linreg_mcmc(
        times, thics, 
        treedepth=20, targetaccept=0.92, 
        prior_intercept=[0, 1], prior_slope=[0, 1]
    ):
    # Scale data to zero mean, unit variance
    times_mean, times_std = times.mean(), times.std()
    thics_mean, thics_std = thics.mean(), thics.std()
    times_scaled = (times - times_mean) / times_std
    thics_scaled = (thics - thics_mean) / thics_std
    with pm.Model() as model:
        mu_int, sigma_int = prior_intercept
        mu_slp, sigma_slp = prior_slope
        intercept = pm.Normal('intercept', mu=mu_int, sigma=sigma_int)
        slope = pm.Normal('slope', mu=mu_slp, sigma=sigma_slp)
        sigma = pm.HalfNormal('sigma', sigma=1)
        nu = pm.Exponential('nu', 1/30)
        mu = intercept + slope * times_scaled
        y_obs = pm.StudentT('y_obs', mu=mu, sigma=sigma, nu=nu, observed=thics_scaled)
        trace = pm.sample(
            3000, tune=1000, target_accept=targetaccept, random_seed=42, 
            nuts_sampler_kwargs={"max_treedepth": treedepth}, return_inferencedata=True
        )
        # Get samples in scaled space
        slope_s = trace.posterior['slope'].values.flatten()
        intercept_s = trace.posterior['intercept'].values.flatten()

        # Convert to original space
        slope_orig = slope_s * thics_std / times_std
        intercept_orig = thics_mean + thics_std * (intercept_s - slope_s * times_mean / times_std)

        slope_mean, slope_std = slope_orig.mean(), slope_orig.std()
        ci_95 = np.percentile(slope_orig, [2.5, 97.5])
        print(f"Mean slope: {slope_mean:.3e} ± {slope_std:.1e}")

        intercept_mean = intercept_orig.mean()
        intercept_std = intercept_orig.std()
        print(f"Mean intercept: {intercept_mean:.3e} ± {intercept_std:.1e}")

    return slope_mean, intercept_mean, ci_95




def bayesian_linreg_mcmc_OLD(times, thics, treedepth=20, targetaccept=0.92, prior_intercept=[0, 1e-4], prior_slope=[1e-7, 1e-6]):
    times_mean = times.mean()
    times_centered = times - times_mean     # Center the data about zero to help with stability

    with pm.Model() as model:
        # Priors
        # For tiny slopes, prior_intercept = [0, 1e-4], prior_slope = [1e-7, 1e-6]
        # For bigger slopes, prior_intercept = [0, 1e-1], prior_slope = [1e-6, 1e-5]
        mu_int, sigma_int = prior_intercept
        mu_slp, sigma_slp = prior_slope
        intercept = pm.Normal('intercept', mu=mu_int, sigma=sigma_int)
        slope = pm.Normal('slope', mu=mu_slp, sigma=sigma_slp)  # Informed prior, positive/slight slope
        sigma = pm.HalfNormal('sigma', sigma=1e-4)
        nu = pm.Exponential('nu', 1/30)

        # Linear model
        mu = intercept + slope * times_centered 

        # Outlier-robust likelihood
        y_obs = pm.StudentT('y_obs', mu=mu, sigma=sigma, nu=nu, observed=thics)

        # MCMC  
        # target_accept 0.95 and treedepth 15 were initial conditions that worked fairly well but led
        # to some chains sending maximum tree depth reached warnings
        # target_accept 0.95 and treedepth 20 didn't seem to change much
        # target_accept 0.98 and treedepth 20 made things worse
        # target_accept 0.92 and treedepth 20 solved it -- no more tree depth warnings!
        trace = pm.sample(3000, tune=1000, target_accept=targetaccept, random_seed=42, 
                    nuts_sampler_kwargs={"max_treedepth": treedepth}, return_inferencedata=True)

        slope_samples = trace.posterior['slope'].values.flatten()
        slope_mean = slope_samples.mean()
        slope_std = slope_samples.std()
        # 95% credible interval
        ci_95 = np.percentile(slope_samples, [2.5, 97.5])
        print(f"Mean slope: {slope_mean:.3e} ± {slope_std:.1e}")

        b_samples = trace.posterior['intercept'].values.flatten()
        b_mean = b_samples.mean()
        b_mean = b_mean - slope_mean * times_mean   # Correct the y-intercept for the centered times

    return slope_mean, b_mean, ci_95


def RANSAC_fit(X, y):
    # Fit the RANSAC model
    ransac = RANSACRegressor(LinearRegression())
    ransac.fit(X.reshape(-1,1), y)

    # The inlier linear estimator is stored in:
    estimator = ransac.estimator_
    # Slope (m): estimator.coef_[0]
    # Intercept (b): estimator.intercept_
    return estimator.coef_[0], estimator.intercept_, ransac


def obtain_indicies(dir_path, targetfile='QCM File Indicies.csv'):
    # Load the indicies file
    try:
        indicies = pd.read_csv(os.path.join(dir_path, targetfile))
        edits_made = False
    except:
        print('Generating new index file...')
        indicies = pd.DataFrame(columns=['Filename', 'Start Index', 'End Index'])
        edits_made = True

    # Loop through all QCM data files 
    for file in scan_dir(dir_path, 'txt', 'QCM'):
        full_path = os.path.join(dir_path, file)
        # Skip processing if the file is already in the master matrix
        if indicies['Filename'].str.contains(file).any():
            print('Skipping ' + file)
            continue
        edits_made = True
        print('Processing ' + file)
        # Load the QCM data from the file
        data = load_QCM(full_path)  # [s] [Hz] [kAng]
        times = np.transpose(data[:,0])  # [s]
        freqs = np.transpose(data[:,1])  # [Hz]
        thics = np.transpose(data[:,2])  # [kAng]
        plt.plot(times, thics, 'k.')
        plt.title(file)
        plt.xlabel('Times [s]')
        plt.ylabel('Thickness [kAngstrom]')
        plt.show()
        # Ask for the start and end indices for line fitting
        startin = input('Start index: ')
        try: start = int(startin)
        except: start = 0
        endin = input('End index: ')
        try: end = int(endin)
        except: end = len(times)

        indicies.loc[indicies.shape[0]] = [file, start, end]

    if edits_made:
        indicies.to_csv(os.path.join(dir_path, targetfile), index=False)
        print('Saved updated index file.')
    else:
        print('No updates performed.')
    return indicies


def obtain_breakpoints(dir_path, targetfile='QCM Breakpoints.csv'):
    # Load the breakpoints file
    try:
        breakpoints = pd.read_csv(os.path.join(dir_path, targetfile))
        edits_made = False
    except:
        print('Generating new index file...')
        breakpoints = pd.DataFrame(columns=['Filename', 'Breakpoints', 'Start', 'End', 'Length'])
        edits_made = True

    # Loop through all QCM data files 
    for file in scan_dir(dir_path, 'txt', 'QCM'):
        full_path = os.path.join(dir_path, file)
        # Skip processing if the file is already in the master matrix
        if breakpoints['Filename'].str.contains(file).any():
            print('Skipping ' + file)
            continue
        edits_made = True

        times, freqs, thics = load_QCM(file, bins=5)  # [s] [Hz] [kAng]

        plt.figure()
        plt.scatter(times, thics, alpha=0.4, label='All Data')
        plt.show()

        n_bkps = input('Number of discontinuities: ')
        try: n_bkps = int(n_bkps)
        except: n_bkps = 0

        # signal is your data, times and thicknesses are 1D numpy arrays
        signal = np.array(thics)

        algo = rpt.Binseg(model='rbf').fit(signal)
        bkps = algo.predict(n_bkps)  # Indices after changepoints, ending with len(signal)

        # Make segment boundaries: include zero at start
        boundaries = [0] + bkps  # e.g. [0, 150, 340, 600]
        segment_lengths = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1)]
        largest_idx = np.argmax(segment_lengths)
        start, end = boundaries[largest_idx], boundaries[largest_idx+1]
        print(boundaries)

        # Extract this biggest segment
        t_segment = times[start:end]
        th_segment = thics[start:end]

        breakpoints.loc[len(breakpoints)] = [file, n_bkps, start, end, len(thics)]

        plt.figure()
        plt.scatter(times, thics, alpha=0.4, label='All Data')
        plt.scatter(t_segment, th_segment, color='orange', alpha=0.7, label='Largest segment')
        # Add regression line if you wish
        plt.legend()
        plt.title(file)
        plt.show()

    if edits_made:
        breakpoints.to_csv(os.path.join(dir_path, targetfile), index=False)
        print('Saved updated index file to ' + os.path.join(dir_path, targetfile))
    else:
        print('No updates performed.')
    return breakpoints


def QCM_extract_metadata(file):
    parts = file.split('_')
    bias = int(parts[2][:-1])
    guard = '300V' in file
    split_angle = parts[1][:-3].split('-')
    if len(split_angle) == 1:
        angle = int(split_angle[0])
    elif len(split_angle) == 2:
        angle = int(split_angle[0]) + int(split_angle[1])/10**(len(split_angle[1]))
    else:
        Exception('Unexpected format for angle in file ' + file)
    return angle, bias, guard


def um_kHr(slope):
    # Convert a slope in kAng/s to um/kHr
    return slope * 60 * 60 * 1000 / 10

def cm_sec(rate):
    # Convert a slope in um/kHr to cm/s
    return rate / 60 / 60 / 1000 * 10


def QCM_construct_database(dir_path, bkp_file='QCM Breakpoints.csv', targetfile='QCM Database.csv', fitall=False):
    print('-----')
    print('Obtaining breakpoints...')
    
    breakpoints = obtain_breakpoints(dir_path, targetfile=bkp_file)    

    QCM_df = pd.DataFrame(columns=['Filename', 'Angle', 'Bias', 'Guard', 'Slope', 'Lower ci', 'Upper ci',
                                    'y_int', 'Rate', 'Lower Rate', 'Upper Rate', 'Start', 'End', 'RANSAC_m', 'RANSAC_b', 'RANSAC Rate'])
    print('-----')
    print('Constructing database...')
    try:
        database = pd.read_csv('QCM Database.csv')
        print('Found existing QCM database file.')
    except:
        database = QCM_df.copy()
        print('No QCM database file found.')

    for i, row in breakpoints.iterrows():
        # Unpack file/breakpoint data from the row
        file = row['Filename']
        start = row['Start']
        end = row['End']
        print('Loading ' + file)

        # Extract metadata from the file header
        angle, bias, guard = QCM_extract_metadata(file)
        # Load the full dataset
        times, _, thics = load_QCM(os.path.join(dir_path, file))  # [s] _ [kAng]
        
        # If the file already exists in the database (and the user hasn't asked to refit all data)...
        if not fitall and database['Filename'].str.contains(file).any():
            row = database.loc[database['Filename']==file].copy()
            # If a bayesian fit for this file aready exists...
            if 'Slope' in row.columns and type(row['Slope'].iloc[0]) == np.float64:                
                # Reuse the existing data for the bayesian fit.
                print('Bayesian fit found.  Continuing...')
                #QCM_df.loc[len(QCM_df)] = row.iloc[0]
                slope = row['Slope'].iloc[0]; y_int = row['y_int'].iloc[0]; rate = row['Rate'].iloc[0]
                lower = row['Lower Rate'].iloc[0]; upper = row['Upper Rate'].iloc[0]
                ci_95 = [cm_sec(lower), cm_sec(upper)]
        else:
            # Perform a Bayesian line fit to the target section of data 
            slope, y_int, ci_95 = bayesian_linreg_mcmc(times[start:end], thics[start:end])  # [kAng/s]
            lower = um_kHr(ci_95[0]); upper = um_kHr(ci_95[1]); rate = um_kHr(slope)
            
        if not fitall and database['Filename'].str.contains(file).any():
            row = database.loc[database['Filename']==file].copy()
            # If a RANSAC fit for this file already exists...
            if 'RANSAC_m' in row.columns and type(row['RANSAC_m'].iloc[0] ) == np.float64:
                print('RANSAC fit found.  Continuing...')
                ransac_m = row['RANSAC_m'].iloc[0]; ransac_b = row['RANSAC_b'].iloc[0]
            else:
                # Perform a RANSAC line fit to the entire dataset
                ransac_m, ransac_b, ransac = RANSAC_fit(times, thics)
                plot_RANSAC(dir_path, file, ransac, times, thics)
        else:
            # Perform a RANSAC line fit to the entire dataset
            ransac_m, ransac_b, ransac = RANSAC_fit(times, thics)
            plot_RANSAC(dir_path, file, ransac, times, thics)

        QCM_df.loc[len(QCM_df)] = [file, angle, bias, guard, slope, ci_95[0], ci_95[1], y_int, rate, 
                                   lower, upper, start, end, ransac_m, ransac_b, um_kHr(ransac_m)]

    QCM_df.to_csv(os.path.join(dir_path, targetfile), index=False)
    print('-----')
    print('Saved QCM database to ' + os.path.join(dir_path, targetfile))
    print('QCM database construction complete.')
    print('-----')
    plot_bayes_dir(dir_path, save=True, disp=False)
    return QCM_df


def select_fit_indices(times, thics):
    """
    Shows a plot and allows the user to select a region. Returns the (start, end) indices.
    """
    fig, ax = plt.subplots()
    line, = ax.plot(times, thics, 'b.', alpha=0.5, label='Thickness')
    plt.title("Drag to select the region for fitting")

    selected = []

    def onselect(xmin, xmax):
        # Convert x-values to indices
        ixmin = np.searchsorted(times, xmin)
        ixmax = np.searchsorted(times, xmax)
        selected.clear()
        selected.append(ixmin)
        selected.append(ixmax)
        plt.close(fig)  # Close the plot after selection

    span = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                        minspan=1, interactive=True)

    plt.show()

    if selected:
        start, end = selected
        return start, end
    else:
        # If nothing selected, use the whole data
        return 0, len(times)


def process_directory(dir_path, targetfile='QCM Database.csv', fitall=False, inspect=False):
    
    QCM_df = pd.DataFrame(columns=['Filename', 'Slope', 'Lower ci', 'Upper ci', 'y_int', 'Rate', 'Lower Rate', 
                                   'Upper Rate', 'Start', 'End', 'RANSAC_m', 'RANSAC_b', 'RANSAC Rate'])
    
    # Scan the directory for files containing 'QCM' and ending in '.txt'
    files = [f for f in os.listdir(dir_path) if 'QCM' in f and f.endswith('.txt')]

    print('-----')
    try:
        database = pd.read_csv(os.path.join(dir_path, 'QCM Database.csv'))
        print('Found existing QCM database file.')
    except:
        database = QCM_df.copy()
        print('No QCM database file found.')

    # Identify breakpoints
    start_dict = {}
    end_dict = {}

    print('Identifying portions of data for fitting...')
    for file in files:
        # If the user wants, inspect the data and select the section for fitting
        if inspect:
            times, _, thics = load_QCM(os.path.join(dir_path, file))  # [s] _ [kAng]
            start, end = select_fit_indices(times, thics)
        else:  # Otherwise default to using the entire dataset
            start = 0
            end = len(thics)
        # Store the start and end indicies for each file
        start_dict[file] = start
        end_dict[file] = end
    print('Identification complete.  Constructing database...')

    for file in files:
        print('--- ' + file + ' ---')
        # Load the full dataset
        times, _, thics = load_QCM(os.path.join(dir_path, file))  # [s] _ [kAng]

        # Unpack the breakpoints from earlier
        start = start_dict[file]
        end = end_dict[file]

        # If the file already exists in the database (and the user hasn't asked to refit all data)...
        if not fitall and database['Filename'].str.contains(file).any():
            row = database.loc[database['Filename']==file].copy()
            # If a bayesian fit for this file aready exists...
            if 'Slope' in row.columns and type(row['Slope'].iloc[0]) == np.float64:                
                # Reuse the existing data for the bayesian fit.
                print('Bayesian fit found.  Continuing...')
                #QCM_df.loc[len(QCM_df)] = row.iloc[0]
                slope = row['Slope'].iloc[0]; y_int = row['y_int'].iloc[0]; rate = row['Rate'].iloc[0]
                lower = row['Lower Rate'].iloc[0]; upper = row['Upper Rate'].iloc[0]
                ci_95 = [cm_sec(lower), cm_sec(upper)]
        else:
            # Perform a Bayesian line fit to the target section of data 
            slope, y_int, ci_95 = bayesian_linreg_mcmc(times[start:end], thics[start:end])  # [kAng/s]
            lower = um_kHr(ci_95[0]); upper = um_kHr(ci_95[1]); rate = um_kHr(slope)

        # Create the plots for the Bayesian and RANSAC fits
        plot_bayes(dir_path, file, slope, y_int, times, thics, start, end)
            
        if not fitall and database['Filename'].str.contains(file).any():
            row = database.loc[database['Filename']==file].copy()
            # If a RANSAC fit for this file already exists...
            if 'RANSAC_m' in row.columns and type(row['RANSAC_m'].iloc[0] ) == np.float64:
                print('RANSAC fit found.  Continuing...')
                ransac_m = row['RANSAC_m'].iloc[0]; ransac_b = row['RANSAC_b'].iloc[0]
            else:
                # Perform a RANSAC line fit to the entire dataset
                ransac_m, ransac_b, ransac = RANSAC_fit(times, thics)
                plot_RANSAC(dir_path, file, ransac, times, thics)
        else:
            # Perform a RANSAC line fit to the entire dataset
            ransac_m, ransac_b, ransac = RANSAC_fit(times, thics)
            plot_RANSAC(dir_path, file, ransac, times, thics)

        # Package the data into the dataframe
        QCM_df.loc[len(QCM_df)] = [file, slope, ci_95[0], ci_95[1], y_int, rate, 
                                   lower, upper, start, end, ransac_m, ransac_b, um_kHr(ransac_m)]
        
    # Save the database to an external file
    QCM_df.to_csv(os.path.join(dir_path, targetfile), index=False)
    print('-----')
    print('Saved QCM database to ' + os.path.join(dir_path, targetfile))
    print('QCM database construction complete.')
    print('-----')
    return QCM_df


def plot_RANSAC(dir_path, file, ransac, X, y):
    # Plot all data
    plt.scatter(X, y, color='gray', label="All Data", alpha=0.5)
    # Optionally highlight inliers
    plt.scatter(X[ransac.inlier_mask_], y[ransac.inlier_mask_], color='blue', label="Inliers", alpha=0.8)
    # For a nice line, make a range of X:
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = ransac.predict(X_line)
    plt.plot(X_line, y_line, color='red', lw=2, label="RANSAC Fit")
    plt.xlabel("Time")
    plt.ylabel("Thickness")
    plt.legend()
    os.makedirs(dir_path + '/Figures/RANSAC', exist_ok=True)
    plt.savefig(dir_path + '/Figures/RANSAC/' + file + '.png', dpi=300)
    plt.close()

def plot_bayes(dir_path, file, m, b, X, y, start, end):
    plt.plot(X, y, 'b.', alpha=0.25, label='All data')
    plt.plot(X[start:end], y[start:end], 'r.', alpha=1, label='Data used for fit')
    plt.plot(X, m*X+b, 'r-')
    plt.title(file)
    plt.xlabel('Times [s]')
    plt.ylabel('Thickness [kAngstrom]')
    plt.legend()
    os.makedirs(dir_path + '/Figures/Bayesian', exist_ok=True)
    plt.savefig(dir_path + '/Figures/Bayesian/' + file + '.png', dpi=300)
    plt.close()

def plot_bayes_dir(dir_path, save=True, disp=True):
    # Read the QCM database file
    database = pd.read_csv(os.path.join(dir_path, 'QCM Database.csv'))
    breakpoints = pd.read_csv(os.path.join(dir_path, 'QCM Breakpoints.csv'))
    
    for i, row in database.iterrows():
        # Extract database information
        file = row['Filename']
        m = row['Slope']; b = row['y_int']
        # Extract breakpoint information
        bkp_row = breakpoints[breakpoints['Filename'].str.contains(file)]
        start = bkp_row['Start'].iloc[0]; end = bkp_row['End'].iloc[0]
        # Load and plot the bayesian inference data
        times, freqs, thics = load_QCM(os.path.join(dir_path, file))
        plt.figure()
        plt.scatter(times, thics, alpha=0.4, label='All Data')
        plt.scatter(times[start:end], thics[start:end], color='orange', alpha=0.7, label='Largest segment')
        plt.plot(times, m*times+b, 'r-')
        plt.title(file)
        plt.xlabel('Times [s]')
        plt.ylabel('Thickness [kAngstrom]')
        if save:
            os.makedirs(dir_path + '/Figures/Bayesian', exist_ok=True)
            plt.savefig(dir_path + '/Figures/Bayesian/' + file + '.png', dpi=300)
        if disp:
            plt.show()
        else:
            plt.close()
    return None



if __name__=="__main__":
    #path = '/Users/braden/Documents/Beam Catcher/2025 Spring Test Campaign/2025-08 Processing Folder/HCD Chris Test (GND beam dump)'
    #os.chdir(path)
    #print(os.getcwd())
    #QCM_df = QCM_construct_database(path, bkp_file='QCM Breakpoints.csv', targetfile='QCM Database.csv', fitall=False)
    dir_path = '/Users/braden/Documents/ProbeTools/Testbed'
    process_directory(dir_path, fitall=True, inspect=True)
    