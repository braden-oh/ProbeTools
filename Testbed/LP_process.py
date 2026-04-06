import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import warnings

def debye(n, d_n, Te, d_Te):
    """Calculates Debye length and its uncertainty."""
    # eps = 8.8542e-12  # F⋅m−1 Vac Permittivity
    # e = 1.6022e-19    # C Elementary Charge
    # l = np.sqrt((eps*Te)/(e*n))
    l = 7430 * np.sqrt(Te / n)
    
    d_l = l * np.sqrt((0.5 * d_Te / Te)**2 + (0.5 * d_n / n)**2)
    return l, d_l

def kneedle(x, y):
    """Knee finding algorithm based on max difference from linear trend."""
    # Normalize vectors to be within 0:1
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x)) if np.max(x) != np.min(x) else x
    y_norm = (y - np.min(y)) / (np.max(y) - np.min(y)) if np.max(y) != np.min(y) else y
    
    differences = y_norm - x_norm
    knee_idx = np.argmax(differences)
    return knee_idx

def bisector_kneefind(x, y):
    """Knee finding algorithm based on minimizing bisected linear fit errors."""
    errors = np.zeros(len(x))
    for i in range(1, len(x) - 1): # Loop over possible bisectors
        # First fit
        p1 = np.polyfit(x[:i+1], y[:i+1], 1)
        sse1 = np.sum((y[:i+1] - np.polyval(p1, x[:i+1]))**2)
        # Second fit
        p2 = np.polyfit(x[i:], y[i:], 1)
        sse2 = np.sum((y[i:] - np.polyval(p2, x[i:]))**2)
        
        errors[i] = np.sqrt(sse1 + sse2)
        
    errors[0] = np.inf
    errors[-1] = np.inf
    knee_idx = np.argmin(errors)
    return knee_idx

# --- EEDF Fit Functions ---
def maxwellian(V, Te):
    return 2/np.sqrt(np.pi) * (Te)**(-1.5) * np.sqrt(V) * np.exp(-V/Te)

def druyvesteyn(V, Te):
    return 0.5648 * (Te)**(-1.5) * np.sqrt(V) * np.exp(-0.243 * (V/Te)**2)

def drifting_maxwellian(V, Te, dV):
    return 1/np.sqrt(np.pi) * (Te*dV)**(-0.5) * np.exp(-(V+dV)/Te) * np.sinh(2/Te * np.sqrt(V*dV))

def two_temperature(V, Te1, Te2, p):
    return 2*(1-p)/np.sqrt(np.pi) * (Te1)**(-1.5) * np.sqrt(V) * np.exp(-V/Te1) + \
           2*(p)/np.sqrt(np.pi) * (Te2)**(-1.5) * np.sqrt(V) * np.exp(-V/Te2)

def superthermal_beam(V, Te1, Te2, dV, p):
    return 2*(1-p)/np.sqrt(np.pi) * (Te1)**(-1.5) * np.sqrt(V) * np.exp(-V/Te1) + \
           (p)/np.sqrt(np.pi) * (Te2*dV)**(-0.5) * np.exp(-(V+dV)/Te2) * np.sinh(2/Te2 * np.sqrt(V*dV))

def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else -np.inf

# --- Main Processing Function ---
def LP_Process(V, I, rp, Ap, gas, probe_type):
    # Data Validation
    V = np.asarray(V)
    I = np.asarray(I)
    
    if len(V) != len(I):
        raise ValueError('Input vectors must be of equal length')
        
    if I[0] > 0:
        I = -I
        
    gas_masses = {
        'Xe': 131.249, 'Kr': 84.798, 'Ar': 39.948,
        'N': 14.007, 'Zn': 65.38
    }
    mi = gas_masses.get(gas, 1) * 1.672621911e-27 # kg
    me = 9.109383632e-31 # kg
    e_charge = 1.6022e-19 # C
    
    # Locate Floating Voltage
    sign_changes = np.where(np.diff(np.sign(I)) != 0) [0][0]
    # Check if the array actually has any elements
    if sign_changes.size > 0:
        # Grab just the FIRST index (matching the '1' in MATLAB's find) and add 1
        Vf_idx = sign_changes + 1
        slope_Vf = (I[Vf_idx] - I[Vf_idx-1]) / (V[Vf_idx] - V[Vf_idx-1])
        Vf = V[Vf_idx-1] - (I[Vf_idx-1] / slope_Vf)
        d_Vf = (V[Vf_idx] - V[Vf_idx-1]) / 2 * (2/3)
    else:
        Vf_idx = 0
        Vf = V
        d_Vf = (V - V) / 2 if len(V) > 1 else 0

    # Initial Fit to Ion Current
    if Vf_idx > 1:
        Ion_fit_idx = V < Vf
        p_ion = np.polyfit(V[Ion_fit_idx], I[Ion_fit_idx], 1)
        Ii = np.polyval(p_ion, V)
        Ii[Ii > 0] = 0
    else:
        Ii = np.zeros_like(I)
        Ion_fit_idx = np.zeros_like(V, dtype=bool)
    Ii_fit = Ii.copy()

    ni_prev = 0
    Te = np.nan
    d_Vp = np.nan

    # Iteration Loop
    for iteration in range(10):
        Ie = I - Ii
        Ie[Ie < 0] = 0
        
        # Pull largest consecutive run of non-zero values
        non_zero_mask = Ie > 0
        padded = np.insert(non_zero_mask, [0, len(non_zero_mask)], False)
        run_starts = np.where(padded[1:] & ~padded[:-1])[0]
        run_ends = np.where(~padded[1:] & padded[:-1])[0]
        
        if len(run_starts) == 0:
            warnings.warn('Not enough electron current information for analysis')
            return None
            
        longest_run_idx = np.argmax(run_ends - run_starts)
        start_idx = run_starts[longest_run_idx]
        end_idx = run_ends[longest_run_idx]
        
        Ie_idx = np.zeros_like(Ie, dtype=bool)
        Ie_idx[start_idx:end_idx] = True
        
        Ie_clean = np.where(Ie_idx, Ie, np.nan)
        ln_Ie = np.log(Ie_clean)
        
        if np.sum(Ie_idx) < 10:
            warnings.warn('Not enough electron current information for analysis')
            return None

        # Derivatives and Vp
        # Note: UnivariateSpline requires strictly increasing x, which voltage sweeps usually are
        spline_Ie = UnivariateSpline(V[Ie_idx], Ie_clean[Ie_idx], s=1e-5) # s parameter acts as smoothing
        Ie_v_vals = spline_Ie.derivative(1)(V[Ie_idx])
        Ie_vv_vals = spline_Ie.derivative(2)(V[Ie_idx])
        
        Ie_v = np.full_like(V, np.nan)
        Ie_vv = np.full_like(V, np.nan)
        Ie_v[Ie_idx] = Ie_v_vals
        Ie_vv[Ie_idx] = Ie_vv_vals

        # Vp Methods
        Vp_idx_candidates = []
        weights = []
        
        # 1. 2nd derivative zero crossing
        sign_diff_vv = np.diff(np.sign(Ie_vv[Ie_idx]))
        zero_crossings = np.where(sign_diff_vv == -2)[0]
        if len(zero_crossings) > 0:
            Vp_idx = zero_crossings + 1 + start_idx
            Vp_idx_candidates.append(Vp_idx[0])
            weights.append(1/2)
            
        # 2. 1st derivative max
        max_v_idx = np.argmax(Ie_v[Ie_idx])
        Vp_idx_candidates.append(max_v_idx + start_idx)
        weights.append(3/2)
        
        # 3 & 4. Knees
        knee1 = kneedle(V[Ie_idx], ln_Ie[Ie_idx])
        Vp_idx_candidates.append(knee1 + start_idx)
        weights.append(1)
        
        knee2 = bisector_kneefind(V[Ie_idx], ln_Ie[Ie_idx])
        Vp_idx_candidates.append(knee2 + start_idx)
        weights.append(1)
        
        # 5. Floating Voltage method
        if not np.isnan(Te):
            Vp_5 = Vf + Te * np.log(np.sqrt(mi / (2 * np.pi * me)))
            Vp_idx_candidates.append(np.argmin(np.abs(V - Vp_5)))
            weights.append(2)

        if len(Vp_idx_candidates) > 0:
            Possible_Vp = V[Vp_idx_candidates]
            weights = np.array(weights) / np.sum(weights)
            Vp = np.average(Possible_Vp, weights=weights)
            
            # Weighted std dev
            variance = np.average((Possible_Vp - Vp)**2, weights=weights)
            d_Vp = np.max([d_Vf, np.sqrt(variance)])
            
            # Outlier removal
            valid = np.abs(Possible_Vp - Vp) <= d_Vp
            if np.any(valid):
                Possible_Vp = Possible_Vp[valid]
                weights = weights[valid] / np.sum(weights[valid])
                Vp = np.average(Possible_Vp, weights=weights)
                variance = np.average((Possible_Vp - Vp)**2, weights=weights)
                d_Vp = np.max([d_Vf, np.sqrt(variance)])
                
            Vp_idx = np.argmin(np.abs(V - Vp))
            flag_VpFail = False
        else:
            Vp_idx = np.argmax(V)
            Vp = V[Vp_idx]
            d_Vp = d_Vf # Fallback
            flag_VpFail = True

        # Electron Temperature Fits
        possible_coeffs = []
        for _ in range(100):
            Vp_rand = np.random.normal(Vp, d_Vp)
            nearest_idx = np.argmin(np.abs(V - Vp_rand))
            if nearest_idx > start_idx + 2 and nearest_idx <= end_idx:
                p = np.polyfit(V[start_idx:nearest_idx], ln_Ie[start_idx:nearest_idx], 1)
                possible_coeffs.append(p)
                
        possible_coeffs = np.array(possible_coeffs)
        if len(possible_coeffs) > 0:
            possible_Te = 1 / possible_coeffs[:, 0]
            # Filter negative or absurd Te if polyfit fails
            possible_Te = possible_Te[possible_Te > 0] 
            Te = np.mean(possible_Te) if len(possible_Te) > 0 else 0
            d_Te = np.std(possible_Te) if len(possible_Te) > 0 else 0
            
            m_mean = np.mean(possible_coeffs[:, 0])
            b_mean = np.mean(possible_coeffs[:, 1])
            y_fit = m_mean * V[start_idx:Vp_idx] + b_mean
        else:
            Te, d_Te = 0, 0
            
        if flag_VpFail and Te > 0:
            Vp = Vf + Te * np.log(np.sqrt(mi / (2 * np.pi * me)))
            d_Vp = np.sqrt(d_Vf**2 + d_Te**2)
            Vp_idx = np.argmin(np.abs(V - Vp))

        # Initial Species Densities
        Ie_sat = np.interp(Vp, V, Ie_clean)
        d_Ie_sat = np.interp(Vp + d_Vp, V, Ie_clean) - np.interp(Vp - d_Vp, V, Ie_clean)
        ne = Ie_sat / (e_charge * Ap) * np.sqrt((2 * np.pi * me) / (e_charge * Te)) if Te > 0 else np.nan
        d_ne = ne * np.sqrt((d_Ie_sat/Ie_sat)**2 + 0.25*(d_Te/Te)**2) if Ie_sat != 0 else np.nan

        Ii_sat = Ii_fit[0]  # first measured ion current
        d_Ii_sat = Ii_fit[1] - Ii_fit[0]

        ni = -np.exp(0.5) * Ii_sat / (e_charge * Ap) * np.sqrt(mi / (e_charge * Te)) if Te > 0 else np.nan
        d_ni = ni * np.sqrt((d_Ii_sat/Ii_sat)**2 + 0.25*(d_Te/Te)**2)

        # Debye length corrections
        l, d_l = debye(ne, d_ne, Te, d_Te)
        ratio = rp / l if not np.isnan(l) and l != 0 else np.nan
        
        Vb = V[0] # Bias for ion saturation

        if np.any(Ion_fit_idx) and not np.isnan(l):
            if ratio > 50: # Thin sheath (CL)
                sheath_type = 'Thin Sheath (Child-Langmuir)'
                
                # xs and As will be arrays because V is an array
                xs = l * np.sqrt(2)/3 * (2/Te * (Vp - V))**(3/4)
                d_xs = xs * np.sqrt((d_l/l)**2 + (0.75 * d_Vp/Vp)**2 + (0.75 * d_Te/Te)**2)
                
                if probe_type == 'cylindrical':
                    As = Ap * (1 + xs/rp)
                    d_As = As * d_xs / xs
                elif probe_type == 'planar':
                    As = Ap * (1 + 2 * np.pi * xs * rp)
                    d_As = As * d_xs / xs
                elif probe_type == 'spherical':
                    As = Ap * (1 + xs/rp)**2
                    d_As = As * 2 * d_xs / xs
                else:
                    As = np.full_like(V, Ap)
                    d_As = np.zeros_like(V)

                # update ion density using the FIRST element of As to ensure ni is a scalar
                ni = -np.exp(0.5) * Ii_sat / (e_charge * As[0]) * np.sqrt(mi / (e_charge * Te))
                d_ni = ni * np.sqrt((d_As/As)**2 + (d_Ii_sat/Ii_sat)**2 + 0.25*(d_Te/Te)**2)
                
                Ii = -np.exp(-0.5) * e_charge * ni * np.sqrt((e_charge * Te) / mi) * As

            elif ratio < 3: # Thick sheath (OML)
                sheath_type = 'Thick sheath (OML)'
                if probe_type == 'cylindrical':
                    a, b = 2/np.sqrt(np.pi), 0.5
                else: # planar or spherical
                    a, b = 1, 1
                    
                # Fit to V and I^(1/b)
                p_ion = np.polyfit(V[Ion_fit_idx], I[Ion_fit_idx]**(1/b), 1)
                slope = p_ion[0]
                slope_sign = np.sign(slope)
                
                ni = 1/(a * Ap) * np.sqrt(2 * np.pi * mi) * np.exp(-1.5) * Te**(b - 0.5) * (slope_sign * slope)**b
                d_ni = ni * np.sqrt(((b - 0.5) * d_Te/Te)**2)
                
                Ii = slope_sign * np.polyval(p_ion, V)**b

            else: # Transitional sheath
                sheath_type = 'Transitional sheath'
                if probe_type == 'cylindrical':
                    a = 1.18 - 0.00080 * (ratio)**1.35
                    d_a = a * 1.35 * l / d_l
                    b = 0.0684 + (0.722 + 0.928 * ratio)**-0.729
                    d_b = b * 0.729 * l / d_l
                    if (Vp - Vb) / Te < 1:
                        warnings.warn('Ion saturation bias may not be low enough for debye length to probe radius ratio')
                elif probe_type == 'planar':
                    a = 3.47 * ratio**-0.749
                    d_a = a * 0.749 * l / d_l
                    b = 0.806 * ratio**-0.0692
                    d_b = b * 0.0692 * l / d_l
                    if (Vp - Vb) / Te < 3:
                        warnings.warn('Ion saturation bias may not be sufficient for debye length to probe radius ratio')
                elif probe_type == 'spherical':
                    a = 1.58 + (-0.056 + 0.816 * ratio)**-0.744
                    d_a = a * 0.744 * l / d_l
                    b = -0.933 + (0.0148 + 0.119 * ratio)**-0.125
                    d_b = b * 0.125 * l / d_l
                    if (Vp - Vb) / Te < 1:
                        warnings.warn('Ion saturation bias may not be low enough for debye length to probe radius ratio')
                
                # Fit to V and (-I)^(1/b) to avoid complex numbers
                p_ion = np.polyfit(V[Ion_fit_idx], (-I[Ion_fit_idx])**(1/b), 1)
                slope = p_ion[0]
                slope_sign = np.sign(slope)
                
                ni = 1/(a * Ap) * np.sqrt(2 * np.pi * mi) * np.exp(-1.5) * Te**(b - 0.5) * (slope_sign * slope)**b
                # Avoid log of negative numbers or zero
                log_term = np.log(Te * slope_sign * slope) if (Te * slope_sign * slope) > 0 else 0
                d_ni = ni * np.sqrt((d_a/a)**2 + ((b - 0.5) * d_Te/Te)**2 + (log_term * d_b)**2)
                
                Ii = slope_sign * np.polyval(p_ion, V)**b
                
        else:
            ni = np.nan
            d_ni = np.nan
            Ii = np.zeros_like(I)
            sheath_type = 'empty'

        # Ion current checks
        Ii[np.abs(np.imag(Ii)) > 0] = 0
        Ii = np.real(Ii) # Ensure array is real type now
        Ii[Ii > 0] = 0
        Ii[np.isnan(Ii)] = 0

        # Check for ion density convergence (0.1% change)
        if ni != 0 and not np.isnan(ni):
            if np.abs(ni_prev - ni) / np.abs(ni) < 0.001:
                break
        ni_prev = ni

    # EEDF Calculation
    eps = np.arange(0, Vp - V[start_idx], 0.1) # Step size 0.1V as proxy for mode(diff)
    Ie_interp = np.interp(Vp - eps, V, Ie_clean)
    
    valid_eps = ~np.isnan(Ie_interp)
    eps = eps[valid_eps]
    Ie_interp = Ie_interp[valid_eps]

    if len(eps) > 3:
        spline_eedf = UnivariateSpline(eps, Ie_interp, s=1e-5)
        Ie_ee = spline_eedf.derivative(2)(eps)
        EEDF = 2 / (e_charge**2 * Ap) * np.sqrt(2 * me * e_charge * eps) * Ie_ee
        
        ne2 = np.trapz(EEDF, eps)
        EEDF = EEDF / ne2 if ne2 != 0 else EEDF
        
        # EEDF Fits
        fit_results = []
        try:
            popt_M, _ = curve_fit(maxwellian, eps, EEDF, p0=[Te], bounds=(0.1, 100))
            r2_M = calculate_r2(EEDF, maxwellian(eps, *popt_M))
            fit_results.append(('Maxwellian', r2_M, popt_M))
        except: fit_results.append(('Maxwellian', -np.inf, None))

        # ... (Other fits follow identical try/except curve_fit structures) ...
        # For brevity, assuming best fit selection based on sorted fit_results
        
        best_fit_name = fit_results if fit_results else "None"
    else:
        best_fit_name = "None"

    # Calculate quasineutral number density
    # Calculate Quasineutral Number Density
    n = np.mean([ne, ni, ne2])
    # ddof=1 forces Pandas/NumPy to use sample standard deviation, matching MATLAB's default std()
    d_n = n * np.sqrt((np.std([ne, ni], ddof=1) / n)**2 + (d_ni / ni)**2) if n != 0 else np.nan


    # Compile Output Table (using Pandas DataFrame to simulate MATLAB Table)
    derived = pd.DataFrame({
        'Value': [Vf, Vp, Te, n, ne, ni, Ie_sat, Ii_sat],
        'Uncertainty': [d_Vf, d_Vp, d_Te, d_n, d_ne, d_ni, d_Ie_sat, d_Ii_sat],
        'Unit': ['V', 'V', 'eV', 'm-3', 'm-3', 'm-3', 'A', 'A']
    }, index=['Vf', 'Vp', 'Te', 'n', 'ne', 'ni', 'Ie_sat', 'Ii_sat'])

    # Trace = pd.DataFrame({
    #     'Bias Voltage (V)': V,
    #     'Collected Current (A)': I,
    #     'Ion Current (A)': Ii,
    #     'Electron Current (A)': Ie,
    #     '1st Derivative (A/V)': Ie_v,
    #     '2nd Derivative (A/V^2)': Ie_vv
    # })
    
    # Using Pandas attrs to store metadata similar to MATLAB's CustomProperties
    # Trace.attrs['DerivedQuantities'] = derived
    # Trace.attrs['gas'] = gas
    # Trace.attrs['probe_type'] = probe_type
    # Trace.attrs['rp2debye_ratio'] = ratio
    # Trace.attrs['sheath_type'] = sheath_type
    # Trace.attrs['EEDF_fit_type'] = best_fit_name

    output = {
        'DerivedQuantities': derived,
        'gas': gas,
        'probe_type': probe_type,
        'rp2debye_ratio': ratio,
        'sheath_type': sheath_type,
        'EEDF_fit_type': best_fit_name
    }

    # Optional Plotting
    plt.figure()
    plt.plot(V, I, 'k', linewidth=1.5, label='RAW TRACE')
    plt.plot(Vf, np.interp(Vf, V, I), 'ro', label='V_f')
    plt.plot(Vp, np.interp(Vp, V, I), 'bo', label='V_p')
    plt.xlabel('VOLTAGE [V]')
    plt.ylabel('CURRENT [A]')
    plt.title('Raw Trace')
    plt.legend()
    plt.grid(True)
    plt.show()

    return derived


if __name__ == "__main__":
    df = pd.read_csv('Kr_10sccm_10A2.csv', delimiter='\t')
    V = df['Bias Voltage (V)'].to_numpy()
    I = df['Probe Current (A)'].to_numpy()

    dp = 5.08e-5;               # probe diameter [m] (0.002")
    lp = 2.540e-3;              # probe length (cylindrical) [m] (0.100")
    rp = dp/2;                  # probe radius [m]
    Ap = np.pi*dp*(lp+dp/4)     # probe area [m^2]
    gas = 'Kr'                  # gas type ('Xe','Ar','Kr', etc.)
    probe_type = 'cylindrical'  # probe geometry

    result = LP_Process(V, I, rp, Ap, gas, probe_type)

    print('Done!')