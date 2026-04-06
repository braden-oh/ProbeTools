function [Trace]  = LP_Process(V,I,rp,Ap,gas,probe_type) 

%LP_Process.m - One line description of what the function or script performs (H1 line)
%Optional file header info (to give more details about the function than in the H1 line)
%
% Syntax:  [Vf, d_Vf, Te, Vp_Te, Vp_knee, LP] = LP_Process(filename, yn_plots) 
%
% Inputs:
%    filename --  LP trace .txt file at 90 degrees with V in first column, I in second column   
%    yn_plots -- 0 = no graphs & 1 = graphs 
%    flag --1 = read from .hdf5,  flag -- 0 = read from .txt or csv
%
% Outputs:
%    output1 -- Description
%    output2 -- Description
%
% Other m-files required: none
% Subfunctions: none
% MEX-files required: none
% MAT-files required: none
%
%
% LP_Process Version 2.0
% MAR-15-2024
% tategill@umich.edu
% Change Notes:
% - Inlcuded iteration over ion current fits based on probe radius to debye
% length ratio
% - Included Druyvesteyn method for EEDF fits
%
%
% LP_Process Version 1.2
% NOV-20-2023
% tategill@umich.edu
% Change Notes:
% - refactored code to only accept matlab vectors, file handling should not
% be embedded in probe analysis code
% - added check for Ie positive, not just V>Vf 
%
% LP_Process Version 1.1
% AUG-17-2023
% tategill@umich.edu
% Change Notes:
% - Added bootstrapping for evalauation of Te to give more insightful
% uncertainty values
% - Added automatic selection of Vp method. This tries (in order) zero
% crossing of 2nd derivative of Ie, max of 1st derivative of Ie, ...
% 
% See also: 
% Author: Tate Gill
% email: tategill@umich.edu
% JAN 2023; Last revision: JAN-02-2023
%
% Revision By: Grace Zoppi
% email: gzoppi@umich.edu
% Last revision: AUG-07-2023
%
%------------- BEGIN CODE ------------------------------------------------%
    
%==========================================================================
% DATA VALIDATION
%==========================================================================

%Check for equal length inputs
if ~length(I) == length(V)
    error('Input vectors must be equal length');
end

%ensure correct 'electron current +' polarity
if I(1) > 0 %check if first entry (ion current) is positive
    I = -I;
end

switch gas %for input of ion mass
    case 'Xe'
        mi = 131.249*1.672621911e-27; %kg
    case 'Kr'
        mi = 84.798*1.672621911e-27; %kg
    case 'Ar' 
        mi = 39.948*1.672621911e-27; %kg
    case 'N'
        mi = 14.007*1.672621911e-27; %kg
    case 'Zn'
        mi = 65.38*1.672621911e-27; %kg, presumably
end

me = 9.109383632e-31; %kg electron mass
e = 1.6022e-19; %C elementary Charge

smooth_factor = .999;

%==========================================================================
% LOCATE FLOATING VOLTAGE
%==========================================================================

%find where current crosses zero, also find starting point: V_floating
if ~isempty(find(diff(sign(I)), 1))
    Vf_idx = find(diff(sign(I)), 1) + 1;
    %linearly interpolate between these two points
    slope_Vf = (I(Vf_idx) - I(Vf_idx-1)) ./ (V(Vf_idx) - V(Vf_idx-1));
    Vf = V(Vf_idx-1) - (I(Vf_idx-1) / slope_Vf);
    
    %Floating potential uncertainty
    d_Vf = (V(Vf_idx) - V(Vf_idx-1)) / 2 * 2/3; %added factor of 2/3's is asumming a 3 sig distribution btween the indices and 95% CI is 2 sigma 
else
    Vf_idx = 1;
    Vf = V(1);
    d_Vf = (V(Vf_idx+1) - V(Vf_idx)) / 2; 
end

%==========================================================================
% INITAL FIT TO ION CURRENT
%==========================================================================

%Fit a line to ion saturation current for V<Vf
if ~(Vf_idx==1 || Vf_idx==2) % i.e. there are measurements below floating voltage
    Ion_fit_idx = V<Vf;
    IonFit = fit(V(Ion_fit_idx),I(Ion_fit_idx),'poly1'); %linear fit object
    Ii = IonFit(V); %ion saturation current
    Ii(Ii>0) = 0; %ion current cannot be positive     
else %no ion saturation region
    Ii = zeros(size(I)); %no ion current
    Ion_fit_idx = zeros(size(V));
end

%==========================================================================
% BEGIN ITERATION LOOP TO CALCULATE ION SATURATION FITS
%==========================================================================

ni_prev = 0; %set inital "previous" ion density to compare changes to

for iter = 1:10

    Ie = I - Ii; %define electron current
    Ie(Ie<0) = 0; %electron current cannot be negative 
    
    %pull out largest consecutive run of non zero values in Ie 
    %(this is the section of electron current we really want)
    nan_loc = find(~([0; Ie; 0])); %find location of zeros
    [~, long_run_idx] = max(diff(nan_loc)); %find index of longest difference between zeros
    list = nan_loc(long_run_idx):nan_loc(long_run_idx+1)-2; %create list of indices for the longest run
    Ie_idx = false(size(Ie)); %create 0-array to store logical indices
    Ie_idx(list) = true; %convert index list to array of logical values
    Ie(~Ie_idx) = nan; %set all data in Ie not in longest run to (zero) (ignore the stragglers)
    
    ln_Ie = log(Ie); %store log of electron current

    Ie_start_idx = find(~isnan(Ie),1);
    Ie_end_idx = find(~isnan(Ie),1,'last');

    if sum(Ie_idx) < 10 %Not enough electron current information for analysis
        warning('Not enough electron current information for analysis');
        [Trace] = deal(nan);
        return
    end
    
    %==========================================================================
    % DERIVATIVES OF ELECTRON CURRENT & LOCATE PLASMA POTENTIAL
    %==========================================================================
    
    %fit smoothing spline to electron current and evaluate derivitives
    f = fit(V(Ie_idx),Ie(Ie_idx),'smoothingspline','SmoothingParam', smooth_factor,'Normalize','on'); 
    [Ie_v, Ie_vv] = differentiate(f,V); %first and second derivitive of electron current
    
    %Vp methods
    Vp_idx = nan(5,1);
    w = [1/2  3/2   1   1   2]; %weights for averaging these values. equal total weight for knee vs derivatives. More trust for first derivative
    %w = [10 10 1 1 1/2];

    %2nd derivative zero crossing (+ to - means -2)
        idx = find(diff(sign(Ie_vv(Ie_idx)))==-2, 1)+1;
        if ~isempty(idx); Vp_idx(1) = idx; end
    %1st derivative max
        [~, idx] = max(Ie_v(Ie_idx)); %find derivative peak
        if ~isempty(idx); Vp_idx(2) = idx; end
    %Knee finding algorithms of ln_Ie
        idx = kneedle(V(Ie_idx),ln_Ie(Ie_idx));
        if ~isempty(idx); Vp_idx(3) = idx; end
        idx = bisector_kneefind(V(Ie_idx),ln_Ie(Ie_idx));
        if ~isempty(idx); Vp_idx(4) = idx; end
    %Floating volatage method
        if exist('Te','var') && ~isnan(Te) %second iteration or higher 
            Vp_5 = Vf + Te*log(sqrt(mi/(2*pi*me)));
            [~, Vp_idx(5)] =  min(abs(V-Vp_5));
        end
    
    %indices are all found with respect to Ie_idx
    Vp_idx(1:4) = Vp_idx(1:4)+Ie_start_idx-1;
    
    Vp_idx(Vp_idx<Ie_start_idx) = nan; %these values must be non physical

    failed_methods = isnan(Vp_idx);
    Vp_idx(failed_methods) = []; %remove nans
    w(failed_methods) = []; %remove nans
    w = w./sum(w); %renormalize weights

    if ~isempty(Vp_idx) %vaild Vp found      
        Possible_Vp = V(Vp_idx);
        Vp = w*Possible_Vp; %weighted average
        d_Vp = max([d_Vf std(Possible_Vp,w)]);
        
        outliers = abs(Possible_Vp-Vp)>d_Vp; %identify outliers
        Vp_idx(outliers) = []; %remove outliers
        w(outliers) = []; %remove outliers
        w = w./sum(w); %renormalize weights
        
        Possible_Vp = V(Vp_idx);
        Vp = w*Possible_Vp; %weighted average
        d_Vp = max([d_Vf std(Possible_Vp,w)]);
        
        [~,Vp_idx] = min(abs(V-Vp)); %closest index to average
    
        flag_VpFail = 0; %flag indicating that Vp is found

        if outliers(2) == 1 %no first derivative method included
            [Vp, Vp_idx] = max(V); %set plasma potential to max of voltage sweep
            %d_Vp = 0;
            flag_VpFail = 1; %flag indicating that Vp is (not) found
        end

    else % all methods failed
        [Vp, Vp_idx] = max(V); %set plasma potential to max of voltage sweep
        %d_Vp = 0;
        flag_VpFail = 1; %flag indicating that Vp is (not) found
    end

    %==========================================================================
    % ELECTRON TEMPERATURE FITS
    %==========================================================================  
    %Fit a bunch of lines for electron temp
    possible_coeffs = zeros(100,2);
    for i = 1:100
        Vp_rand = randn*d_Vp+Vp; %get normally distrbuted random Vp from distribution
        [~,nearest_idx_to_Vp_rand] = min(abs(V-Vp_rand)); %nearest index in V to Vp_rand
        if nearest_idx_to_Vp_rand == 1 || nearest_idx_to_Vp_rand-Ie_start_idx <=2 || nearest_idx_to_Vp_rand > Ie_end_idx
            continue %toss out point if it corresponds to the first index or isnt far enough away from start of electron current
        else
        f = fit(V(Ie_start_idx:nearest_idx_to_Vp_rand),ln_Ie(Ie_start_idx:nearest_idx_to_Vp_rand),'poly1'); %linear fit between Vf and nearest index to Vp_rand
        end
        %store coefficients of fit
        possible_coeffs(i,1) = f.p1; 
        possible_coeffs(i,2) = f.p2; 
    end
    
    %remove invalid points (zeros)
    possible_coeffs(possible_coeffs(:,1) == 0,:) = [];
    
    %aggregate fit
    m = mean(possible_coeffs(:,1));
    b = mean(possible_coeffs(:,2));
    f = cfit(fittype('poly1'),m,b);
    y_fit = feval(f,V(Ie_start_idx:Vp_idx));
    
    %compute statistics on all fitted Te's
    possible_Te = 1./possible_coeffs(:,1);
    if isempty(possible_Te)
        Te = 0;
        d_Te = 0;
    else 
        Te = mean(possible_Te);
        d_Te = std(possible_Te);
    end
    
    if flag_VpFail %use floating voltage and electron temp to estimate Vp
        Vp = Vf + Te*log(sqrt(mi/(2*pi*me)));
        d_Vp = sqrt(d_Vf^2 + d_Te^2);
        [~, Vp_idx] =  min(abs(V-Vp));
        y_fit = feval(f,V(Ie_start_idx:Vp_idx)); %recalc y_fit b/c Vp_idx changes
    end
    
    %================================= =========================================
    % COMPUTE INITAL SPECIES DENSITES
    %==========================================================================
    
    Ie_sat = interp1(V,Ie,Vp); %linearly interp  at Vp to find electron saturation current
    d_Ie_sat = interp1(V,Ie,Vp+Vp) - interp1(V,Ie,Vp-Vp);  % Possible typo here -- should this be Vp+d_Vp and Vp-d_Vp?
    ne = Ie_sat/(e*Ap)*sqrt((2*pi*me)/(e*Te));
    d_ne = ne * sqrt((d_Ie_sat/Ie_sat)^2 + 1/4*(d_Te/Te)^2);
    
    Ii_sat = Ii(1); %first measured ion current;
    d_Ii_sat = Ii(2)-Ii(1); 
    ni = -exp(1/2)*Ii_sat/(e*Ap)*sqrt((mi)/(e*Te)); %thin sheath (probe area)
    d_ni = ni * sqrt((d_Ii_sat/Ii_sat)^2 + 1/4*(d_Te/Te)^2);
    
    %==========================================================================
    % USE PROBE RADIUS TO DEBYE LENGTH TO CORRECT SPECIES DENSITIES
    %==========================================================================
    
    [l, d_l] = debye(ne,d_ne,Te,d_Te); %using ne because it *should be* unaffected by sheath size uncertainty
    ratio = rp/l;
        
    Vb = V(1); %bias for ion saturation
    
    if any(Ion_fit_idx & ~isnan(l))

        if ratio > 50 %Thin sheath (CL)   
            sheath_type = 'Thin Sheath (Child-Langmuir)';
            xs = l*sqrt(2)/3*(2/Te*(Vp-V)).^(3/4); %sheath thickness
            d_xs = xs*sqrt((d_l/l)^2+(3/4*d_Vp/Vp)^2+(3/4*d_Te/Te)^2); 
            
            switch probe_type
                case 'cylindrical'
                    As = Ap*(1+xs/rp);
                    d_As = As.*d_xs./xs;
                case 'planar' %this one is made up. Assuming sheath now includes a cylindrical "label" of length xs
                    As = Ap*(1+2*pi*xs*rp);
                    d_As = As.*d_xs./xs;
                case 'spherical'
                    As = Ap*(1+xs/rp).^2;
                    d_As = As.*2.*d_xs./xs;
            end
            
            %update ion density
            ni = -exp(1/2)*Ii_sat/(e*As(1))*sqrt((mi)/(e*Te));
            d_ni = ni * sqrt((d_As(1)./As(1))^2+(d_Ii_sat/Ii_sat)^2 + 1/4*(d_Te/Te)^2);
        
            Ii = -exp(-1/2)*e*ni*sqrt((e*Te)/(mi))*As; %ion current
           
        elseif ratio < 3 %Thick sheath (OML)
            sheath_type = 'Thick sheath (OML)';
            switch probe_type
                case 'cylindrical'
                    a = 2/sqrt(pi);
                    b = 1/2;
                case 'planar' 
                    a = 1;
                    b = 1;
                case 'spherical'
                    a = 1;
                    b = 1;
            end
            
            IonFit = fit(V(Ion_fit_idx),I(Ion_fit_idx).^(1/b),'poly1');
            slope_sign = sign(IonFit.p1);
            ni = 1/(a*Ap)*sqrt(2*pi*mi)*e^(-3/2)*Te^(b-1/2)*(slope_sign*IonFit.p1)^b;
            d_ni = ni*sqrt(((b-1/2)*d_Te/Te)^2); %needs fit uncertainty
        
            Ii = slope_sign*IonFit(V).^b; %ion current

        else %transistional sheath
            sheath_type = 'Transistional sheath';
            switch probe_type
                case 'cylindrical'
                    a = 1.18 - 0.00080*(ratio)^1.35;
                    d_a = a*1.35*l/d_l;
                    b = 0.0684 + (0.722 + .928*ratio)^-0.729;
                    d_b = b*.729*l/d_l;
                    if (Vp-Vb)/Te < 1
                        warning('ion saturation bias may not be low enough for debye length to probe radius ratio')
                    end
                case 'planar' 
                    a = 3.47*ratio^-0.749;
                    d_a = a*.749*l/d_l;
                    b = 0.806*ratio^-0.0692;
                    d_b = b*.0692*l/d_l;
                    if (Vp-Vb)/Te < 3
                        warning('ion saturation bias may not be sufficicent for debye length to probe radius ratio')
                    end
                case 'spherical'
                    a = 1.58 + (-0.056 + .816*ratio)^-0.744;
                    d_a = a*0.744*l/d_l;
                    b = -0.933 + (0.0148 + .119*ratio)^-0.125;
                    d_b = b*.125*l/d_l;
                    if (Vp-Vb)/Te < 1
                        warning('ion saturation bias may not be low enough for debye length to probe radius ratio')
                    end
            end
            
            IonFit = fit(V(Ion_fit_idx),(-I(Ion_fit_idx)).^(1/b),'poly1'); %fit to positive ion current to avoid complex results
            slope_sign = sign(IonFit.p1);
            ni = 1/(a*Ap)*sqrt(2*pi*mi)*e^(-3/2)*Te^(b-1/2)*(slope_sign*IonFit.p1)^b;
            d_ni = ni*sqrt((d_a/a)^2+((b-1/2)*d_Te/Te)^2+(log(Te*slope_sign*IonFit.p1)*d_b)^2); % needs fit uncertainty
        
            Ii = slope_sign*IonFit(V).^b; %ion current            
        end

    else
        ni = nan;
        d_ni = nan;
        Ii = zeros(size(I)); %no ion current
        sheath_type = 'empty';
    end
    
    %ion current checks
    Ii(abs(imag(Ii))>0) = 0; %imaginary results set to zero
    Ii(Ii>0) = 0; %ion current cannot be positive 
    Ii(isnan(Ii)) = 0; %needs to be a number

    %Check for ion desnity convergence .1% change
    if abs(ni_prev-ni)/ni < (0.1/100); break
    else; ni_prev = ni; end   

end

%==========================================================================
% CALCULATE EEDF AND FIT
%==========================================================================
%This is the no smoothing way to calculate this
%{
% e_retard_idx = Ie_idx & V<=Vp; %indicies where electron are retarded.
% eps = Vp-V(e_retard_idx);
% Ie_e = gradient(Ie(e_retard_idx),eps);
% Ie_ee = gradient(Ie_e,eps);
% eps = flipud(eps); Ie_ee = flipud(Ie_ee); %flip order of arrays for increaseing eps

% figure(1); clf; hold on; grid on;
% plot(eps,Ie_e)
% 
% figure(2); clf; hold on; grid on;
% plot(eps,Ie_ee)
%}

eps = (0:abs(mode(diff(V)/2)):Vp-V(Ie_start_idx:Vp_idx))'; %mode(diff(V)/2) is as much resolution as you can increase before you get numerical oscillations
Ie_interp = interp1(Vp-V,Ie,eps);

%ignore nans
eps(isnan(Ie_interp)) = []; 
Ie_interp(isnan(Ie_interp)) = []; 

%fit smoothing spline to electron current and evaluate derivitives
f = fit(eps,Ie_interp,'smoothingspline','SmoothingParam', smooth_factor,'Normalize','on'); 
[~, Ie_ee] = differentiate(f,eps); %first and second derivitive of electron current

% figure(1)
% plot(eps,Ie_e)
% 
% figure(2)
% plot(eps,Ie_ee)

EEDF = 2/(e^2*Ap) * sqrt(2*me*e*eps).*Ie_ee;

ne2 = trapz(eps,EEDF);
Te2 = 2/3*trapz(eps,EEDF.*eps)/ne2;

EEDF = EEDF/ne2; %normalize by density

% See Phys. Plasmas 26, 063513 (2019); doi: 10.1063/1.5093892 
% "Enhanced method for analyzing Langmuir probe data and characterizing 
% the Electron Energy Distribution Function (EEDF) K. Trent et. al.

%bunch of EEDF fits
maxwellian = fittype('2/sqrt(pi)*(Te)^(-3/2)*sqrt(V).*exp(-V/Te)',...
    'independent',{'V'});
try
    [f_M, gof_M] = fit(eps,EEDF,maxwellian,...
        'startpoint',  Te,...
             'lower',  0.1,...
             'upper', 100);
catch %if fitting fails assign empty to fit and zero r^2
    f_M = []; 
    gof_M.rsquare = -inf;
    warning('EEDF Fit: Maxwellian Failed');
end

druyvesteyn = fittype('0.5648*(Te)^(-3/2)*sqrt(V).*exp(-.243*(V/Te)^2)',...
     'independent',{'V'});
try
    [f_D, gof_D] = fit(eps,EEDF,druyvesteyn,...
    'startpoint',  Te,...
         'lower', 0.1,...
         'upper', 100);
catch %if fitting fails assign empty to fit and zero r^2
    f_D = []; 
    gof_D.rsquare = -inf;
    warning('EEDF Fit: Druyvesteyn Beam Failed');
end

drifting_maxwellian = fittype('1/sqrt(pi)*(Te*dV)^(-1/2)*exp(-(V+dV)/Te).*sinh(2/Te*sqrt(V*dV))',...
     'independent',{'V'});
[~,idx_max] = max(EEDF);
try
    [f_Mv, gof_Mv] = fit(eps,EEDF,drifting_maxwellian,...
        'startpoint',[ Te, eps(idx_max)],...
             'lower',[0.1,            0],...
             'upper',[100,          100]);
catch %if fitting fails assign empty to fit and zero r^2
    f_Mv = []; 
    gof_Mv.rsquare = -inf;
    warning('EEDF Fit: Drifting Maxwellian Failed');
end

two_temperature = fittype('2*(1-p)/sqrt(pi)*(Te1)^(-3/2)*sqrt(V).*exp(-V/Te1) + 2*(p)/sqrt(pi)*(Te2)^(-3/2)*sqrt(V).*exp(-V/Te2)',...
    'independent',{'V'});
try
    [f_2t, gof_2t] = fit(eps,EEDF,two_temperature,...
        'startpoint',[ Te, 2*Te,  0.1],...
             'lower',[0.1,  0.1, 0.05],...
             'upper',[100,  100,   .9]);
catch %if fitting fails assign empty to fit and zero r^2
    f_2t = []; 
    gof_2t.rsquare = -inf;
    warning('EEDF Fit: Two Temperature Failed');
end

superthermal_beam = fittype('2*(1-p)/sqrt(pi)*(Te1)^(-3/2)*sqrt(V).*exp(-V/Te1) + (p)/sqrt(pi)*(Te2*dV)^(-1/2)*exp(-(V+dV)/Te2).*sinh(2/Te2*sqrt(V*dV))',...
    'independent',{'V'});
try
    [f_stb, gof_stb] = fit(eps,EEDF,superthermal_beam,...
        'startpoint',[ Te, 0.1*Te, eps(idx_max),  0.1],...
             'lower',[0.1,    0.1,            0, 0.05],...
             'upper',[100,    100,          100,  0.9]);
catch %if fitting fails assign empty to fit and zero r^2
    f_stb = []; 
    gof_stb.rsquare = -inf; 
    warning('EEDF Fit: Superthermal Beam Failed');
end

fits = {f_M, f_D, f_Mv, f_2t, f_stb};
gofs = [gof_M.rsquare, gof_D.rsquare, gof_Mv.rsquare, gof_2t.rsquare, gof_stb.rsquare];

[~,bestfit] = max(gofs);
%bestfit = 1; %PLOT MAXWELLIAN ALWAYS

fit_types = {'Maxwellian','Druyvesteyn','Drifting Maxwellian','Two Temperature','Superthermal Beam'};
EEDF_fit_type = fit_types{bestfit};
EEDF_fit = evalc('display(fits{bestfit})');

legend_str = {'measured EEDF';...
    sprintf('Maxwellian Fit, r^2 = %.4f',gof_M.rsquare);...
    sprintf('Druyvesteyn Fit, r^2 = %.4f',gof_D.rsquare);...
    sprintf('Drifting Maxwellian Fit, r^2 = %.4f',gof_Mv.rsquare);...
    sprintf('Two Temp. Maxwellian Fit, r^2 = %.4f',gof_2t.rsquare);...
    sprintf('Super Thermal Beam Maxwellian Fit, r^2 = %.4f',gof_stb.rsquare)};

% figure(6); clf; hold on; grid on;
% scatter(eps,EEDF,'k')
% plot(f_M,'r')
% plot(f_D,'b')
% plot(f_Mv,'g')
% plot(f_2t,'c')
% plot(f_stb,'m')
% legend(legend_str);


%==========================================================================
% COMPILE OUTPUT
%==========================================================================

%Assign Arrayed values to table
Trace = table(V,I,Ii,Ie,ln_Ie,Ie_v,Ie_vv);
Trace.Properties.VariableNames = {'Bias Voltage','Collected Current','Ion Current','Electron Current','Natural Log of Electron Current','1st Derivative of Electron Current','2nd Derivative of Electron Current'};
Trace.Properties.VariableUnits = {'V','A','A','A','ln(A)','A/V','A/V^2',};


n = mean([ne,ni,ne2]); %calaulate quasineutral number density
d_n = n*sqrt((std([ne,ni],0)/n)^2 + (d_ni/ni)^2);

      Value = [  Vf;   Vp;   Te;     n;    ne;    ni;   Ie_sat;   Ii_sat];
Uncertainty = [d_Vf; d_Vp; d_Te;   d_n;  d_ne;  d_ni; d_Ie_sat; d_Ii_sat];
       Unit = { 'V';  'V'; 'eV'; 'm-3'; 'm-3'; 'm-3';      'A';      'A'};
      Names = {'Vf'; 'Vp'; 'Te';   'n';  'ne';  'ni'; 'Ie_sat'; 'Ii_sat'};

DerivedQuantites = table(Value,Uncertainty,Unit,'RowNames',Names);

Trace.Properties.UserData = DerivedQuantites; %assign calculated variables to user data

%include gas, probe area, probe radius, probe type, sheath type, ratio, best EEDF fit. 
Trace = addprop(Trace,{  'gas', 'probe_type', 'probe_area', 'probe_radius', 'rp2debye_ratio', 'sheath_type', 'EEDF_fit_type', 'EEDF_fit'},...
                      {'table',      'table',      'table',        'table',          'table',       'table',         'table',    'table'});

Trace.Properties.CustomProperties.gas = gas;
Trace.Properties.CustomProperties.probe_type = probe_type;
Trace.Properties.CustomProperties.probe_area = [num2str(Ap),' m^2'];
Trace.Properties.CustomProperties.probe_radius = [num2str(rp),' m'];
Trace.Properties.CustomProperties.rp2debye_ratio = num2str(ratio);
Trace.Properties.CustomProperties.sheath_type = sheath_type;
Trace.Properties.CustomProperties.EEDF_fit_type = EEDF_fit_type;
Trace.Properties.CustomProperties.EEDF_fit = EEDF_fit;

%==========================================================================
% OUTPUT PLOT
%==========================================================================

figure;
plot(V,I,'k','linewidth',1.5)%plot raw trace
hold on
errorbar(Vf,interp1(V,I,Vf),0,0,d_Vf,d_Vf,'ro','linewidth',1.5) %Plot floating voltage
errorbar(Vp,interp1(V,I,Vp),0,0,d_Vp,d_Vp,'bo','linewidth',1.5) %Plot plasma potential
hold off
legend({'RAW TRACE','V_f','V_p'},...
   'location','best')
xlabel('VOLTAGE [V]')
ylabel('CURRENT [A]')
title('Raw Trace')

%{
%PLOTS
fig = figure(1); clf;
fig.Units = 'pixels';
%fig.Position = [100 100 1200 1000];
t = tiledlayout(2,3,'TileSpacing','compact');
%title(t, filename, 'Interpreter', 'none')

nexttile(t,1); hold on; grid on
    plot(V,I,'k','linewidth',1.5)%plot raw trace
    plot(V,Ii,'k--','LineWidth',1); %plot ionsaturation current fit to data
    errorbar(Vf,interp1(V,I,Vf),0,0,d_Vf,d_Vf,'ro','linewidth',1.5) %Plot floating voltage
    errorbar(Vp,interp1(V,I,Vp),0,0,d_Vp,d_Vp,'bo','linewidth',1.5) %Plot plasma potential
    legend({'RAW TRACE','Ion current fit','V_f','V_p'},...
       'location','best')
    xlabel('VOLTAGE [V]')
    ylabel('CURRENT [A]')
    title('Raw Trace')

nexttile(t,2); hold on; grid on
    plot(V,ln_Ie,'k','linewidth',1.5); %plot log of electron current
    plot(V(Ie_start_idx:Vp_idx),y_fit,'k--','LineWidth',1); %plot aggregate linear fit to data
    errorbar(Vp,interp1(V,ln_Ie,Vp),0,0,d_Vp,d_Vp,'bo','linewidth',1.5) %plot plasma potential
    xlabel('VOLTAGE [V]')
    ylabel('LN(CURRENT)')
    title('Natural Log of Electron Current')
    legend({'LN(ELECTRON CURRENT)','Average Linear Fit', 'V_p'},...
       'location','southeast')

nexttile(t,3); hold on; grid on
    histogram(possible_Te);
    xline(Te,'k--');
    title(sprintf('T_e = %.2f +/- %.2f eV',Te, d_Te))
    grid on
    xlabel('T_e, eV'); ylabel('counts');  
    
nexttile(t,4); hold on; grid on
    plot(V,Ie_v,'k','linewidth',1.5)%plot 1st derivative
    errorbar(Vp,interp1(V,Ie_v,Vp),0,0,d_Vp,d_Vp,'bo','linewidth',1.5) %plot plasma potential
    xlabel('VOLTAGE [V]')
    ylabel('[A/V]')
    title('1st Derivative of I_e')
    legend('1st DERIVITIVE OF I_e','V_p','location','best')
    xlim([V(Ie_start_idx) Vp+d_Vp]); %only show plot for V < VP

nexttile(t,5);  hold on; grid on
    plot(V,Ie_vv,'k','linewidth',1.5)%plot 2nd derivative
    errorbar(Vp,interp1(V,Ie_vv,Vp),0,0,d_Vp,d_Vp,'bo','linewidth',1.5) %plot plasma potential
    xlabel('VOLTAGE [V]')
    ylabel('[A/V^2]')  
    title('2nd Derivative of I_e')
    legend('2nd DERIVITIVE OF I_e','V_p','location','best')
    xlim([V(Ie_start_idx) Vp+d_Vp]); %only show plot for V < VP

nexttile(t,6);  hold on; grid on
    scatter(eps,EEDF,'k','LineWidth',1.5)
    l = plot(fits{bestfit},'r-');
    l.LineWidth = 1.5;
    xlabel('REL. VOLTAGE TO PLASMA POT. [V_p - V]')
    ylabel('EEDF, arb.')
    title('EEDF')
    legend('Measured EEDF',[EEDF_fit_type,' fit '],'location','best')
    xlim([0 Vp-V(Ie_start_idx)]);
%}    


end
%------------- END OF CODE ------------------------------------------------%


function knee_idx = kneedle(x,y)
    %normalize vectors to be within 0:1
    x = normalize(x,1,"range");
    y = normalize(y,1,"range");

    differences = (y-x);

    [~,knee_idx] = max(differences);
end

function knee_idx = bisector_kneefind(x,y)
    
errors = zeros(size(x));
for i = 2:length(x)-1 %loop over all possible bisectors
    [~,gof1] = fit(x(1:i),y(1:i),'poly1'); %first fit
    [~,gof2] = fit(x(i:end),y(i:end),'poly1'); %second fit
    
    errors(i) = sqrt(gof1.sse + gof2.sse);
end
    errors([1,end]) = []; %delete invalid bisector points
    [~,knee_idx] = min(errors);
    knee_idx = knee_idx+1; %add back the offset from the removed inital point 
end

function [l, d_l] = debye(n,d_n,Te,d_Te)
    eps = 8.8542e-12; %F⋅m−1 Vac Permitivity
    e = 1.6022e-19; %C Elementary Charge

    l = sqrt((eps*Te)./(e*n));
    l = 7430*sqrt(Te/n);
    d_l = l*sqrt((1/2*d_Te/Te)^2+(1/2*d_n/n)^2);
end
