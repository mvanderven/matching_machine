# -*- coding: utf-8 -*-
"""
Created on Sun May 16 09:34:18 2021

@author: mvand
"""


import pandas as pd 
import numpy as np 
from scipy import stats

# import matplotlib.pyplot as plt 


##########################################
####         STAT FUNCTIONS           ####
##########################################

def calc_gof(model, stat):
    
    ###################### SAVED BUT NOT IMPLEMENTED
    ## chi-squared 
    ## transform both timeseries to frequencies 
    # model_hist, model_bins = np.histogram(model, bins=5) 
    # stat_hist, stat_bins = np.histogram(stat, bins = model_bins)
    
    ## calculate chi-squared 
    # chi_stat, chi_p = stats.chisquare(model_hist, stat_hist)
    # print(chi_stat, chi_p)
    ######################    
    
    ## K-S test     
    D, p = stats.ks_2samp(model, stat) 
    
    ## return result of K-test 
    ## if D greater than critical p-value --> rejected 
    ## D > p 
    ## if D < p --> accepted (1, 0 if rejected)
    ## return int(D<p) 
    
    ## return: 
    ## 0 if p < 0.05 
    ## p significance score if p > 0.05 (significance level)
    if p < 0.05:
        return 0 
    else:
        return p 
 
    
def calc_distr_normal(ts):
    
    ## drop remaining missing values 
    ts = ts.dropna() 
    
    if len(ts) == 0:
        return [np.nan, np.nan, np.nan]
    
    mu = np.mean(ts)
    sigma = np.std(ts) 
        
    ## calculate goodness of fit 
    ## create an artificial dataset based on
    ## derived mu and sigma
    try:
        gof = calc_gof(ts, stats.norm.rvs(loc=mu, scale=sigma, size=len(ts)))
    except:
        gof = 0
    return [mu, sigma, gof]

def calc_distr_log(ts, eps=1e-6):
    
    ## drop remaining missing values 
    ts = ts.dropna() 
    
    if len(ts) == 0:
        return [np.nan, np.nan, np.nan]
    
    ## transform 
    Y = np.log(ts+eps) 
    Y_mu = np.mean(Y)
    Y_sigma = np.std(Y)
        
    ## calculate goodness-of-fit 
    ## create an artificial dataset
    ## based on derived distribution
    ## parameters 
    try:
        gof = calc_gof(ts, stats.lognorm.rvs(s=Y_sigma, scale=np.exp(Y_mu), size=len(ts))) 
    except:
        gof = 0

    ## alternative transformation - similar results  
    # Y_mu_alt = np.log( np.mean(ts)**2 / (( np.mean(ts)**2 + np.std(ts)**2 )**0.5 )  )
    # Y_sigma_alt = (np.log( 1 + ((np.std(ts)/np.mean(ts))**2) ))**0.5
    # gof = calc_gof(ts, stats.lognorm.rvs(s=Y_sigma_alt, scale=np.exp(Y_mu_alt), size=len(ts))) 

    return [Y_mu, Y_sigma, gof]

def calc_distr_gev(ts):
    
    ## drop remaining missing values 
    ts = ts.dropna() 
    if len(ts) == 0:
        return [np.nan, np.nan, np.nan]    
    
    ## calculate gev parameters 
    ## gumbel, so k = 0
    try:
        a = np.pi / ((6**0.5)*np.std(ts))
        u = np.mean(ts) - (0.577/a)
    
        ## calculate goodness of fit 
        ## create an artificial dataset based on
        ## derived a and u 
        gof = calc_gof(ts, stats.genextreme.rvs(c=0, loc = u, scale = a, size=len(ts))) 
    except:
        return [np.nan, np.nan, np.nan]
    return [u,a, gof]

def calc_distr_gamma(ts):
    
    ## drop remaining missing values 
    ts = ts.dropna() 
    if len(ts) == 0:
        return [np.nan, np.nan, np.nan] 
    
    try:
        ## calculate gamma parameters 
        mu = np.mean(ts)
        sigma = np.std(ts)
    
        k = (mu/sigma)**2 
        theta = sigma / (mu/sigma)
        
        ## calculate goodness of fit 
        ## create an artificial dataset based on
        ## derived k and theta (rvs)
        gof = calc_gof(ts, stats.gamma.rvs(a = k, scale = theta, size=len(ts)))
    except:
        return [np.nan, np.nan, np.nan] 
    
    return [k, theta, gof]

def calc_distr_poisson(ts):
    
    ts = ts.dropna() 

    if len(ts) == 0:
        return [np.nan, np.nan]
    
    ## poisson - number of events that will occur during
    ## a specfic time interval - her: time interval of observations 
    bins = np.arange(0, round(ts.max()+1))
    
    ## calculate n times flow has exceeded each bin    
    n_exceeded = [ (ts>bin_val).sum() for bin_val in bins  ] 
    
    ## lambda value is mean number if events in time 
    calc_lambda = np.nanmean(n_exceeded)
    
    ## calculate goodness of fit
    try:
        gof = calc_gof(n_exceeded, stats.poisson.rvs( mu = calc_lambda, size=len(ts)) )
    except:
        return [calc_lambda, np.nan]
    
    ##     lambda,  gof 
    return [calc_lambda, gof]

##########################################
####           CORRELATION           #####
##########################################


def calc_auto_correlation(ts, lag=0):
    return ts.corr(ts.shift(lag))

def calc_cross_correlation(ts0, ts1, lag=0):   
    return ts0.corr(ts1.shift(lag))

##########################################
####       FLOW DURATION CURVE        ####
##########################################

def calc_FDC(ts):
    
    ## FROM
    ## Searcy (1959) Flow-Duration Curves
    ##      Total period method: all discharges placed
    ##      according of magnitude
    
    ## drop missing values 
    ts_drop = ts.dropna() 

    if len(ts_drop) == 0:
        return int(0), int(0)
    
    ## sort values from small to large 
    ts_sorted = ts_drop.sort_values()
    
    ## calculate ranks of data 
    ## rankdata methods:
    ## (1)method=ordinal: unique rank value 
    ranks = stats.rankdata(ts_sorted, method='ordinal') 
    ## (2) method=dense: duplicate Q values get same rank value
    # ranks = stats.rankdata(ts_sorted, method='dense') 
    
    ## reverse rank order
    ranks = ranks[::-1] 
    
    ## calculate probability 
    prob = ((ranks / (len(ts_sorted)+1) )) * 100 
    return prob, ts_sorted.values  


def calc_FDC_q(ts, fdc_q):
    return_q = []
    
    ## calc FDC curve 
    fdc_prob, fdc_val = calc_FDC(ts)
    
    if ( type(fdc_prob) == int) or (type(fdc_val) == int):
        return [np.nan]*len(fdc_q)
    
    ## find corresponding quantiles 
    for q in fdc_q:        
        ix_nearest = ( np.abs(fdc_prob - q) ).argmin() 
        return_q.append(fdc_val[ix_nearest])  
        
    # print( len(ts), len(return_q), type(return_q))
    return return_q 

def calc_FDC_slope(ts, eps = 10e-6):
    
    #### from: https://github.com/naddor/camels/blob/master/hydro/hydro_signatures.R
    ####
    #### FDC slope calculation procedure from:
    ####    Sawicz et al. (2011) Catchment classifcation: empirical analysis
    ####    of hydrological similarity based on catchment function in the
    ####    eastern USA
    ####
    #### FDC_slope = log(Q_33)-log(Q_66)/(0.66-0.33)    
    
    ## calculate FDC_Q33 and Q_66
    Q33, Q66 = calc_FDC_q(ts, [33, 66])  
    return (np.log10(Q33+eps) - np.log10(Q66+eps)) / (0.66-0.33)

def calc_LF_ratio(ts, eps = 10e-6):    
    
    #### Low Flow Ratio calculation from:
    ####    Nijzink et al. (2018) Constraining conceptual hydrological
    ####    models with multiple information sources
    ####
    #### LF = FDC_Q90 / FDC_Q50
    Q90, Q50 = calc_FDC_q(ts, [90, 50]) 
    return (Q90+eps) / (Q50+eps)

def calc_ci(ts):
    
    #### Concavity Index from:
    ####    Sauquet, & Catalogne (2011) Comparison of catchment
    ####    grouping methods for flow duration curve estimation
    ####    at ungauged sites in France
    ####
    #### CI = (FDC_Q10 - FDC_Q99) / (FDC_Q1 - FDC_Q99)
    
    Q1, Q10, Q99 = calc_FDC_q(ts, [1, 10, 99])     
    return (Q10-Q99)/(Q1-Q99) 

def calc_FDC_HV(ts):
    
    #### FDC High Segment Volume from:
    ####    Shafii & Tolson (2015): Optimizing hydrological consistency
    ####    by incorporating hydrological signatures into model
    ####    calibration objectives 
    ####
    #### FDC_HV = sum(Q_h) 
    #### with: Q_h are flow values with exceedance
    #### probabilities below 0.02 
    ####
    
    ## calculate FDC
    p_vals, Q_vals = calc_FDC(ts)  
    
    ## check if FDC calculation complete 
    if not isinstance(p_vals, int):
        
        ## find high flow values (p < 0.05)
        ix_high_flows = np.where(p_vals <= 2)[0] 

        ## return sum of all high flow values 
        return Q_vals[ix_high_flows].sum()
    
    ## FDC calculation failed 
    else:
        return np.nan 


def calc_FDC_LV(ts):
    
    #### FDC Low Segment Volume from:
    ####    Shafii & Tolson (2015): Optimizing hydrological consistency
    ####    by incorporating hydrological signatures into model
    ####    calibration objectives 
    ####
    #### FDC_LV = sum( log(Q_l) - log(Q_min)  )
    #### with Q_l are flow values with exceedance
    #### probabilities over 0.7 with Q_min the minimum flow value 
    
    ## calculate FDC
    p_vals, Q_vals = calc_FDC(ts) 
    
    ## check if FDC calculation complete 
    if not isinstance(p_vals, int):
        
        ## find low flow values (0.7 > p > 1.)
        ix_low_flows = np.where(p_vals >= 70)[0] 

        ## find lowest flow value
        ## log transform discharge values
        try:
            Q_low = np.log10(Q_vals[0])
            Q_low_flows = np.log10( Q_vals[ix_low_flows] +10e-6 )
        except:
            Q_low = np.log10(Q_vals[0]+ 10e-6)
            Q_low_flows = np.log10( Q_vals[ix_low_flows] + 10e-6)

        return (Q_low_flows - Q_low).sum()
    
    ## FDC calculation failed 
    else:
        return np.nan 


##########################################
####       HYDROLOGIC INDICES         ####
##########################################

def calc_limb_slope(ts):
    
    #### Morin et al. (2002) Objective, observations-based automatic
    #### estimation of catchment response timescale: 
    ####    Indicate if slope of hydrograph increases (+1) or decreases (-1) 
    ####    "A peak is a series of +1 points followed by -1 points, possibly
    ####    with zeros in between"
    
    #### Shamir et al. (2005) The role of hydrograph indices
    #### in parameter estimation of rainfall-runoff models 
    ####    All time steps showing a positive or negative change from 
    ####    previous time step, regardless of magnitude of change, were 
    ####    included in calculation of cumulative duration of rising or
    ####    declining steps. 
    ####
    ####    A peak is defined as a time step that has a higher value from
    ####    previous and latter time steps
    
    ## calculate differences between time steps 
    ts_diff = ts.diff()  
    
    ## find peaks 
    ts_peaks = ts[ (ts> ts.shift(1) ) & (ts>  ts.shift(-1) ) ] 
    
    ## indicate if increase or decrease between time steps 
    # ts_slope = np.where( ts_diff > 0, 1, -1)
    
    ## detect peaks
    ## where a +1 point is followed by a -1 point 
    # mask_peaks = (ts_slope[:-1] > 0) & (ts_slope[1:] < 0)
    # return ts_diff, mask_peaks.sum()
    return ts_diff, len(ts_peaks)


def calc_RLD(ts):
    
    #### From:
    ####    Morin et al. (2002) Objective, observations-based automatic
    ####    estimation of catchment response timescale 
    
    #### Peak Density = 
    ####            Total peak numbers / total rising limb duration 
    ####    "Peak is a series of +1 (positive slope hydrograph) points
    ####    followed by -1 (negative slope hydrograph) points, possibly
    ####    with zeros in between"
    ####
    #### Also known as: Rising Limb Density
    
    ## get dQ and number peaks 
    slope, N_peaks = calc_limb_slope(ts)
    
    ## calculate timestep 
    delta_T = slope.index.to_series().diff()

    ### calculate total duration of positive limbs
    ### in given period 
    mask_positive_limbs = slope > 0     
    T_rising_limbs = delta_T.loc[ mask_positive_limbs ].sum()
    
    if T_rising_limbs.days > 0:
        return N_peaks / T_rising_limbs.days 
    else:
        return np.nan 

def calc_DLD(ts):
    
    #### Derived from "Peak Density" by Morin et al. (2002).
    #### Expanded by:
    ####    Shamir et al. (2005) The role of hydrograph indices
    ####    in parameter estimation of rainfall-runoff models
    
    #### Declining limb density = 
    ####              total peak numbers / total declining limb duration
                               
    
    slope, N_peaks = calc_limb_slope(ts)

    delta_T = slope.index.to_series().diff()
    
    ### calculate total duration of negative limbs 
    ### in given period 
    mask_declining_limbs = slope < 0 
    T_declining_limbs = delta_T.loc[mask_declining_limbs].sum() 
    
    if T_declining_limbs.days > 0:
        return N_peaks / T_declining_limbs.days 
    else:
        return np.nan 
    
    # T_declining_limbs = delta_T.where(slope<0).sum()
    
    # return N_peaks/T_declining_limbs.days

def calc_i_bf(ts):
    
    #### Base Flow Index following:
    ####    Sawicz et al. (2011) Catchment classification: empirical
    ####    analysis of hydrologic similarity based on catchment
    ####    function in the eastern USA
    ####
    #### Defines Base Flow Index as ratio of long-term baseflow to
    #### total streamflow. Calculated with following steps:
    #### (1) use one-parameter single-pass digital filter method 
    ####     to separate baseflow 
    ####     Q_Dt = c * Q_Dt-1 + (1+c)/2 (Q_t - Q_t-1), c = 0.925 
    #### (2) Baseflow equals: Q_Bt = Q_t - Q_Dt 
    #### (3) The baseflow index is then: I_BF = Sum (Q_B / Q)     
    
    #### coding examples from:
    #### https://github.com/naddor/camels/blob/master/hydro/hydro_signatures.R     
    #### with source code:
    ####    https://raw.githubusercontent.com/TonyLadson/BaseflowSeparation_LyneHollick/master/BFI.R 
        
    #### handle missing values 
    if ts.isnull().any():
        if len(ts) == ts.isnull().sum():
            return np.nan 
        else:
            ts = ts.fillna(method='ffill').dropna() 
            
    #### following parameter formatting by Sawicz (2011)
    Q_t  = ts.values 
    Q_D = np.zeros(len(Q_t)) 
    
    ## set initial value equal to surface flow 
    Q_D[0] = Q_t[0] 
        
    #### Value of c determined by Eckhardt (2007) A comparison of baseflow 
    #### indices, which were cal-culated with seven different baseflow 
    #### separation methods
    c = 0.925 

    #### (1) Separate direct flow from baseflow 
    for i in range(len(Q_t)-1):
        Q_D[i+1] = (c * Q_D[i]) + ( ( (1+c)*0.5 ) * (Q_t[i+1] - Q_t[i]) )
    
    ## mask Q_D values below zero 
    Q_D[Q_D<0] = 0.
    
    #### (2) Subtract direct flow from total flow to extract baseflow 
    Q_B = Q_t - Q_D  
        
    #### (3) Calculate baseflow index 
    if sum(Q_B) > 0 and sum(Q_t) > 0:
        return sum(Q_B) / sum(Q_t)
    else:
        return np.nan 


def calc_RBF(ts):
    
    #### Richard-Baker Flashiness from:
    ####    Kuentz et al. (2017) Understanding hydrologic variability
    ####    across Europe through catchment classfication 
    ####
    #### Based on:
    ####    Baker et al. (2004) A new flashiness index: characeteristics
    ####    and applications to midswestern rivers and streams 1 
    ####
    #### Application derived from:
    ####    Holko et al. (2011) Flashiness of mountain streams in 
    ####    Slovakia and Austria 
    ####
    #### RBF defined as: Sum of absolute values of day-to-day changes in
    #### average daily flow divided by total discharge during time interval.
    #### Index calculated for seasonal or annual periods, can be averaged
    #### accross years 
        
    #### SPLIT TIMESERIES 
    ts = ts.dropna()
    
    if len(ts) == 0:
        ## if length is 0, RBF cannot be calculated 
        ## return NaN 
        return np.nan 
    
    ## get delta_T to estimate if dates are continuous
    ## or with jumps 
    dT = ts.index.to_series().diff()  
        
    ## check for jumps in timeseries  
    if dT.max() > pd.Timedelta('1d'): #pd.Timedelta(value=1, unit='days'): 
        
        t_start = ts.index.values[0] 
        t_end = ts.index.values[-1] 
        
        dti = pd.date_range(start=t_start, end = t_end, freq = '12MS')
        
        ## TEST FOR MULTIPLE MOMENTS 
        ## determine chunk size 
        # t_start = ts.index.values[0] 
        # delta_T = dT.max() 
        # t_end = t_start + delta_T
        
        # ts_chunk = ts.loc[(ts.index >= t_start) & (ts.index < t_end)]  
        # n_chunks = int( len(ts) / len(ts_chunk) ) 
                
        ## get periods based on start date and frequency
        # dti = pd.date_range(start=t_start, freq='12MS', periods = n_chunks) 
        # print(dti)
        
        # print(t_start, ts.index.values[-1])
        # dti = pd.date_range(start=t_start, end = ts.index.values[-1], freq='12MS')
        # print(dti)
        
    ## else split based on years 
    else:
        iter_list = ts.index.year.unique()
        dti = pd.date_range(start = ts.index[0], freq='YS', periods = len(iter_list))
    
    ## go over all periods 
    ## calculate RBF
    collect_RBF = []
    for i in range(len(dti)):

        if i < (len(dti)-1):  
            mask = (ts.index >= dti[i]) & (ts.index < dti[i+1])
            
        if i == (len(dti)-1):
            mask = (ts.index >= dti[i])
        
        ## get period of interest
        _ts = ts.loc[mask]
        
        if len(_ts) > 0:
            ## sum absolute day-to-day differences 
            _sum_abs_diff = _ts.diff().abs().sum() 
            
            ## sum total flow 
            _sum_flows = _ts.sum() 

            if (_sum_abs_diff > 0) & (_sum_flows > 0):
                collect_RBF.append((_sum_abs_diff/_sum_flows))
    
    ## return average Richard-Baker Flashiness
    if len(collect_RBF) > 0:
        return_RBF = np.nanmean(np.array(collect_RBF))
    else:
        return_RBF = np.nan 
    return return_RBF

def calc_recession_curve(ts):
    
    #### Recession curve characteristics based on:
    ####    Stoelzle et al. (2013) Are streamflow recession characteristics 
    ####    really characteristic?
    #### and:
    ####    Vogel & Kroll (1992) Regional geohydrologic-geomorphic 
    ####    relationships for the estimation of low-flow statistics
    ####    
    #### Method described by Vogel & Kroll:
    ####    "A recession period begins when a 3-day moving average begins
    ####    to decrease and ends when it begins to increase. Only recession 
    ####    of 10 or more consecutive daysare accepted as candidate 
    ####    hydrograph recessions."
    ####
    ####    "If the total length of a candidate recession segment is I,  
    ####    then the initial portion is predominatnly surface or storm 
    ####    runoff, so first lambda*I days were removed from the 
    ####    candidate recession segment."
    ####    First guess for lambda = 0.3 
    ####
    ####    "To avoid spurious observations, only accept pairs of stream-
    ####    flow where Q_t > 0.7*Q_t-1"
    ####
    ####    "Fit log-log relation:
    ####    ln(-dQ/dt) = ln(a) + b*len(Q) + eps "
    ####    Linear reservoir assumption: b = 1 
    ####    Use ordinary least squares estimators for a and b 
    ####    using all accepted dQ and Q pairs 
    
    ## Vogel & Kroll chose lambda = 0.3, looking at summer months only 
    ## Here, (multi-) annual periods, seasons and months are used
    ## In their study, lambda was varied between 0 and 0.8, to choose
    ## a value of lambda corresponding to an average value of b = 1 
    
    ## lower value of lambda decreases b 
    ## higher value of lambda increases b 
    # init_lambda = 0.3  
    init_lambda = 0.05
    
    ts = ts.dropna()
   
    if len(ts) == 0:
        return [np.nan, np.nan]
    
    ## calculate 3-day moving average 
    ts_mv = ts.rolling(window=3, center=True).mean() 
    
    ## calculate differences between two time steps 
    ts_mv_diff = ts_mv.diff() 

    ## mask recession periods 
    recession_mask = ts_mv_diff <= 0. 

    if recession_mask.sum() == 0:
        ## no recession periods detected 
        return [np.nan, np.nan]
        
    ## collect in dataframe 
    _df = pd.DataFrame()
    _df['Q'] = ts_mv[recession_mask]     
    _df['dQ'] = ts_mv_diff[recession_mask]    
    _df['dT'] = _df.index.to_series().diff() 
        
    ## identify periods 
    _df['periods'] =  _df['dT'].dt.days.ne(1).cumsum() 
    
    ## identify period length 
    for period_id in _df['periods'].unique():
        p_df = _df[ _df['periods'] == period_id]         
        _df.loc[ p_df.index, 'len'] = len(p_df)
    
    ## drop all periods with a length below 10 days 
    _df_dropped = _df[ _df['len'] >= 10].copy()
    
    ## check change in differences 
    ## Q_t > 0.7 Q_t-1
    Q_t1 = _df_dropped['Q'] 
    Q_t0 = 0.7 * _df_dropped['Q'].shift(-1)
    
    ## add results to df 
    _df_dropped['check_change'] = Q_t1 > Q_t0 
    
    ## collect final results 
    ## after checking change magnitudes 
    _df_dropped['analyse'] = 0 
    for period_id in _df_dropped['periods'].unique():
        p_df = _df_dropped[ _df_dropped['periods']==period_id] 
        
        ## compute number of False occurencies 
        n_false = len(p_df) - sum(p_df['check_change']) 
        last_false = False 
        
        if n_false == 1:
            ## get index of false value 
            idx_false = p_df[ p_df['check_change'] == False ].index         
                    
            ## if false idx and last idx match: continue 
            if idx_false == p_df.tail(1).index:
                last_false = True 
        
        if (n_false == 0) | (last_false) :
            ## if regression period is accepted, discard the first 
            ## lambda * len(p_df) values from the period
            n_skip = round( init_lambda * len(p_df) )
            ix_to_keep = p_df.iloc[n_skip:].index  
            _df_dropped.loc[ix_to_keep, 'analyse'] = 1
    
    ## keep marked Q/dQ couples 
    _df_analyse = _df_dropped[ _df_dropped['analyse'] == 1][['Q', 'dQ']].copy()
    
    if not len(_df_analyse) > 0:
        return [np.nan, np.nan]
    
    ## plot Q and delta_Q on a natural logarithmic scale 
    _df_analyse['log_Q'] = np.log( (_df_analyse['Q']+ 10e-6) )
    _df_analyse['log_dQ'] = np.log( abs(_df_analyse['dQ'] + 10e-6) )
    
    ## use ordinary least squares regression to find a and b in:
    ## ln(-dQ) = ln(a) + b*len(Q) 
    slope, intercept, r_value, p_value, std_err = stats.linregress( _df_analyse['log_Q'], _df_analyse['log_dQ'] )     
    
    ## interpret results 
    b = slope 
    a = intercept 
    # a = np.exp(intercept)    
    return [b, a]

def calc_peak_distr(ts):
    
    #### From:
    ####    Euser et al. (2013) A framework to assess the realism 
    ####    of model structures usinghydrological signatures
    ####
    #### "Signature shows whether peak discharges are of equal height"
    #### "Peak is defined as discharge at a time step of which both
    #### previous and following timestep have a lower discharge" 
    #### An FDC is constructed based on peak data 
    #### Average slope between 10th and 50th percentile is 
    #### calculated 
    #### By taking 10th and 50th percentile, only higher peaks
    #### are taken into account, but not the extremes 
    
    #### 1 - find peaks 
    ts_peaks = ts[ ( ts > ts.shift(1) ) & ( ts > ts.shift(-1) ) ] 
    
    if len(ts_peaks) > 0:
        
        #### 2 - construct FDC based on peak data
        #### and calculate values for Q10 and Q50 
        Q10, Q50 = calc_FDC_q(ts_peaks, [10, 50])
        
        ## return slope         
        return (Q10 - Q50) / (0.9-0.5)
    
    ## no peaks found 
    else:
        return np.nan 


def calc_event_stats(ts, threshold_value, condition = '>'):
    
    ts = ts.dropna() 
    
    if len(ts) == 0:
        return np.nan, np.nan 
    
    #### split dataseries based on years or periods in data 
    dT = ts.index.to_series().diff()  
        
    ## check for jumps in timeseries  
    if dT.max() > pd.Timedelta('1d'): #pd.Timedelta(value=1, unit='days'): 
        
        t_start = ts.index.values[0] 
        t_end = ts.index.values[-1] 
        
        dti = pd.date_range(start=t_start, end = t_end, freq = '12MS')
                
    ## else split based on years 
    else:
        iter_list = ts.index.year.unique()
        dti = pd.date_range(start = ts.index[0], freq='YS', periods = len(iter_list)) 
    
    ## go over all periods 
    collect_frequency = [] 
    collect_duration = []
    
    for i in range(len(dti)):

        if i < (len(dti)-1):  
            mask = (ts.index >= dti[i]) & (ts.index < dti[i+1])
            
        if i == (len(dti)-1):
            mask = (ts.index >= dti[i])
        
        ## get period of interest
        _ts = ts.loc[mask]
        if len(_ts) > 0:
            
            if condition == '>':
                _ts_event = _ts.loc[ _ts >= threshold_value] 
            else:
                _ts_event = _ts.loc[ _ts <= threshold_value] 
            
            if len(_ts_event) > 0:
                                
                _df = pd.DataFrame() 
                _df['obs'] = _ts_event 
                _df['dT'] = dT.loc[ _ts_event.index ]
                _df['periods'] =  _df['dT'].dt.days.ne(1).cumsum()
                                  
                _gby = _df.groupby(['periods', 'dT']).size().reset_index(level=1, 
                                                                        drop=True)
                
                n_events = len(_gby) 
                len_events = _gby.values 
                                 
                collect_frequency.append(n_events) 
                
                for len_event in len_events:
                    collect_duration.append(len_event)
                
            else:
                collect_frequency.append(0)
                collect_duration.append(0)
                
    return np.mean(collect_frequency), np.mean(collect_duration)


def high_flow_events(ts):
    
    #### High Flow Event Frequency & Duration:
    ####    Westerberg & McMillan (2015) Uncertainty
    ####    in hydrological signatures
    ####
    #### High Flow Event Frequency:
    ####    Average nr of daily high flow events per year 
    ####    with high flow events defined as 9*Q_median 
    ####
    #### High Flow Event Duration:
    ####    Average duration of daily flow events with 
    ####    consecutive days of flow > 9*Q_median 
    
    ts = ts.dropna() 
    
    if len(ts) == 0:
        return np.nan, np.nan 
    
    ## calculate high flow threshold for entire time series 
    high_flow_threshold = 9 * ts.median() 
    
    ## calc and return frequency and duration
    return calc_event_stats(ts, high_flow_threshold, condition = '>')
    
def low_flow_events(ts):
    
    #### Low Flow Event Frequency & Duration:
    ####    Westerberg & McMillan (2015) Uncertainty
    ####    in hydrological signatures
    #### and
    ####    Clausen & Biggs (2000) Flow variables for 
    ####    ecological studies in temperate streams:
    ####    groupings based on covariance
    ####
    #### Low Flow Event Frequency:
    ####    Average nr of daily low flow events per year 
    ####    with low flow events defined as 0.2*Q_mean
    ####
    #### Low Flow Event Duration:
    ####    Average duration of daily flow events with 
    ####    consecutive days of flow < 0.2*Q_mean    
    
    ts = ts.dropna() 
    
    if len(ts) == 0:
        return np.nan, np.nan 
    
    ## calculate low flow threshold for entire time series 
    low_flow_threshold = 0.2 * ts.mean() 
    
    ## calc and return frequency and duration
    return calc_event_stats(ts, low_flow_threshold, condition = '<')


def str_order_nr():
    return 

def upA_match():
    return 










