# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 16:02:05 2021

@author: mvand
"""

import pandas as pd 
import numpy as np 
from pathlib import Path 
import xarray as xr 

import signatures_functions 


##########################################
####       SUPPORT FUNCTIONS          #### 
##########################################

def read_gauge_data(fn_list, transform = False, src_proj = None, dst_proj = None, resample24hr=False):
    
    '''
    Function that reads different types of gauge data - 
    and returns a dataframe with same set/order of columns:
    [loc_id, quantity, unit, date, time, value, epsg, x, y]
    
    fn         path to (csv) datafile 
    dtype      datasource - file will be opened accordingly 
    transform  transform coordinate system 
    '''
    
    out_df = None
    meta = {}

    output_cols = ['date', 'time', 'value', 'quantity', 
                   'epsg', 'nan_val', 'loc_id', 'y', 
                   'x', 'upArea', 'unit']
    
    out_df = pd.DataFrame(columns=output_cols)
        
    for fn in fn_list:   
        
        assert Path(fn).exists(), '[ERROR] file not found'
        
        ## get gauge id nr from filename             
        gauge_id_nr = fn.name.split('_')[0]
        
        ## open file 
        ##  other options for encoding:
        ##  encoding = 'mbcs'(windows only - not for linux) # encoding = 'ansi'
		##  tried: ansi (does not work on linux?), utf-8, ascii, cp500 
        df = pd.read_csv(fn, skiprows=36, delimiter=';', encoding = 'cp850') 
        #print(df)
        #print(df.columns)
		
        ## extract data 
        discharge = df[' Value'].values 
        dt_dates = pd.to_datetime(df['YYYY-MM-DD'], yearfirst=True, 
                                  format='%Y-%m-%d')

        ## what time
        dt_time =  pd.Series(['00:00:00']*len(dt_dates))
        # dt_time =  pd.Series(['12:00:00']*len(dt_dates))

        ## create output dataframe 
        temp_df = pd.DataFrame( {'date':dt_dates, 'time':dt_time, 
                                'value':discharge})

        ## get metadata 
        byte_df = open(fn, encoding='cp850')
        lines = byte_df.readlines()[:36]
        
        temp_df['loc_id'] = gauge_id_nr 
        meta['loc_id'] = gauge_id_nr 
        
        for line in lines:
            vals = line.split(' ')

            if 'Station:' in vals:
                meta['loc_name'] = vals[-1].replace('\n', '').lower()
                temp_df['loc_id_name'] = meta['loc_name']

            if 'missing' in vals:
                meta['nan'] = float(vals[-1])
                temp_df['nan_val'] = float(vals[-1])

            if 'Latitude' in vals:
                meta['lat'] = float(vals[-1])
                # temp_df['y'] = meta['lat']
                temp_df['lat'] = meta['lat']

            if 'Longitude' in vals:
                meta['lon'] = float(vals[-1])
                # temp_df['x'] = meta['lon']
                temp_df['lon'] = meta['lon']

            if 'Unit' in vals:
                meta['unit'] = vals[-1].replace('\n', '')
                temp_df['unit'] = meta['unit']

            if 'area' in vals:
                meta['upArea(km2)'] = float(vals[-1])
                temp_df['upArea'] = meta['upArea(km2)'] 
              
            if 'Altitude' in vals:
                elevation = float(vals[-1]) 
                if elevation <= -999.:
                    elevation = np.nan 
                    
                meta['elevation'] = elevation 
                temp_df['elevation'] = meta['elevation']

            if 'Content:' in vals:
                meta['description'] = ' '.join( vals[-3:] ).replace('\n', '').lower()

            temp_df['quantity'] = 'Q'
            temp_df['epsg'] = 4326

        ## set nan values 
        temp_df.loc[ temp_df['value'] == meta['nan'], 'value' ] = np.nan        

        ## close meta file 
        byte_df.close()
        lines = None 

        ## aggregate date and time - set as index 
        ## set format same as glofas/efas date format 
        temp_df.index = pd.to_datetime( temp_df['date'],
                                            format='%Y-%m-%dT%H:%M:%S.%f' 
                                           ).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        out_df = out_df.append(temp_df)
    return out_df, meta

def rows_to_cols(df, id_col, index_col, target_col, resample_24hr = False):
    
    out_df = pd.DataFrame()
    
    unique_ids = df[id_col].unique() 
    
    for target_id in unique_ids:
        
        _df = df[ df[id_col] == target_id ] 
        col_names = _df.index.unique() 
        
        for col_name in col_names:
            col_df = _df[ _df.index == col_name ] 
            col_df = col_df.set_index(index_col) 
            
            out_df[col_name] = col_df[target_col]
    
    if resample_24hr:
        out_df.index = pd.to_datetime( out_df.index  )
        out_df = out_df.resample('D').mean()
        
    out_df.index = pd.to_datetime(out_df.index)    
    return out_df


##########################################
####     ORGANIZE FEATURE TYPES       ####
####               &                  ####
####          TIME WINDOWS            #### 
##########################################

#### FEATURE TYPES 
sorted_features = {
    'stats':        ['normal', 'log', 'gev', 'gamma', 'poisson'],
    'correlation':  ['n-acorr', 'n-ccorr'],
    'fdc':          ['fdc-q', 'fdc-slope', 'lf-ratio', 'conc-index', 
                     'fdc-hv', 'fdc-lv'],
    'hydro':        ['bf-index', 'dld', 'rld', 'rbf', 'src', 'peak-distr',
                     'hf-stat', 'lf-stat']
    }

stat_options  = sorted_features['stats'] 
corr_options  = sorted_features['correlation']
fdc_options   = sorted_features['fdc']
hydro_options = sorted_features['hydro']

feature_keys = list(sorted_features.keys())
feature_options = []

for key in feature_keys:
    for val in sorted_features[key]:
        feature_options.append(val)


option_time_window = ['all', 'annual', 'seasonal', 'monthly', 'weekly']
time_format = {
    'all':          [':'],
    'annual':       ['Y'],
    'seasonal':     [month%12 // 3 + 1 for month in range(1, 13)],
    'monthly':      ['M'],
    'weekly':       ['W']}


##########################################
####            OVERVIEW              #### 
####   dictionary to call feature     #### 
####   functions with named output    #### 
####            variables             ####
##########################################
 
func_dict = {
    'normal':   {
        'func': signatures_functions.calc_distr_normal,
        'cols': ['Nm-{}', 'Ns-{}', 'N-gof-{}']
        },
    'log':      {
        'func': signatures_functions.calc_distr_log,
        'cols': ['Lm-{}', 'Ls-{}', 'L-gof-{}']        
        },
    'gev':      {
         'func': signatures_functions.calc_distr_gev,
        'cols': ['Gu-{}', 'Ga-{}', 'Gev-gof-{}']       
        },
    'gamma':    {
        'func': signatures_functions.calc_distr_gamma,
        'cols': ['Gk-{}', 'Gt-{}', 'G-gof-{}']        
        },
    'poisson':  {
         'func': signatures_functions.calc_distr_poisson,
        'cols': ['Pl-{}', 'P-gof-{}']       
        },
    'n-acorr':  {
         'func': signatures_functions.calc_auto_correlation,
        'cols': ['alag-{}-{}']       
        },
    'n-ccorr':  {
         'func': signatures_functions.calc_cross_correlation,
        'cols': ['clag-{}-{}']
        },
    'fdc-q':  {
         'func': signatures_functions.calc_FDC_q,
        'cols': ['fdcQ-{}-{}']       
        },
    'fdc-slope':  {
         'func': signatures_functions.calc_FDC_slope,
        'cols': ['fdcS-{}']  
        },
    'lf-ratio':  {
         'func': signatures_functions.calc_LF_ratio,
        'cols': ['lf-{}']  
        },
    'conc-index':  {
        'func': signatures_functions.calc_ci,
        'cols': ['ci-{}']  
        },
    'fdc-hv':  {
        'func': signatures_functions.calc_FDC_HV,
        'cols': ['fhv-{}']  
        },
    'fdc-lv':  {
        'func': signatures_functions.calc_FDC_LV,
        'cols': ['flv-{}']  
        },
    'bf-index':  {
         'func': signatures_functions.calc_i_bf,
        'cols': ['bfi-{}']  
        },
    'rld':  {
         'func': signatures_functions.calc_RLD,
        'cols': ['rld-{}']  
        },
    'dld':  {
         'func': signatures_functions.calc_DLD,
        'cols': ['dld-{}']  
        },
    'rbf':  {
         'func': signatures_functions.calc_RBF,
        'cols': ['rbf-{}']  
        },
    'src':  {
         'func': signatures_functions.calc_recession_curve,
        'cols': ['b_rc-{}', 'a_rc-{}']  
        },
    'peak-distr':  {
         'func': signatures_functions.calc_peak_distr,
        'cols': ['pks-{}']  
        },
    'hf-stat':  {
         'func': signatures_functions.high_flow_events,
        'cols': ['hf-f-{}', 'hf-t-{}']  
        },
    'lf-stat':  {
         'func': signatures_functions.low_flow_events,
        'cols': ['lf-f-{}', 'lf-t-{}']  
        },
    'order-nr':  {
         'func': signatures_functions.str_order_nr,
        'cols': ['str-nr']  
        },
    'upA-match':  {
         'func': signatures_functions.upA_match,
        'cols': ['upA-m']  
        }
    }


def calc_signatures(df, gauge_col,
                    features = feature_options, time_window = option_time_window,
                    fdc_q = [1, 2, 5, 10, 50, 90, 95, 99],
                    n_alag = [1], n_clag = [0,1]):
    
    '''
    Function that calculates given features with specified time windows. 
    
    INPUT
        df          dataframe with observations on each row, indexed with
                    a datetime index
                    in each column, observations of a different location 
                    can be found, with column name the identifier of the
                    location 
                    
        gauge_col   the column name of the observation column, the rest
                    of the columns are assumed to be simulation columns
        
        features    names of the features to be calculated in list. 
                    if nothing is selected, all
                    available features are calculated 
                    
                    Options:
                        Statistical distributions: 
                            normal, log-normal, gumbel & gamma
                            ['normal', 'log', 'gev', 'gamma']
                            
                        Correlation:
                            n-lagged autocorrelation, n-lagged cross-
                            correlation (observations with simulations)
                            ['n-acorr', 'n-ccor']
                            
                        FDC:
                            i-th quantile of FDC, slope of FDC, 
                            low flow index 
                            ['fdc-q', 'fdc-slope', 'lf-ratio']
                            
                        Hydro inidices:
                            baseflow index, declining limb density,
                            rising limb density, Richard-Baker Flashiness,
                            recession curve parameters 
                            ['bf-index', 'dld', 'rld', 'rbs', 'src']
        
        time_window time spans over which signatures are calculated. 
                    Options:                    
                        ['all', 'annual', 'seasonal', 'monthly', 'weekly']
                        
        fdc_q       If quantiles of FDC ['fdc-q'] are calculated, the
                    desired quantiles can be specified in a list 
                    with integers. 
                    
                    Example:
                        fdc_q = [1, 50, 99] 
                        will calculate the 1st, 50th and 99th quantile of 
                        the flow duration curve
        
        n_alag      If n-lagged autocorrelation ['n-acorr'] is calculated,
                    the target lag(s) can be specified in a list 
                    with integers. 
                    
                    Example:
                        n_alag = [1, 5] will calculate the 1 and 5 lagged
                        autocorrelation of the time series 
        
        n_clag      If n-lagged cross-correlation ['n-ccorr'] is calculated,
                    the target lag(s) can be specified in a list with 
                    integers.
                    
                    Example:
                        n_alag = [0,1] will calculate the 0-lag and 1-lag 
                        cross correlation of simulated and observed [gauge_col]
                        timeseries.    
    
    OUTPUT
        df_out      Dataframe with calculated features - on each row the
                    features belonging to an input column can be found, with 
                    in each output column a feature value.
    '''
    
    print('\n [START] signature calculation')  

    calc_cols = df.columns 
    obs_cols = [col for col in calc_cols if not gauge_col in col] 
    
    ## organize features 
    stat_features =  [feat for feat in features if feat in stat_options]
    corr_features =  [feat for feat in features if feat in corr_options]
    fdc_features =   [feat for feat in features if feat in fdc_options]
    hydro_features = [feat for feat in features if feat in hydro_options] 

    df_out = pd.DataFrame() 
    df_out['ID'] = calc_cols 
    df_out = df_out.set_index('ID') 
    
    ### setup for different time windows 
    for tw in time_window: 
                
        time_index = df.index 
        
        if tw == 'all':
            df['slice'] = 0 
        
        if tw == 'annual':
            df['slice'] = time_index.year 
        
        if tw == 'seasonal':
            time_windows = dict(zip(range(1,13), time_format[tw])) 
            df['slice'] = time_index.month.map(time_windows)
            
        if tw == 'monthly':
            df['slice'] = time_index.month 

        if tw == 'weekly':
            df['slice'] = time_index.isocalendar().week
          
        ## calc feature 
        for feature in features:
            # print(' [CALC] {}'.format(feature))
            ## get output names
            result_cols = func_dict[feature]['cols']
            
            ## go over time window 
            for _slice in df['slice'].unique():
                
                ## get data 
                calc_df = df[ df['slice'] == _slice][calc_cols]
                
                ## get name 
                if tw == 'all':
                    result_name = '{}'.format(tw) 
                else:
                    result_name = '{}_{}'.format(tw, _slice) 
                
                ## perform calculations 
                if feature in stat_features:
                    
                    ## calculate statistics 
                    results = calc_df.apply(func_dict[feature]['func'])  
                                        
                    ## save 
                    for i, ix in enumerate(results.index):                      
                        _col = result_cols[i].format(result_name) 
                        df_out.loc[ results.columns, _col ] = results.loc[ix] 
                
                if feature in corr_features: 
                                        
                    if 'acorr' in feature:
                        
                        ## loop over lag times 
                        for i, lag in enumerate(n_alag):
                            if lag < len(calc_df):
                                ## calculate lagged correlation
                                results = calc_df.apply( func_dict[feature]['func'], lag = lag ) 
                                ## save 
                                _col = result_cols[i].format(lag, result_name) 
                                df_out.loc[ results.index, _col ] = results.values 
                                            
                    if 'ccorr' in feature: 
                                                
                        for i, lag in enumerate(n_clag):
                            if lag < len(calc_df):
                                 
                                ## setup name 
                                _col = result_cols[0].format(lag, result_name)
                                
                                ## calculate cross-correlation per column 
                                for col in obs_cols:
                                    result =  func_dict[feature]['func']( calc_df[gauge_col], calc_df[col], lag=lag  ) 
                                    df_out.loc[col, _col] = result
                                    
                if feature in fdc_features: 
                    
                    if 'fdc-q' in feature:
                        
                        results = calc_df.apply(func_dict[feature]['func'], fdc_q = fdc_q)
                        
                        for i, q in enumerate(fdc_q): 
                            _col = result_cols[0].format(q, result_name)                             
                            df_out.loc[results.columns, _col] = results.loc[i,:].values 
                    
                    else:
                        results = calc_df.apply(func_dict[feature]['func']) 
                        _col = result_cols[0].format(result_name)  
                        df_out.loc[results.index, _col] = results.values
                        
                
                if feature in hydro_features:

                    results = calc_df.apply(func_dict[feature]['func']) 
                    
                    if len(result_cols) > 1:
                        for i, col in enumerate(result_cols):
                            _col = col.format(result_name) 
                            df_out.loc[results.columns, _col] = results.loc[i,:].values
                            
                    else:
                        _col = result_cols[0].format(result_name) 
                        df_out.loc[results.index, _col] = results.values 
     
    print(' [FINISH] signature calculation') 
    return df_out 

def calc_vector(df_signatures, gauge_id, cols_preserve = ['clag', 'n_buffer']):
    
    '''
    Functions that takes a dataframe containing signature values from 
    one observation point and corresponding simulations and calculates 
    similarities of simulations with observations. 
    
    INPUT
        df_signatures   dataframe with signature values - each row contains
                        a unique simulation or observation, with signature
                        values in columns. Unique IDs in index. 
        
        gauge_id        the ID of the row containing observation signatures,
                        used to compare the simulation signatures with 
        
        cols_preserve   columns that are left out of the similarity 
                        calculation. Input as list with string values.
                        
                        Example (& default value):
                            cors_preserve = ['clag']
                            
                            Since 'clag' is cross-correlation, it is already
                            a measurement of similarity and does not need
                            to be taken into account
    
    OUTPUT
        df_sim          a dataframe with on each row a similarity values
                        of simulations with observations, with in each 
                        column a similarity feature value 
    '''
    
    
    print('\n [START] similarity vector calculation')
    
    ## list columns for subtraction of gauge signature values
    ## to observe similarity 
    cols_intact = [] 
    for keep_col in cols_preserve:
        cols = [col for col in df_signatures.columns if keep_col in col] 
        for col in cols:
            cols_intact.append(col)
        
    cols_subtract = [col for col in df_signatures.columns if col not in cols_intact]  

    ## separate observation and simulation data from signature data 
    df_obs = df_signatures.loc[gauge_id] 
    df_sim = df_signatures.drop(index=gauge_id) 

    ## subtract signature values of observations from
    ## signature values of simulations
    df_sim[cols_subtract] = df_sim[cols_subtract] - df_obs[cols_subtract] 
    
    print('\n [FINISH] similarity vector calculation')
    return df_sim 

def assign_labels(df, key, gauge_meta):
    
    '''
    Function that assigns target labels to features as preparation for
    a predictive algorithm 
    
    INPUT
        df          dataframe containing feature values belonging to 
                    samples 
        
        key         link between df and gauge_meta - sample ID in df 
                    is looked up in key to match with gauge_meta
        
        gauge_meta  dataframe containing target values, is linked through
                    'key' with df 
    
    OUTPUT
        df          a dataframe with an additional column 'target',
                    in which all samples have a value 0, 1 or -1.
                    
                    In the first case, if the target value is found in the
                    available samples, the target sample is labelled with 1,
                    while the other samples are labelled with 0.
                    
                    In other cases, if the target value is not found in the
                    available samples, all samples are labelled with -1.
    '''
        
    print('\n [START] assigning target labels') 
    
    target_X, target_Y = gauge_meta[['Lisflood_X', 'Lisflood_Y']] 
    
    target_cell = key[ (key['x'] == target_X) & (key['y'] == target_Y) ]
        
    if len(target_cell) > 0:
        ## target cell in buffer 
        df['target'] = 0 
        df.loc[target_cell.index, 'target'] = 1
    else:
        ## target cell not in buffer 
        df['target'] = -1
    
    print('\n [FINISH] assigning target labels')
    return df 

def df_to_ds(df, grid_key, gauge_id, data_type): 
    
    '''
    Function that transforms all data in a dataframe ('df'), with the help
    of a key ('grid_key') into a xarray dataset
    
    INPUT
        df          dataframe with samples on each row and features in 
                    columns, identified by IDs in index 
        
        grid_key    dataframe linked with IDs in index, identical to 
                    IDs in 'df', containing spatial data of each sample 
        
        gauge_id    overall identifier of samples (simulations) belonging
                    to gauge observations
        
        data_type   description of data type 
    
    OUTPUT
        ds_out      xarray dataset with feature data of each sample in a 
                    spatial grid 
    '''
        
    print(' [START] translate dataframe to netcdf')
    ## find size of grid - square root of number of cells 
    n_buffer = int( len(df)**0.5 ) 
    
    df_vars = df.columns 

    fill_grid = np.zeros(( int(len(df_vars)+1) ,n_buffer, n_buffer)) 
    lon_grid = np.zeros((n_buffer, n_buffer)) 
    lat_grid = np.zeros((n_buffer, n_buffer))

    x_out = grid_key['x'].unique() 
    y_out = grid_key['y'].unique() 
    
    ## loop over x_out and y_out 
    ## based on cell_id
    ## create multi-dimensional output grid 
    ## with on each layer a variable 
    for j, y_cell in enumerate(y_out):
        for i, x_cell in enumerate(x_out):
            
            ## identify cell id to get data 
            cell_id = grid_key[ (grid_key['x'] == x_cell) & (grid_key['y'] == y_cell) ] 
                        
            ## get row of data belonging to each cell 
            df_data = df.loc[cell_id.index] 

            ## first layer is id of cell 
            fill_grid[0,j,i] = '{}{}'.format( int(i+1),int(j+1) ) 
            
            ## fill remaining layers with data variables 
            fill_grid[1:,j,i] = df_data.values  
            
            ## save lat and lon in separate grids 
            lon_grid[j,i] = cell_id['lon'].values[0]
            lat_grid[j,i] = cell_id['lat'].values[0]
            
   
    ## transform numpy array to xarray         
    ds_out = xr.Dataset( 
        data_vars = {
            'id': ( ("y", "x"), fill_grid[0]) 
            },
        coords = {
            "y": (y_out),
            "x": (x_out),
            "lon": (("y", "x"), lon_grid),
            "lat": (("y", "x"), lat_grid)
            },
        attrs={"gauge":     gauge_id,
               "type":      data_type}) 
    
    ## add all variables        
    for n, variable in enumerate(df_vars):
        n_layer = int(n+1) 
        ds_out[variable] = ( ("y", "x"), fill_grid[n_layer])
    
    print(' [FINISH] translate dataframe to netcdf')        
    return ds_out 


def extract_ncfile(df_loc, nc_file, feature_name, ds_parameter = 'dem_mean'):
    
    ## open dataset 
    ds = xr.open_dataset(nc_file)
    
    ## get x and y values     
    search_x = df_loc['x'].unique()
    search_y = df_loc['y'].unique() 
    
    ## create a searching area  
    mask_x = ( ds['x'] >= min(search_x) ) & ( ds['x'] <= max(search_x) )
    mask_y = ( ds['y'] >= min(search_y) ) & ( ds['y'] <= max(search_y) ) 
    
    ## execute mask search 
    ds_search = ds.where( mask_x & mask_y, drop=True) 
    
    ## convert data to dataframe 
    df_search = ds_search.to_dataframe().reset_index()
    
    ## loop through search dataframe
    ## find correct values by matching x and y 
    for ix in df_search.index:
        search_result = df_search.loc[ix]
                        
        ## assign value 
        df_loc.loc[ (df_loc['x']  == search_result['x']) &
                    (df_loc['y']  == search_result['y']),
                    feature_name ] = search_result[ds_parameter] 
    return df_loc 


def pa_calc_signatures(gauge_id, input_dir, obs_dir, gauge_fn, var='dis24'): 
    
    '''
    Function that takes data of a single gauge and observations and prepares
    signature calculations 
    
    INPUT 
        gauge_id    ID matching with a GRDC gauge 
                     
        input_dir   directory containing simulation data 
        
        obs_dir     directory containing gauge observation data 
        
        gauge_fn    file containing gauge metadata 
        
        var         name of variable in simulation data 
    
    OUTPUT
        1 / -1      If succesful, returns 1, else -1 
    
    '''
    
    fn_sim = input_dir / "buffer_{}_size-4.nc".format(gauge_id) 
    fn_obs = obs_dir / '{}_Q_Day.Cmd.txt'.format(gauge_id) 
    
        
    if fn_sim.exists() and fn_obs.exists() and gauge_fn.exists(): 
        
        df_gauge_meta = pd.read_csv(gauge_fn, index_col=0) 
        df_gauge_meta = df_gauge_meta.loc[gauge_id] 
        
        ## open simulation dataset 
        ds = xr.open_dataset(fn_sim) 
        df_sim = ds.to_dataframe().reset_index() 
        
        ## reshape to dataframe 
        df = pd.DataFrame() 
        df['date'] = pd.to_datetime( df_sim['time'].unique() )
        df = df.set_index('date') 
        
        ## key between assigned cell_id and location in grid
        df_key = pd.DataFrame() 
        
        ## shape 
        nx = df_sim['x'].nunique()
        ny = df_sim['y'].nunique() 
        
        x_center = int( nx / 2) 
        y_center = int( ny / 2)        
        
        for i, x_cell in enumerate(df_sim['x'].unique()):
            
            for j, y_cell in enumerate(df_sim['y'].unique()):
                
                _df = df_sim[ (df_sim['x'] == x_cell) & (df_sim['y'] == y_cell)] 
                                
                cell_id = '{}_{}{}'.format( gauge_id, int(i+1), int(j+1))
                
                time_ix = pd.to_datetime( _df['time'] )
                df.loc[time_ix, cell_id] = _df[var].values 
                
                ## save x,y and cell id 
                df_key.loc[cell_id, ['x', 'y']] = x_cell, y_cell 
                
                ## save lat lon 
                lat_cell = _df['lat'].unique()[0] 
                lon_cell = _df['lon'].unique()[0]
                df_key.loc[cell_id, ['lat', 'lon']] = lat_cell, lon_cell 
                
                ## save cell upArea
                df_key.loc[cell_id, 'upArea'] = _df['upArea'].unique()[0] 
                
                ## save distance to center cell 
                df_key.loc[cell_id, 'n_buffer'] = max(  abs(x_center - i) ,  abs(y_center - j) )


        ## read gauge data 
        df_gauge, meta = read_gauge_data([fn_obs]) 
        df_obs = df_gauge[(df_gauge['date'] >= '1991') &  (df_gauge['date'] < '2021')].copy()
        
        ## get gauge date range 
        date_range = pd.to_datetime( df_obs['date'] ) 
        df_obs = df_obs.set_index(date_range) 
        
        ## add gauge observations to dataframe for calculations 
        gauge_idx = '{}_gauge'.format(gauge_id)
        df[gauge_idx] = df_obs['value'] 
        
        ## drop values based on missing values in gauge observations         
        df = df.dropna(subset=[gauge_idx]) 
                
        ## calc signatures 
        df_signatures = calc_signatures( df, gauge_idx,
                                        time_window = ['all', 'seasonal'],
                                        
                                        ## new features
                                        features = ['peak-distr', 
                                                    'conc-index',
                                                    'fdc-hv', 'fdc-lv',
                                                    'lf-stat', 'hf-stat'])
                                                    
        # not viable
        # 'order-nr', 'upA-match'
                                 
        ## add buffer distance to center (for later filtering of distances) 
        df_signatures['n_buffer'] = 0 
        df_signatures.loc[df_key.index, 'n_buffer'] = df_key['n_buffer'] 
        
        ## also add coordinates 
        coord_pars = ['x', 'y', 'lat', 'lon']
        for parameter in coord_pars:
            df_signatures.loc[gauge_idx, parameter] = df_obs[parameter].unique()[0]
            df_signatures.loc[df_key.index, parameter] = df_key[parameter]
        
        #### add upArea 
        ## also a static file to extract simulations
        ## upArea exists 
        # upArea_file = input_dir/ 'EFAS_upArea4.0.nc'
        # print('A: ', upArea_file.exists())
        cell_upArea = df_key['upArea']                         ## in m2             
        gauge_upArea = df_gauge['upArea'].unique()[0] * 10**6  ## from km2 to m2
        
        df_signatures.loc[ gauge_idx, 'upArea'] = gauge_upArea 
        df_signatures.loc[ cell_upArea.index, 'upArea'] = cell_upArea 

        
    
        ## add elevation values 
        dem_file = input_dir / 'EFAS_dem.nc'
        if dem_file.exists(): 
            
            cell_elevation = extract_ncfile(df_key, dem_file, 'elevation')['elevation'] 
            gauge_elevation = df_gauge['elevation'].unique()[0] 
            
            df_signatures.loc[ gauge_idx, 'elevation'] = gauge_elevation 
            df_signatures.loc[ cell_elevation.index, 'elevation'] = cell_elevation
               
        ## label signatures 
        df_signatures = assign_labels(df_signatures, df_key, df_gauge_meta) 
        ## rename the gauge label
        df_signatures.loc[ gauge_idx, 'target'] = np.nan 
                
        ## save signatures 
        fn_signatures = input_dir / 'signatures_{}-p2.csv'.format(gauge_id) 
        df_signatures.to_csv(fn_signatures)
        
        
        ## calculate similarity vector
        # df_similarity_vector = calc_vector(df_signatures.drop(['target'], axis=1), gauge_idx)
        ## label data - identify match
        # df_similarity_vector = assign_labels(df_similarity_vector, df_key, df_gauge_meta) 
        ## save labelled data 
        # fn_similarity = input_dir / 'vector_similarity_{}.csv'.format(gauge_id) 
        # df_similarity_vector.to_csv(fn_similarity)
        
        ## RESHAPE DATA 
        ## FROM DATAFRAME TO NETCDF
        
        ## drop coordinate data 
        # df_similarity_vector = df_similarity_vector.drop(coord_pars, axis=1)
        # df_signatures = df_signatures.drop(coord_pars, axis=1)
        
        ## reshape output of similarity vector to xarray  
        # ds_similarity = df_to_ds(df_similarity_vector, df_key,
                                 # gauge_id, 'similarity vector') 
        ## save output
        # fn_similarity_nc = input_dir / 'vector_similarity_{}.nc'.format(gauge_id) 
        # ds_similarity.to_netcdf(fn_similarity_nc)
        
        ## also output of signatures in grid 
        ## without gauge signature data 
        # df_signatures = df_signatures.iloc[:-1].copy() 
        
        ## reshape data to grid 
        # ds_signatures = df_to_ds(df_signatures, df_key,
        #                           gauge_id, 'signatures')

        ## save output
        # fn_signatures_nc = input_dir / 'signatures_{}.nc'.format(gauge_id)
        # ds_signatures.to_netcdf( fn_signatures_nc)
        return 1 

    else:
        
        print('[ERROR] files not found, skip') 
        return -1



