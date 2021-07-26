# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 16:21:29 2021

@author: mvand
"""

import pandas as pd  
import xarray as xr     
import time 

def extract_ts(gauge_id, gauge_df, ds_file_list, out_dir, buffer_size=2, cell_size=5000., var='dis24'):
    
    ## load EFAS data 
    ds = xr.open_mfdataset(ds_file_list, chunks = {"time": 30}) 
    
    ## get gauge coordinates in EFAS grid 
    gauge_X, gauge_Y = gauge_df.loc[gauge_id, ['proj_X', 'proj_Y']] 
    
    ## find closest cell center form (gauge_X, gauge_Y)
    center_search = ds.sel( {'x': gauge_X,
                             'y': gauge_Y},
                           method = 'nearest') 
        
    ## extract x and y from search results 
    center_cell_X = center_search.x.values 
    center_cell_Y = center_search.y.values 
    
    ## create a searching area 
    ## based on center_cell XY and buffer size 
    mask_x = ( ds['x'] >=  center_cell_X - (1.1*buffer_size*cell_size) ) & ( ds['x'] <=  center_cell_X + (1.1*buffer_size*cell_size) )
    mask_y = ( ds['y'] >= center_cell_Y - (1.1*buffer_size*cell_size) ) & ( ds['y'] <= center_cell_Y + (1.1*buffer_size*cell_size) ) 
    
    ## execute mask search 
    ds_buffer = ds.where( mask_x & mask_y, drop=True) 
    
    ## save buffer search 
    fn_buffer = out_dir / 'gauge_{}_buffer-{}.nc'.format(gauge_id, buffer_size) 
    ds_buffer.to_netcdf(fn_buffer)
    
    return fn_buffer

