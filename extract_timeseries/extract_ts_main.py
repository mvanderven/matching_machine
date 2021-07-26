# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 13:52:31 2021

@author: mvand
"""

#%% Import modules 
import pandas as pd  
from pathlib import Path 
import time 

from pathos.threading import ThreadPool as Pool
from extract_ts_utils import extract_ts 

#%% Define function 

def run_parallel(fn, efas_files, input_dir, buffer_size=2):
        
    ## load gauge data 
    df = pd.read_csv(fn, index_col=0)
    
    ## get gauge IDs [TEST]
    # gauge_idx = df.index.values ss
    gauge_idx = df.sample(n=5).index.values 
      
    ## set function parameters 
    df_list = [df]*len(gauge_idx) 
    efas_list = [efas_files] * len(gauge_idx) 
    buffer_list = [buffer_size]*len(gauge_idx) 
    dir_list = [input_dir] * len(gauge_idx)
    
    print('[START] parallel processing') 
    time_parallel = time.time() 
    
    ## create pool 
    p = Pool() 
    
    ## map function in Pool 
    results = p.map(extract_ts, gauge_idx, df_list, efas_list, dir_list, buffer_list) 
    
    print('[FINISH] parallel processing in {:.2f} minutes'.format( (time.time() - time_parallel)/60. )) 
    
    return results 
 

#%% Run 

if __name__ == '__main__':
    
    ## input_dir 
    input_dir = Path(r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\data\dev_ts") 
    # input_dir = Path("/scratch-shared/mizzivdv/efas_input/")
    
    ## gauge metadata file 
    fn = input_dir / 'V1_grdc_efas_selection-cartesius.csv'    
    
    ## get all efas files 
    efas_files = [file for file in input_dir.glob('*.nc')]  
    
    ## run 
    buffer_files = run_parallel(fn, efas_files, input_dir, buffer_size=4)
    
    















