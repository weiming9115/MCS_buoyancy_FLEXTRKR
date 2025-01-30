import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from datetime import datetime
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
from matplotlib.patches import FancyArrowPatch

import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.util import add_cyclic_point
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import warnings
warnings.filterwarnings('ignore')

BL_dir = Path('/neelin2020/ERA-5_buoy/layer_thetae')
data_temp = xr.open_dataset(BL_dir / 'era5_2layers_thetae_2008_06_19.nc').sel(lat=slice(-30,30))
lon_re = data_temp.lon
lat_re = data_temp.lat

year_list = np.arange(2002,2015)
MCScounts_map = np.zeros((len(year_list),len(lat_re), len(lon_re)))

for y,year in enumerate(year_list):
    print('processing year : {}'.format(year))
    env_dir = Path('/scratch/wmtsai/temp_mcs/mcs_stats/envs_track/{}/tropics_extend'.format(year))
    stats_dir = Path('/scratch/wmtsai/temp_mcs/mcs_stats/mcs_tracks_non2mcs/')
    
    data_stats = xr.open_dataset(stats_dir / 'mcs_tracks_non2mcs_{}.tropics30NS.extend.nc'.format(year))
    
    for track in data_stats.tracks.values:
        
        # get the timestamp at the initial phase
        idt_mcs_init = data_stats.idt_mcs_init.sel(tracks=track)
        timestamp = data_stats.base_time.sel(tracks=track).isel(times=idt_mcs_init).values
        hour = str(timestamp)[11:13]
        
        if ((hour == '00') or (hour == '03') or (hour == '06') or (hour == '09') or
            (hour == '12') or (hour == '15') or (hour == '18') or (hour == '21')):
            
            try:

                data_env = xr.open_dataset(env_dir / 'mcs_era5_3D_envs_{}.{}.LD.nc'.format(year, str(track).zfill(5)))
                # get MCS grid points
                mask_init = data_env.cloudtracknumber_nomergesplit.sel(mcs_phase='Init')

                meanlon = data_stats.meanlon.sel(tracks=track).isel(times=idt_mcs_init).values # MCS center at the initial phase
                if meanlon < 0: # if negative 
                    meanlon = meanlon + 360 #(0-360)
                meanlat = data_stats.meanlat.sel(tracks=track).isel(times=idt_mcs_init).values

                # geolocate the MCS grids accordingly
                idx_meanlon = np.argmin(abs(lon_re.values - meanlon))
                idx_meanlat = np.argmin(abs(lat_re.values - meanlat))
                mask_init['x'] = lon_re.values[idx_meanlon-20:idx_meanlon+20]
                mask_init['y'] = lat_re.values[idx_meanlat-20:idx_meanlat+20]

                for lon_val in mask_init['x'].values:
                    for lat_val in mask_init['y'].values:
                        if mask_init.sel(x=lon_val, y=lat_val) == 1: # if MCS grid 
                            idx_lon_re = np.where(lon_re == lon_val)[0]
                            idx_lat_re = np.where(lat_re == lat_val)[0]

                            MCScounts_map[y,idx_lat_re, idx_lon_re] += 1
 
            except: # if something wierd happens
                
                #print('mcs_era5_3D_envs_{}.{}.LD.nc....{}'.format(year, str(track).zfill(5)
                #                                           , len(lat_re.values[idx_meanlat-20:idx_meanlat+20])))
                continue

# writeout as dataset            
ds_MCSfreq = xr.Dataset(data_vars = dict(counts = (['year','lat','lon'], MCScounts_map)),
                        coords = dict(year = (['year'], year_list),
                                      lon = (['lon'], lon_re.values),
                                      lat = (['lat'], lat_re.values)),
                        attrs = dict(description = 'MCS frequency',
                                     mcs_type = 'MCS grids at their initial phase. Non2MCS tracks',
                                     temporal_base = '3-hourly'
                                     )
                       )

out_dir = Path('/scratch/wmtsai/temp_mcs/output_stats/')
ds_MCSfreq.to_netcdf(out_dir / 'MCScounts_map_initial_3hrly.2002.2014.update.nc')
print('MCScounts_map_initial_3hrly.2002.2014.nc ...saved')
ds_MCSfreq.to_netcdf()
