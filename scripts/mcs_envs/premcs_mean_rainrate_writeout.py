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

import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.util import add_cyclic_point
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# for t-test precip. mean
from scipy.stats import ttest_ind

import warnings

warnings.filterwarnings('ignore')

dir_mcs_track = Path('/neelin2020/mcs_flextrkr/mcs_stats/')
dir_era5 = Path('/neelin2020/ERA-5/NC_FILES/')

year = int(sys.argv[1])
data_non2mcs_complete = xr.open_dataset(dir_mcs_track / 'mcs_tracks_non2mcs_{}.IndoPacific.nc'.format(year))

ds_all_list = []
                    
for n,track_number in enumerate(data_non2mcs_complete.tracks.values):
       
    print('processing track number: ({} / {})'.format(n, len(data_non2mcs_complete.tracks.values)))
          
    data = data_non2mcs_complete.sel(tracks=track_number, times=0)
                     
    start_basetime = data_non2mcs_complete.sel(tracks=track_number).start_basetime.values
    timestamp_str = str(start_basetime)
    year = timestamp_str[:4]
    month = timestamp_str[5:7]
    day = timestamp_str[8:10]
    hour = timestamp_str[11:13]
                     
    # get geolocation info at times = 0 
    meanlon = data.meanlon
    if meanlon < 180:
        meanlon_era5 = meanlon
    elif meanlon < -180:
        meanlon_era5 = meanlon + 360
    meanlat = data.meanlat
                     
    # creat timestamps up to 23 hours prior to the CCS detection
    datetime_list = pd.date_range(start=datetime(int(year),int(month),int(day),int(hour)), periods=24, freq='-1H')
    # flip list for proceeding times
    datetime_list = np.flip(datetime_list)
   
    mean_mpr_list = []
    mean_gpm_list = []
    for timestamp in datetime_list:
        
        yr = timestamp.year
        mon = timestamp.month
        dd = timestamp.day
        hh = timestamp.hour
        
        # get era5 & GPM precip [0-359.75]
        data_mpr = xr.open_dataset(dir_era5 / '{}/era-5.mpr.{}.{}.nc'.format(year, year, str(mon).zfill(2)))
        data_mpr = data_mpr.reindex(latitude=list(reversed(data_mpr.latitude)))
        data_gpm = xr.open_dataset('/neelin2020/RGMA_feature_mask/GPM_ncfiles_{}/GPM_IMERGE_V06_{}{}{}_{}00.nc'.format(
                   yr, yr, str(mon).zfill(2), str(dd).zfill(2), str(hh).zfill(2)))                     
    
        # mean of a 6-deg box 
        # in a unit of (mm/hr)
        mean_mpr = 3600*data_mpr.mtpr.sel(time=timestamp, method='nearest').sel(longitude=slice(meanlon_era5-3,meanlon_era5+3)
                                                           ,latitude=slice(meanlat-3,meanlat+3)).mean()
        mean_gpm = data_gpm.precipitationCal.sel(time=timestamp, method='nearest').sel(lon=slice(meanlon-3,meanlon+3)
                                                           ,lat=slice(meanlat-3,meanlat+3)).mean()
        mean_mpr_list.append(mean_mpr)                 
        mean_gpm_list.append(mean_gpm)                 
    
    mean_mpr_xr = xr.concat(mean_mpr_list, dim=pd.Index(np.arange(-23,1), name='rel_times')).drop('time')
    mean_gpm_xr = xr.concat(mean_gpm_list, dim=pd.Index(np.arange(-23,1), name='rel_times')).drop('time')
    ds_single_xr = xr.merge([mean_mpr_xr, mean_gpm_xr])
                                   
    ds_all_list.append(ds_single_xr)
                                   
ds_all_xr = xr.concat(ds_all_list, dim=pd.Index(data_non2mcs_complete.tracks.values, name='tracks'))

# save mean rainrate data 
dir_out = Path('/neelin2020/mcs_flextrkr/era5_envs')
ds_all_xr.to_netcdf(dir_out / 'mcs_era5_non2mcs_rainrate_{}.IndoPacific.nc'.format(year))
