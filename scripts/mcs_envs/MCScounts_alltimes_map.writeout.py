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

from scipy.stats import linregress

import warnings
warnings.filterwarnings('ignore')

BL_dir = Path('/neelin2020/ERA-5_buoy/layer_thetae')
data_temp = xr.open_dataset(BL_dir / 'era5_2layers_thetae_2008_06_19.nc').sel(lat=slice(-30,30))
lon_re = data_temp.lon
lat_re = data_temp.lat

year = int(sys.argv[1]) # processing year
MCScounts_map = np.zeros((12, len(lat_re), len(lon_re)))

print('processing year: {}'.format(year))
    
#env_dir = Path('/scratch/wmtsai/temp_mcs/mcs_stats/envs_track/{}/tropics_extend'.format(year))
stats_dir = Path('/neelin2020/mcs_flextrkr/{}0101.0000_{}0101.0000'.format(year,year+1))
  
for m,month in enumerate(np.arange(1,13)):
    print('month: {}'.format(month))
    files = list(stats_dir.glob('mcstrack_{}{}*.nc'.format(year,str(month).zfill(2))))
    #for file in files:
    data_mask = xr.open_mfdataset(files)
    cloudmask = data_mask.cloudtracknumber_nomergesplit.sel(lat=slice(-30,30))
    # generate binary mask 0 or 1
    cloudmask = cloudmask.where(cloudmask > 0, 0)
    cloudmask = cloudmask.where(cloudmask == 0, 1)
    cloudmask['lon'] = cloudmask.lon.where(cloudmask.lon > 0, cloudmask.lon + 360)
    cloudmask = cloudmask.reindex(lon = sorted(cloudmask.lon))
    cloudmask = cloudmask.interp(lon = lon_re, lat = lat_re)
    MCScounts_map[m,:,:] += cloudmask.sum('time').astype('int').values
       
# writeout as dataset            
ds_MCSfreq = xr.Dataset(data_vars = dict(counts = (['month','lat','lon'], MCScounts_map)),
                        coords = dict(month = (['month'], np.arange(1,13)),
                                      lon = (['lon'], lon_re.values),
                                      lat = (['lat'], lat_re.values)),
                        attrs = dict(description = 'MCS frequency',
                                     mcs_type = 'MCS masks. All tracks',
                                     temporal_base = 'hourly'
                                     )
                       )

out_dir = Path('/scratch/wmtsai/temp_mcs/output_stats/MCScounts_map/')
ds_MCSfreq.to_netcdf(out_dir / 'counts_map_MCSfreq_hrly.{}.nc'.format(year))
print('counts_map_MCSfreq_hrly.{}.nc .... done'.format(year))
