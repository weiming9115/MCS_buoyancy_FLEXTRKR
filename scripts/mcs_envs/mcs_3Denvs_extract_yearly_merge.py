import os
import sys
import xarray as xr
import numpy as np
import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
from pathlib import Path
import cartopy.crs as ccrs
import warnings

warnings.filterwarnings('ignore')

year = sys.argv[1]

out_dir = Path('/scratch/wmtsai/temp_mcs/mcs_stats/envs_track/{}/'.format(year))
mcsenv_dir = Path('/scratch/wmtsai/temp_mcs/mcs_stats/envs_track/{}/tropics_extend/'.format(year))

files = sorted(list(mcsenv_dir.glob('*.nc')))
track_list = []
for file in files:
    track_list.extend([int(str(file.name)[-11:-6])])
print('total number of tracks: {}'.format(len(track_list)))
        
ds_merged = []
for n,file in enumerate(files):
    tmp = xr.open_dataset(file)
    ds_merged.append(tmp)
ds_merged_xr = xr.concat(ds_merged, dim=pd.Index(track_list, name='tracks'))
ds_merged_xr.to_netcdf(out_dir / 'mcs_era5_3D_envs_{}.alltracks.nc'.format(year))

