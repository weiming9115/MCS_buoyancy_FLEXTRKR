import os
import sys
import time
import xarray as xr
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
start_time = time.time()

# specify the year of the FLEXTRKR cloud mask output
year = sys.argv[1] 
print('current precessing year: {}'.format(year))

freq_map = np.zeros((600, 3600)) # 0.1-deg., 30S-30N tropical region
mcs2d_dir = Path('/neelin2020/mcs_flextrkr/{}'.format(year))
files = sorted(list(mcs2d_dir.glob('*.nc')))

ds = xr.open_mfdataset(files)
ds_tropics = ds.sel(lat=slice(-30,30))
cloudmask = ds_tropics.cloudtracknumber_nomergesplit
tmp = cloudmask.where(cloudmask > 0, 0) # set as a binary map
cloudmask_binary = tmp.where(tmp == 0, 1)
freq_map[:,:] = cloudmask_binary.sum('time')  
               
# writeout as xarray dataset
ds_out = xr.Dataset(data_vars=dict(counts=(['lat','lon'], freq_map)),
                coords=dict(lat=(['lat'], ds_tropics.lat.values),
                            lon=(['lon'], ds_tropics.lon.values))
                    )
ds_out.attrs['description'] = 'MCS counts based on the origional, hourly 2-D cloud mask. Use cloudtracknumber_nomergesplit.'
ds_out.to_netcdf('/scratch/wmtsai/temp_mcs/output_stats/MCScount_geomap/MCScounts_geomap_{}.nc'.format(year))
print('MCScounts_geomap_{}.nc ...completed'.format(year))
print("--- %s seconds ---" % (time.time() - start_time))
