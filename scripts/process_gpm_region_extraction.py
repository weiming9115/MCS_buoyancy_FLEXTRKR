import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

def coordinates_processors(data):
    """
    converting longitude/latitude into lon/lat
    data: xarray.dataset coordinated horizontally in lat/lon
    """

    coord_names = []
    for coord_name in data.coords:
        coord_names.append(coord_name)

    if (set(coord_names) & set(['lon','lat'])): # if coordinates set this way...

        data2 = data.rename({'lat': 'latitude'})
        data2 = data2.rename({'lon': 'longitude'})

    else:
        data2 = data

    # check if lon from -180
    if data2.longitude[0] != 0: # -180 to 180

        lon_reset = data2.longitude
        lon_reset = lon_reset.where(lon_reset > 0, 360+lon_reset) # converting lon as 0 to 359.75
        data2.coords['longitude'] = lon_reset # converting lon as -180 to 180
        data2= data2.sortby('longitude')

    # check if latitutde is decreasing
    if (data2.latitude[1] - data2.latitude[0]) < 0:
        data2 = data2.isel(latitude=slice(None, None, -1)) # flipping latitude accoordingly

    return data2

def process_gpm_subdomain(i):
    
    data = xr.open_dataset(files[i])
    data_update = coordinates_processors(data)
    data_AMAZON = data_update.sel(longitude=slice(-85+360,-30+360), latitude=slice(-22,8))
    # save into the directory
    data_AMAZON.to_netcdf(out_dir / '{}.AMAZON.nc'.format(files[i].name[:-3]))

if __name__ == '__main__' :

    # load gpm precip
    year = sys.argv[1]

    gpm_dir = Path('/neelin2020/RGMA_feature_mask/GPM_ncfiles_{}'.format(year))
    files = sorted(list(gpm_dir.glob('*.nc')))

    out_dir = Path('/neelin2020/mcs_flextrkr/gpm_AMAZON/{}'.format(year))
    if out_dir.exists() == False:
        print('create folder: {}'.format(year))
        os.system('mkdir {}'.format(out_dir))
  
    from multiprocessing import Pool

    pool = Pool(processes=10)
    results = pool.map(process_gpm_subdomain, range(len(files)))
    pool.close()
    pool.join()


