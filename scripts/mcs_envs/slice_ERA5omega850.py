import os
import sys
import xarray as xr
from pathlib import Path

year = sys.argv[1]
dir_era5 = Path('/scratch/wmtsai/ERA-5/NC_FILES/{}'.format(year))

for mon in range(1,13):
    ds = xr.open_dataset(dir_era5 / 'era-5.omega.{}.{}.nc'.format(year,str(mon).zfill(2)))
    ds_850 = ds.sel(level=850)
    ds_850.to_netcdf(dir_era5 / 'era-5.omega850.{}.{}.nc'.format(year,str(mon).zfill(2)))
    print('era-5.omega850.{}.{}.nc .....completed'.format(year,str(mon).zfill(2)))
