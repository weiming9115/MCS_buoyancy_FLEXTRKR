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

######################################################
year_list = np.arange(2002,2015)

BL_dir = Path('/neelin2020/ERA-5_buoy/layer_thetae')
data_temp = xr.open_dataset(BL_dir / 'era5_2layers_thetae_2008_06_19.nc').sel(lat=slice(-30,30))
lon_re = data_temp.lon
lat_re = data_temp.lat
counts_map = np.zeros((len(year_list), 12, len(lat_re), len(lon_re)))

for y,year in enumerate(year_list):

    print('processing year: {}'.format(year))
    era5_dir = Path('/neelin2020/ERA-5/NC_FILES/{}'.format(year))

    for m,month in enumerate(np.arange(1,13)):
        print('month: {}'.format(month))        

        # read file as mfdataset 
        data_sp = xr.open_dataset(era5_dir / 'era-5.sp.{}.{}.nc'.format(year,str(month).zfill(2))).sel(latitude=slice(30,-30)).SP/100 # hPa
        data_sp = data_sp.reindex(latitude=sorted(list(data_sp.latitude))) # fliping latitude order: -30 to 30
        data_sp = data_sp.interp(latitude=lat_re.values, longitude=lon_re.values)
        data_sp = data_sp.rename({'latitude': 'lat', 'longitude': 'lon'})

        # 3-hourly buoyancy data
        files_theta = sorted(list(BL_dir.glob('era5_2layers_thetae_{}_{}_*.nc'.format(year,str(month).zfill(2)))))
        data = xr.open_mfdataset(files_theta).sel(lat=slice(-30,30))
        sp = data_sp.sel(time=data.time)

        thetae_bl = data.thetae_bl
        thetae_sat_lt = data.thetae_sat_lt
        thetae_lt = data.thetae_lt

        delta_pl=sp-100-500 # top at 500hPa
        delta_pb=100
        wb=(delta_pb/delta_pl)*np.log((delta_pl+delta_pb)/delta_pb)
        wl=1-wb

        # calculate buoyancy estimate
        Buoy_CAPE = wb * ((thetae_bl-thetae_sat_lt)/thetae_sat_lt) * 340
        Buoy_SUBSAT = wl * ((thetae_sat_lt-thetae_lt)/thetae_sat_lt) * 340
        Buoy_TOT = Buoy_CAPE - Buoy_SUBSAT # degree Kelvin (K)

        Buoy_TOT_mask = Buoy_TOT.where(Buoy_TOT > -2.5, 0)
        Buoy_TOT_binary = Buoy_TOT_mask.where(Buoy_TOT_mask == 0, 1) # replace values by 1 

        counts_map[y,m,:,:] += Buoy_TOT_binary.sum('time').values

ds_exceedBLfreq = xr.Dataset(data_vars = dict(counts = (['year','month','lat','lon'], counts_map)),
                             coords = dict(year = (['year'], year_list),
                                           month = (['month'], np.arange(1,13)),
                                           lon = (['lon'], lon_re.values),
                                           lat = (['lat'], lat_re.values)),
                             attrs = dict(description = 'counts of BL values exceeding the defined value',
                                          source = '/neelin2020/ERA-5_buoy/',
                                          wlwb_weight = 'Yes',
                                          BL_cri = '-2.5 (degree Kelvin)')
                            )

out_dir = Path('/scratch/wmtsai/temp_mcs/output_stats/')
ds_exceedBLfreq.to_netcdf(out_dir / 'counts_map_BLcritval_3hrly.2002.2014.n25K.nc')
