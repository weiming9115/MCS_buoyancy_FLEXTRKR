import os
import sys
import time
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

def buoy_calc(year, month):

    # 1. 0.25-deg, 3-hourly buoyancy measure: BL = BL,cape - BL,subsat
    buoy_files = list(buoy_dir.glob('era5_2layers_thetae_{}_{}_*.nc'.format(year,str(month).zfill(2))))
    data_buoy = xr.open_mfdataset(buoy_files).sel(lat=slice(-30,30))

    data_sp = xr.open_dataset(era5_dir / 'era-5.sp.{}.{}.nc'.format(year,str(month).zfill(2))).sel(latitude=slice(30
    ,-30)).SP/100 # hPa
    data_sp = data_sp.reindex(latitude=sorted(list(data_sp.latitude))) # fliping latitude order: -30 to 30
    data_sp = data_sp.interp(latitude=data_buoy.lat.values, longitude=data_buoy.lon.values)
    data_sp = data_sp.rename({'latitude': 'lat', 'longitude': 'lon'})
    sp = data_sp.sel(time=data_buoy.time)

    thetae_bl = data_buoy.thetae_bl
    thetae_sat_lt = data_buoy.thetae_sat_lt
    thetae_lt = data_buoy.thetae_lt

    delta_pl=sp-100-500 # top at 500hPa
    delta_pb=100
    wb=(delta_pb/delta_pl)*np.log((delta_pl+delta_pb)/delta_pb)
    wl=1-wb

    Buoy_CAPE = wb * ((thetae_bl-thetae_sat_lt)/thetae_sat_lt) * 340
    Buoy_SUBSAT = wl * ((thetae_sat_lt-thetae_lt)/thetae_sat_lt) * 340
    Buoy_TOT = Buoy_CAPE - Buoy_SUBSAT # degree Kelvin (K)

    return Buoy_TOT

if __name__ == '__main__':

    start_time = time.time()

    year = sys.argv[1]
    buoycri_dir = Path('/scratch/wmtsai/temp_mcs/output_stats')
    mask_dir = Path('/neelin2020/mcs_flextrkr/{}'.format(year))
    buoy_dir = Path('/neelin2020/ERA-5_buoy/layer_thetae/')
    era5_dir = Path('/neelin2020/ERA-5/NC_FILES/{}'.format(year))

    ds_month_list = []
    for month in range(1,13):
        print('processing month: {}'.format(month))

        # calculate buoyancy measure
        ds_buoy = buoy_calc(year, month)
        # get 3-hourly MCS mask on the coordinated grids with buoy in a specific month
        mcscounts = np.zeros((ds_buoy.shape[1], ds_buoy.shape[2]))
        buoy_value = np.copy(mcscounts)

        total_time = 0
        for time_sel in ds_buoy.time.values:

            year = str(time_sel)[:4]
            month = str(time_sel)[5:7]
            day = str(time_sel)[8:10]
            hour = str(time_sel)[11:13]

            try:
            
                ds_mcs = xr.open_dataset(mask_dir / 'mcstrack_{}{}{}_{}30.nc'.format(year,month,day,hour))
                buoy = ds_buoy.sel(time=time_sel)
            
                # regrid flextrkr MCS mask
                cldmask = ds_mcs.cloudtracknumber_nomergesplit.isel(time=0).sel(lat=slice(-30,30))
                cldmask['lon'] = cldmask.lon.where(cldmask.lon > 0, cldmask.lon + 360)
                cldmask= cldmask.reindex(lon = sorted(cldmask.lon))
                cldmask_regrid = cldmask.interp(lon = ds_buoy.lon, lat = ds_buoy.lat)
                cldmask_regrid = cldmask_regrid.where(cldmask_regrid > 0, 0) 
                cldmask_regrid = cldmask_regrid.where(cldmask_regrid == 0 ,1)
                
                # counts when local BL > -5K 
                mcscounts += cldmask_regrid.values
                buoy_value += buoy.where(cldmask_regrid == 1, 0).values
                total_time += 1

            except:

                print('failed date: {}'.format(str(time_sel)[:13]))
        
        ds_month = xr.Dataset(data_vars=dict(mcscounts=(['lat','lon'], mcscounts),
                                             buoy_value=(['lat','lon'], buoy_value),
                                            ),
                            coords=dict(lat=(['lat'], buoy.lat.values),
                                        lon=(['lon'], buoy.lon.values),
                                        total_time = total_time),
                            attrs=dict(description = 'BL summation when MCS exists. 3-hrly MCS mask that matches 3-hourly BL data.'))
        ds_month_list.append(ds_month)

    # merge all months 
    ds_month_merged = xr.concat(ds_month_list, pd.Index(np.arange(1,13),name='month'))
    out_dir = Path('/scratch/wmtsai/temp_mcs/output_stats/MCSbouy_counts_n5K') 
    ds_month_merged.to_netcdf(out_dir / 'MCSbuoy_BLvalue_{}.nc'.format(year))

    end_time = time.time()
    print('execution time: {} secs'.format(time.time() - start_time))
