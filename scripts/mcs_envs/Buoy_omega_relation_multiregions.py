import os
import sys
import xarray as xr
import numpy as np
from numpy import unravel_index
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

def write_histogram_regions(Buoy_TOT, omega, bins_bl, geo_info):
    """
    write out histogram of BL values by the given bins
    """
    buoy_samples = np.zeros(len(bins_bl)-1)
    omega_sum = np.copy(buoy_samples)
   
    Buoy_sub = Buoy_TOT.sel(lat=slice(geo_info[0],geo_info[1])
                                , lon=slice(geo_info[2],geo_info[3]))
    omega_sub = omega.sel(lat=slice(geo_info[0],geo_info[1])
                                , lon=slice(geo_info[2],geo_info[3])).w
   
    # get 1-D BL values over the specified region
    BL_1d = np.reshape(Buoy_sub.values, (len(Buoy_sub.lat)*len(Buoy_sub.lon)))
    omega_1d = np.reshape(omega_sub.values, (len(omega_sub.level),len(Buoy_sub.lat)*len(Buoy_sub.lon))) # transpose to match BL_1d (lat,lon)

    for n in range(len(bins_bl)-1):
        idx = np.where(np.logical_and(BL_1d >= bins_bl[n], BL_1d < bins_bl[n+1]))[0]
        buoy_samples[n] = len(idx)
        omega_sum[n,:] = np.sum(omega_1d[:,idx])
    
    return (buoy_samples, omega_sum)

if __name__ == '__main__':

    # defined tropical regions
    WPC = [-10,10,110,140]
    IND = [-10,5,70,90]
    EPC = [0,10,240,260]
    ATL = [0,10,320,340]
    WAF = [-10,10,0,30]
    MC  = [-7,7,95,125]
    AMZ = [-10,5,285,310]

    BL_dir = Path('/neelin2020/ERA-5_buoy/layer_thetae')
    data_temp = xr.open_dataset(BL_dir / 'era5_2layers_thetae_2008_06_19.nc').sel(lat=slice(-30,30))
    lon_re = data_temp.lon
    lat_re = data_temp.lat

    # processing data writeout
    bins_bl = np.arange(-30,10,0.25) # degree Kelvin
    buoy_samples = np.zeros((7,len(bins_bl)-1)) # 7 tropical regions
    omega_sum = np.zeros((7,27,len(bins_bl)-1)) # levels, BL bins

    year_list = np.arange(2014,2015)

    for year in year_list:

        print('processing year: {}'.format(year))
        buoy_dir = Path('/neelin2020/ERA-5_buoy/layer_thetae/')
        era5_dir = Path('/neelin2020/ERA-5/NC_FILES/{}/'.format(year))
        omega_dir = Path('/scratch/wmtsai/ERA-5/NC_FILES/{}/'.format(year))

        for month in np.arange(1,2):
            print('month: {}'.format(str(month).zfill(2)))

            # 1. 0.25-deg, 3-hourly buoyancy measure: BL = BL,cape - BL,subsat
            buoy_files = list(buoy_dir.glob('era5_2layers_thetae_{}_{}_*.nc'.format(year,str(month).zfill(2))))
            data_buoy = xr.open_dataset(buoy_files).sel(lat=slice(-30,30))

            data_sp = xr.open_dataset(era5_dir / 'era-5.sp.{}.{}.nc'.format(year,str(month).zfill(2))).sel(latitude=slice(30
            ,-30)).SP/100 # hPa
            data_sp = data_sp.reindex(latitude=sorted(list(data_sp.latitude))) # fliping latitude order: -30 to 30
            data_sp = data_sp.interp(latitude=lat_re.values, longitude=lon_re.values)
            data_sp = data_sp.rename({'latitude': 'lat', 'longitude': 'lon'})
            sp = data_sp.sel(time=data_buoy.time)

            data_omega = xr.open_dataset(omega_dir / 'era-5.omega.{}.{}.nc'.format(year,str(month).zfill(2))).sel(latitude=slice(30
            ,-30)) # pa/s
            data_omega = data_omega.reindex(latitude=sorted(list(data_omega.latitude))) # fliping latitude order: -30 to 30
            data_omega = data_omega.interp(latitude=lat_re.values, longitude=lon_re.values)
            data_omega = data_omega.rename({'latitude': 'lat', 'longitude': 'lon'})
            omega = data_omega.sel(time=data_buoy.time)

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

            # 3. sampling by BL bins over regions
            # WPC = [lat_s, lat_n, lon_s, lon_n]
            for t in Buoy_TOT.time:

                for n, geo_info in enumerate([WPC, IND, EPC, ATL, WAF, MC, AMZ]):
                    (b_samples, omega_samples_sum) = write_histogram_regions(Buoy_TOT.sel(time=t)
                                                                , omega.sel(time=t), bins_bl, geo_info)

                    buoy_samples[n,:] += b_samples
                    omega_sum[n,:,:] += omega_samples_sum

    # writeout data
    ds = xr.Dataset(data_vars = dict(samples = (['region','BL_bins'], buoy_samples),
                                     omega_sum = (['region','BL_bins'], omega_sum)),
                    coords = dict(BL_bins = (['BL_bins'], bins_bl[:-1]),
                                  region = (['region'], ['WPC','IND','EPC','ATL','WAF','MC','AMZ'])),
                    attrs = dict(description = 'buoy-omega relationship in tropical regions',
                                 bins_unit = 'degree Kelvin')
                   )  

    out_dir = Path('/scratch/wmtsai/temp_mcs/output_stats')
    ds.to_netcdf(out_dir / 'buoy_omega_multiregions.2014.test.nc')


