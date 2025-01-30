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

def write_histogram_regions(Buoy_TOT, cwv, bins_bl, geo_info):
    """
    write out histogram of BL values by the given bins
    """
    buoy_samples = np.zeros(len(bins_bl)-1)
    cwv_sum = np.copy(buoy_samples)
    
    Buoy_sub = Buoy_TOT.sel(lat=slice(geo_info[0],geo_info[1])
                                , lon=slice(geo_info[2],geo_info[3]))
    cwv_sub = cwv.sel(lat=slice(geo_info[0],geo_info[1])
                                , lon=slice(geo_info[2],geo_info[3]))
    
    # get 1-D BL values over the specified region
    BL_1d = Buoy_sub.values.ravel()
    cwv_1d = cwv_sub.values.ravel()

    for n in range(len(bins_bl)-1):
        idx = np.where(np.logical_and(BL_1d >= bins_bl[n], BL_1d < bins_bl[n+1]))[0]
        buoy_samples[n] = len(idx)
        cwv_sum[n] = np.sum(cwv_1d[idx])
    
    return (buoy_samples, cwv_sum)

if __name__ == '__main__':

    # defined tropical regions
    WPC = [-10,5,130,170]
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
    bins_bl = np.arange(-30,10,0.2) # degree Kelvin
    buoy_samples = np.zeros((7,len(bins_bl)-1)) # 7 tropical regions
    cwv_sum = np.copy(buoy_samples)

    year_list = np.arange(2002,2006)

    for year in year_list:

        print('processing year: {}'.format(year))
        buoy_dir = Path('/neelin2020/ERA-5_buoy/layer_thetae/')
        era5_dir = Path('/neelin2020/ERA-5/NC_FILES/{}'.format(year))
        cwv_dir = Path('/neelin2020/ERA-5_cwv_That')

        for month in np.arange(1,13):
            print('month: {}'.format(str(month).zfill(2)))

            # 1. 0.25-deg, 3-hourly buoyancy measure: BL = BL,cape - BL,subsat
            buoy_files = list(buoy_dir.glob('era5_2layers_thetae_{}_{}_*.nc'.format(year,str(month).zfill(2))))
            data_buoy = xr.open_mfdataset(buoy_files).sel(lat=slice(-30,30))

            data_sp = xr.open_dataset(era5_dir / 'era-5.sp.{}.{}.nc'.format(year,str(month).zfill(2))).sel(latitude=slice(30
            ,-30)).SP/100 # hPa
            data_sp = data_sp.reindex(latitude=sorted(list(data_sp.latitude))) # fliping latitude order: -30 to 30
            data_sp = data_sp.interp(latitude=lat_re.values, longitude=lon_re.values)
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

            # 2. 0.25-deg, 3-hrly GPM-IMERG, calibrated
            cwv_files = list(cwv_dir.glob('era5_cwv_that_{}_{}_*.nc'.format(year,str(month).zfill(2))))
            data_cwv = xr.open_mfdataset(cwv_files).sel(lat=slice(-30,30))
            cwv = data_cwv.cwv

            # some missing data for cwv under /neelin2020, so match available times for both datasets
            time_match = np.intersect1d(cwv.time, Buoy_TOT.time)
            cwv = cwv.sel(time=time_match)
            Buoy_TOT = Buoy_TOT.sel(time=time_match)

            # 3. sampling by BL bins over regions
            # WPC = [lat_s, lat_n, lon_s, lon_n]
            for n, geo_info in enumerate([WPC, IND, EPC, ATL, WAF, MC, AMZ]):
                (b_samples, cwv_sub) = write_histogram_regions(Buoy_TOT
                                                             , cwv, bins_bl, geo_info)

                buoy_samples[n,:] += b_samples
                cwv_sum[n,:] += cwv_sub

    # writeout data
    ds = xr.Dataset(data_vars = dict(samples = (['region','BL_bins'], buoy_samples),
                                     cwv_sum = (['region','BL_bins'], cwv_sum)),
                    coords = dict(BL_bins = (['BL_bins'], bins_bl[:-1]),
                                  region = (['region'], ['WPC','IND','EPC','ATL','WAF','MC','AMZ'])),
                    attrs = dict(description = 'buoy-CWV relationship in tropical regions',
                                 bins_unit = 'degree Kelvin')
                   )  

    out_dir = Path('/scratch/wmtsai/temp_mcs/output_stats')
    ds.to_netcdf(out_dir / 'buoy_cwv_multiregions.nc')


