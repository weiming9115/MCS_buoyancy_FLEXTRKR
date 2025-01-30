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

BL_dir = Path('/neelin2020/ERA-5_buoy/layer_thetae')
data_temp = xr.open_dataset(BL_dir / 'era5_2layers_thetae_2008_06_19.nc').sel(lat=slice(-30,30))
lon_re = data_temp.lon.values
lat_re = data_temp.lat.values

# landsea mask 0.25-deg
data_ls = xr.open_dataset('/neelin2020/RGMA_feature_mask/ERA5_LandSeaMask_regrid.nc4')
lsmask = data_ls.landseamask.interp(longitude=lon_re, latitude=lat_re)
lsmask = lsmask.rename({'longitude':'lon', 'latitude':'lat'})

# processing data writeout
surface_type = ['land','ocean']
conv_type = ['nonMCS','MCS']
bins_tot = np.arange(-30,10,0.2)
bins_samples = np.zeros((2,2,len(bins_tot)-1)) # (surface_type, conv_type, bins_BL)
prec_gpm_sum = np.copy(bins_samples)

year_list = int(sys.argv[1])

for year in [year_list]:

    print('processing year: {}'.format(year))
    buoy_dir = Path('/neelin2020/ERA-5_buoy/layer_thetae/')
    era5_dir = Path('/neelin2020/ERA-5/NC_FILES/{}'.format(year))
    mcs_dir = Path('/neelin2020/RGMA_feature_mask/data_product/{}/MERGED_FP'.format(year))
    tb_dir = Path('/neelin2020/RGMA_feature_mask/data_product/{}/MERGE-IR'.format(year))

    for month in np.arange(1,13):
        print('month: {}'.format(str(month).zfill(2)))
        
        # 1.using RGMA 6-hrly dataset to avoid complex processing steps
        data_mcsmask = xr.open_dataset(mcs_dir / 'GPM_feature_merged_{}_v4.nc'.format(str(month).zfill(2))).sel(latitude=slice(-30,30))
        data_mcsmask = data_mcsmask.interp(longitude = lon_re, latitude = lat_re) # regridding into ERA5 coordinates
        data_mcsmask = data_mcsmask.rename({'longitude': 'lon', 'latitude': 'lat'})
        
        prec_gpm = data_mcsmask.precipitationCal.sel(time=data_mcsmask.time)
        mcsmask = data_mcsmask.mcs_tag.sel(time=data_mcsmask.time)
        
        # 2. Tb hourly
        data_tb = xr.open_dataset(tb_dir / 'Tb_MERGE_IR_{}_{}_hrly.compress.nc'.format(year,str(month).zfill(2))).sel(lat=slice(-30,30))
        tb = data_tb.tb.sel(time=data_mcsmask.time, method='nearest') # some floats in the data not exactly at .00
        tb['time'] = data_mcsmask.time.values # replace time values to match the time coordinate format
        tb = tb.interp(lon = lon_re, lat = lat_re)
        
        # 3. 0.25-deg, 3-hourly buoyancy measure: BL = BL,cape - BL,subsat
        buoy_files = sorted(list(buoy_dir.glob('era5_2layers_thetae_{}_{}_*.nc'.format(year,str(month).zfill(2)))))
        data_buoy = xr.open_mfdataset(buoy_files).sel(lat=slice(-30,30))

        data_sp = xr.open_dataset(era5_dir / 'era-5.sp.{}.{}.nc'.format(year,str(month).zfill(2))).sel(latitude=slice(30
        ,-30)).SP/100 # hPa
        data_sp = data_sp.reindex(latitude=sorted(list(data_sp.latitude))) # fliping latitude order: -30 to 30
        data_sp = data_sp.interp(latitude=lat_re, longitude=lon_re)
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
        
        # select time accordingly 
        Buoy_TOT = Buoy_TOT.sel(time=data_mcsmask.time)  
       
        # for non-MCS type systems (cloud < 241 K), ocean
        prec_gpm_nonmcs_oce = prec_gpm.where((tb < 241) & (mcsmask == 0) & (lsmask == 100))
        buoy_nonmcs_oce = Buoy_TOT.where((tb < 241)  & (mcsmask == 0) & (lsmask == 100))
        # for MCS type systems, ocean
        prec_gpm_mcs_oce = prec_gpm.where((mcsmask > 0) & (lsmask == 100))
        buoy_mcs_oce = Buoy_TOT.where((mcsmask > 0) & (lsmask == 100))

        BL_1d_nonmcs_oce = buoy_nonmcs_oce.values.ravel()
        prec_gpm_1d_nonmcs_oce = prec_gpm_nonmcs_oce.values.ravel()
        BL_1d_mcs_oce = buoy_mcs_oce.values.ravel()
        prec_gpm_1d_mcs_oce = prec_gpm_mcs_oce.values.ravel()

        # for non-MCS type systems (cloud < 241 K), land
        prec_gpm_nonmcs_land = prec_gpm.where((tb < 241) & (mcsmask == 0) & (lsmask < 100))
        buoy_nonmcs_land = Buoy_TOT.where((tb < 241)  & (mcsmask == 0) & (lsmask < 100))
        # for MCS type systems, land
        prec_gpm_mcs_land = prec_gpm.where((mcsmask > 0) & (lsmask < 100))
        buoy_mcs_land = Buoy_TOT.where((mcsmask > 0) & (lsmask < 100))

        BL_1d_nonmcs_land = buoy_nonmcs_land.values.ravel()
        prec_gpm_1d_nonmcs_land = prec_gpm_nonmcs_land.values.ravel()
        BL_1d_mcs_land = buoy_mcs_land.values.ravel()
        prec_gpm_1d_mcs_land = prec_gpm_mcs_land.values.ravel()

        for i in range(len(bins_tot)-1):

            # land
            idx = np.where(np.logical_and(BL_1d_nonmcs_land >= bins_tot[i], BL_1d_nonmcs_land < bins_tot[i+1]))[0]
            bins_samples[0,0,i] += len(idx)
            prec_gpm_sum[0,0,i] += np.sum(prec_gpm_1d_nonmcs_land[idx])
            
            idx = np.where(np.logical_and(BL_1d_mcs_land >= bins_tot[i], BL_1d_mcs_land < bins_tot[i+1]))[0]
            bins_samples[0,1,i] += len(idx)
            prec_gpm_sum[0,1,i] += np.sum(prec_gpm_1d_mcs_land[idx])
            
            # ocean
            idx = np.where(np.logical_and(BL_1d_nonmcs_oce >= bins_tot[i], BL_1d_nonmcs_oce < bins_tot[i+1]))[0]
            bins_samples[1,0,i] += len(idx)
            prec_gpm_sum[1,0,i] += np.sum(prec_gpm_1d_nonmcs_oce[idx])
            
            idx = np.where(np.logical_and(BL_1d_mcs_oce >= bins_tot[i], BL_1d_mcs_oce < bins_tot[i+1]))[0]
            bins_samples[1,1,i] += len(idx)
            prec_gpm_sum[1,1,i] += np.sum(prec_gpm_1d_mcs_oce[idx])

# writeout data
ds = xr.Dataset(data_vars = dict(samples = (['surface_type','conv_type','bins_BL'], bins_samples),
                                 prec_gpm_sum = (['surface_type','conv_type','bins_BL'], prec_gpm_sum)),
                coords = dict(surface_type = (['surface_type'], surface_type),
                              conv_type = (['conv_type'], conv_type),
                              bins_BL = (['bins_BL'], bins_tot[:-1])),
                attrs = dict(description = 'buoy-precipitation relationship in tropical regions. MCS vs Non-MCS (Tb < 241K)',
                             bins_unit = 'degree Kelvin')
               )

out_dir = Path('/scratch/wmtsai/temp_mcs/output_stats')
ds.to_netcdf(out_dir / 'hist1d_BL_precip_nonMCSvsMCS.{}.nc'.format(year_list))

