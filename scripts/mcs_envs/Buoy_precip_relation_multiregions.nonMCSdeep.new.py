import os
import sys
import xarray as xr
import numpy as np
from numpy import unravel_index
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

def write_histogram_regions(Buoy_TOT, prec_gpm, bins_bl, geo_info):
    """
    write out histogram of BL values by the given bins
    """
    buoy_samples = np.zeros(len(bins_bl)-1)
    prec_gpm_sum = np.copy(buoy_samples)
    
    Buoy_sub = Buoy_TOT.sel(lat=slice(geo_info[0],geo_info[1])
                                , lon=slice(geo_info[2],geo_info[3]))
    prec_sub = prec_gpm.sel(lat=slice(geo_info[0],geo_info[1])
                                , lon=slice(geo_info[2],geo_info[3]))
    
    # get 1-D BL values over the specified region
    BL_1d = Buoy_sub.values.ravel()
    prec_gpm_1d = prec_sub.values.ravel() 
    # deal with nan
    prec_mask = prec_gpm_1d[~np.isnan(prec_gpm_1d)]
    BL_mask = BL_1d[~np.isnan(prec_gpm_1d)]

    for n in range(len(bins_bl)-1):
        idx = np.where(np.logical_and(BL_mask >= bins_bl[n], BL_mask < bins_bl[n+1]))[0]
        buoy_samples[n] = len(idx)
        prec_gpm_sum[n] = np.sum(prec_mask[idx])
    
    return (buoy_samples, prec_gpm_sum)

if __name__ == '__main__':

    # defined tropical regions
    WPC = [-10,10,110,140]
    IND = [-10,5,70,90]
    EPC = [0,10,240,260]
    ATL = [0,10,320,340]
    WAF = [-10,10,0,30]
    MC  = [-7,7,95,125]
    AMZ = [-10,5,285,310]

    #BL_dir = Path('/neelin2020/ERA-5_buoy/layer_thetae')
    #data_temp = xr.open_dataset(BL_dir / 'era5_2layers_thetae_2008_06_19.nc').sel(lat=slice(-30,30))
    lon_re = np.arange(0,360,0.25)
    lat_re = np.arange(-30,30.25,0.25)

    # processing data writeout
    bins_bl = np.arange(-30,10,0.2) # degree Kelvin
    buoy_samples = np.zeros((7,len(bins_bl)-1)) # 7 tropical regions
    prec_gpm_sum = np.copy(buoy_samples)

    year = sys.argv[1] # input year

    print('processing year: {}'.format(year))
    buoy_dir = Path('/neelin2020/ERA-5_buoy/layer_thetae/')
    era5_dir = Path('/neelin2020/ERA-5/NC_FILES/{}'.format(year))
    mcs_dir = Path('/neelin2020/RGMA_feature_mask/data_product/{}/MERGED_FP'.format(year))    
    tb_dir = Path('/neelin2020/RGMA_feature_mask/data_product/{}/MERGE-IR'.format(year))

    for month in np.arange(1,13):

        print('month: {}'.format(str(month).zfill(2)))

        # MCS mask
        data_mcs = xr.open_dataset(mcs_dir / 'GPM_feature_merged_{}_vfinal.nc'.format(str(month).zfill(2)))
        data_mcs = data_mcs.rename({'latitude':'lat','longitude':'lon'}).sel(lat=slice(-30,30))

        # Tb, need to adjust the time stamps with extra floats 
        data_tb = xr.open_dataset(tb_dir / 'Tb_MERGE_IR_{}_{}_hrly.compress_new.nc'.format(year,str(month).zfill(2))).sel(lat=slice(-30,30))
        date_list = []
        for tt in data_tb.time.values:
            timestamp = str(tt)
            year = int(timestamp[:4])
            month = int(timestamp[5:7])
            day = int(timestamp[8:10])
            hour = int(timestamp[11:13])
            date_list.append(datetime(year,month,day,hour))
        data_tb['time'] = date_list

        # find common timestamps between MCS and Tb 
        common_dates = list(set(data_mcs.time.values) & set(data_tb.time.values))
        mcsmask = data_mcs.mcs_tag.sel(time=common_dates)
        tb = data_tb.tb.sel(time=common_dates)
        prec_gpm = data_mcs.precipitationCal.sel(time=common_dates)

        # 1. 0.25-deg, 3-hourly buoyancy measure: BL = BL,cape - BL,subsat
        buoy_files = list(buoy_dir.glob('era5_2layers_thetae_{}_{}_*.nc'.format(year,str(month).zfill(2))))
        data_buoy = xr.open_mfdataset(buoy_files).sel(lat=slice(-30,30))
        data_buoy = data_buoy.interp(lon=lon_re, lat=lat_re)
        data_buoy = data_buoy.sel(time=mcsmask.time)

        data_sp = xr.open_dataset(era5_dir / 'era-5.sp.{}.{}.nc'.format(year,str(month).zfill(2))).sel(latitude=slice(30
        ,-30)).SP/100 # hPa
        data_sp = data_sp.reindex(latitude=sorted(list(data_sp.latitude))) # fliping latitude order: -30 to 30
        data_sp = data_sp.rename({'latitude': 'lat', 'longitude': 'lon'})
        sp = data_sp.sel(time=mcsmask.time)

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

        # get nonMCS-assocaited grids only
        prec_gpm_nonmcs = prec_gpm.where((mcsmask == 0) & (tb <= 241))
        Buoy_TOT_nonmcs = Buoy_TOT.where((mcsmask == 0) & (tb <= 241))

        # 3. sampling by BL bins over regions
        for n, geo_info in enumerate([WPC, IND, EPC, ATL, WAF, MC, AMZ]):
            (b_samples, gpm_sum) = write_histogram_regions(Buoy_TOT_nonmcs
                                                            , prec_gpm_nonmcs, bins_bl, geo_info)

            buoy_samples[n,:] += b_samples
            prec_gpm_sum[n,:] += gpm_sum

    # writeout data
    ds = xr.Dataset(data_vars = dict(samples = (['region','BL_bins'], buoy_samples),
                                     prec_gpm_sum = (['region','BL_bins'], prec_gpm_sum)),
                    coords = dict(BL_bins = (['BL_bins'], bins_bl[:-1]),
                                  region = (['region'], ['WPC','IND','EPC','ATL','WAF','MC','AMZ'])),
                    attrs = dict(description = 'buoy-precipitation relationship in tropical regions. NonMCS=Tb < 241 K',
                                 bins_unit = 'degree Kelvin')
                   )  

    out_dir = Path('/scratch/wmtsai/temp_mcs/output_stats/buoy_precip_1D_multiregions/')
    os.system('mkdir -p {}'.format(out_dir))
    ds.to_netcdf(out_dir / 'buoy_precipitation_multiregions.nonMCS.Tb241.{}.nc'.format(year))
    print('buoy_precipitation_multiregions.nonMCS.Tb241.{}.nc ...completed'.format(year))

