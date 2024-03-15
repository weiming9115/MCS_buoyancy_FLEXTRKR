import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from scipy.stats import linregress

# for converting UTC to local time
from timezonefinder import TimezoneFinder 
import pytz  # 3rd party: $ pip install pytz

import warnings
warnings.filterwarnings('ignore')

def add_localtime_var(data_xr, longitude, latitude):
    """
    add local time as a new coordinate for a 1-D timeseries
    """
    obj = TimezoneFinder() 
    timezone_str = obj.timezone_at(lng=longitude, lat=latitude)

    # converting UTC to local time
    time_local_list = []
    for t in range(len(data_xr.time)):
        time_sel = data_xr.isel(time=t).time.values # a given datetime
        timestamp = ((time_sel - np.datetime64('1970-01-01T00:00:00'))
                    / np.timedelta64(1, 's'))
        time_utc = datetime.utcfromtimestamp(timestamp) # converting datetime64 to datetime
        time_utc = time_utc.replace(tzinfo=pytz.utc) # NOTE: it works only with a fixed utc offset   
        
        # converting datetime to datetime64: np.datetime64('2019-08-26T00:00:00.000000000')
        time_local = time_utc.astimezone(pytz.timezone(timezone_str))
        time_dt64 = np.datetime64('{}-{}-{}T{}:00:00.000000000'.format(time_local.year, str(time_local.month).zfill(2),
                                                            str(time_local.day).zfill(2), str(time_local.hour).zfill(2)))
        time_local_list.append(time_dt64)
    
    data_xr = data_xr.to_dataset()
    data_xr.coords['time_local'] = ('time', time_local_list)
    data_xr = data_xr.swap_dims({'time': 'time_local'})
        
    return data_xr

def estimate_peak_time(pcp_diurnal):

    time_shift = np.linspace(0,23,24)
    tt = np.arange(0,24,1)
    rval_list = []
    for dt in time_shift:
    
        pcp_amp = pcp_diurnal.mean().values
        pcp_range = np.ptp(pcp_diurnal.values)
        p_harmonic = pcp_amp + pcp_range*np.sin((tt-dt)/24*(2*np.pi))
       
        try:
            stats = linregress(p_harmonic, pcp_diurnal.values)
            r_val = stats[2] # correlation coefficient
            rval_list.append(r_val)
        except: 
            rval_list.append(0)

    idx_peak = np.argmax(np.asarray(rval_list))

    hour_peak = 6 + time_shift[idx_peak]
    if hour_peak == 24:
        hour_peak = 0

    return hour_peak

def process_diurnal_peak(i):

    # get lat,lon info from the merged lat-lon dimension
    lon_sel = float(data_gpm_reshape.geo_loc[i].longitude.values)
    lat_sel = float(data_gpm_reshape.geo_loc[i].latitude.values)

    data_gpt = data_gpm_regrid['precipitationCal'].sel(longitude=lon_sel, latitude=lat_sel)
    data_gpt_update = add_localtime_var(data_gpt, longitude=360-data_gpt.longitude, latitude=data_gpt.latitude)
    # calculating the composite diurnal cycle
    pcp_diurnal = data_gpt_update.precipitationCal.groupby('time.hour').mean() 
    hour_peak = estimate_peak_time(pcp_diurnal) # ranges from 0 to 23

    return hour_peak, pcp_diurnal

if __name__ == '__main__':

    start_time = datetime.now()
    # load gpm precip
    year_list = np.arange(2014,2015)
    print('analyzed years: {}'.format(year_list))

    files = []
    for year in year_list:
        gpm_dir = Path('/neelin2020/mcs_flextrkr/gpm_AMAZON/{}/merged'.format(year))
        files += sorted(list(gpm_dir.glob('*.nc')))
    data_gpm = xr.open_mfdataset(files)

    mcs_stats_dir = Path('/neelin2020/RGMA_feature_mask/data_product/2014/MCS_orig')
    files = sorted(list(mcs_stats_dir.glob('*.nc')))
    # AMAZON: (85W-30W, 22S-8N)
    data_era5 = xr.open_dataset(files[0]).sel(lat=slice(-22,8), lon=slice(-85+360, -30+360))
    # regridding into 0.25 deg
    data_gpm_regrid = data_gpm.interp(longitude=data_era5.lon.values
                                    , latitude=data_era5.lat.values).compute()

    # stack lat/lon for parallel
    data_gpm_reshape = data_gpm_regrid.stack(geo_loc=('latitude','longitude'))

    ##### initialize parallel computing #####
    from multiprocessing import Pool

    pool = Pool(processes=12)
    results = pool.map(process_diurnal_peak, range(len(data_gpm_reshape.geo_loc)))
    
    pool.close()
    pool.join()
    
    #### end parallel computing ####

    hr_peak_map = np.zeros((len(data_gpm_regrid.latitude)*len(data_gpm_regrid.longitude)))
    pcp_diurnal_map = np.zeros((24,len(data_gpm_regrid.latitude)*len(data_gpm_regrid.longitude)))    
    for i in range(len(results)):
        hr_peak_map[i] = results[i][0]
        pcp_diurnal_map[:,i] = results[i][1]
    hr_peak_map = hr_peak_map.reshape(len(data_gpm_regrid.latitude), len(data_gpm_regrid.longitude))
    pcp_diurnal_map = pcp_diurnal_map.reshape(24, len(data_gpm_regrid.latitude), len(data_gpm_regrid.longitude))

    # writeout netcdf file
    diurnal_stats_xr = xr.Dataset(
             data_vars=dict(hr_peak=(['latitude','longitude'], hr_peak_map),
                            precipitationCal=(['time_local','latitude','longitude'], pcp_diurnal_map)),

             coords=dict(time_local=(['time_local'], np.arange(0,24,1)),
                         longitude=(['longitude'], data_gpm_regrid.longitude.values),
                         latitude=(['latitude'], data_gpm_regrid.latitude.values)),

             attrs=dict(description='Diurnal cycle of precipitation from GPM-IMERG V6 gridded observations.',
                        years=str(year_list),
                        resolution='Hourly. Regridded into the 0.25-degree ERA-5 coordinates.'
                        ))

    out_dir = Path('/neelin2020/mcs_flextrkr/diurnal_analysis')
    diurnal_stats_xr.to_netcdf(out_dir / 'diurnal_precip_stats.nc')

    end_time = datetime.now()
    print('Data processing completed')
    print('Execution time spent: {}'.format(end_time - start_time))
