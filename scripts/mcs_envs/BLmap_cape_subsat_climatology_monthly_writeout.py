import os
import sys
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

if __name__ == "__main__":

    year = sys.argv[1]
    print('processing year: {}'.format(year))
    ds_month_list = []
    for month in np.arange(1,13):
        print('month: ', month)
        buoy_dir = Path('/neelin2020/ERA-5_buoy/layer_thetae/')              
        # 1. 0.25-deg, 3-hourly buoyancy measure: BL = BL,cape - BL,subsat
        buoy_files = list(buoy_dir.glob('era5_2layers_thetae_{}_{}_*.nc'.format(year,str(month).zfill(2))))
        data_buoy = xr.open_mfdataset(buoy_files).sel(lat=slice(-30,30))

        era5_dir = Path('/neelin2020/ERA-5/NC_FILES/{}'.format(year))
        data_sp = xr.open_dataset(era5_dir / 'era-5.sp.{}.{}.nc'.format(year,str(month).zfill(2))).sel(latitude=slice(30
        ,-30)).SP/100 # hPa
        data_sp = data_sp.reindex(latitude=sorted(list(data_sp.latitude))) # fliping latitude order: -30 to 30
        data_sp = data_sp.interp(latitude=data_buoy.lat.values, longitude=data_buoy.lon.values)
        data_sp = data_sp.rename({'latitude': 'lat', 'longitude': 'lon'})
        sp = data_sp.sel(time=data_buoy.time)

        thetae_bl = data_buoy.thetae_bl
        thetae_sat_lt = data_buoy.thetae_sat_lt
        thetae_lt = data_buoy.thetae_lt

        # replace 0 somehow occurring with nan
        thetae_bl = thetae_bl.where(thetae_bl > 0)
        thetae_sat_lt = thetae_sat_lt.where(thetae_sat_lt > 0)
        thetae_lt = thetae_lt.where(thetae_lt > 0)

        delta_pl=sp-100-500 # top at 500hPa
        delta_pb=100
        wb=(delta_pb/delta_pl)*np.log((delta_pl+delta_pb)/delta_pb)
        wl=1-wb

        Buoy_CAPE = (9.81/(340*3)) * wb * ((thetae_bl-thetae_sat_lt)/thetae_sat_lt) * 340
        Buoy_SUBSAT = (9.81/(340*3)) * wl * ((thetae_sat_lt-thetae_lt)/thetae_sat_lt) * 340
        Buoy_TOT = Buoy_CAPE - Buoy_SUBSAT # degree Kelvin (K)
        
        # get monthly mean
        Buoy_CAPE_mean = Buoy_CAPE.mean('time').rename('Buoy_CAPE_mavg')
        Buoy_SUBSAT_mean = Buoy_SUBSAT.mean('time').rename('Buoy_SUBSAT_mavg')
        Buoy_TOT_mean = Buoy_TOT.mean('time').rename('Buoy_TOT_mavg')      

        ds_merged = xr.merge([Buoy_CAPE_mean, Buoy_SUBSAT_mean, Buoy_TOT_mean])
        ds_month_list.append(ds_merged)
        
    # writeout
    ds_month = xr.concat(ds_month_list, pd.Index(np.arange(1,13), name='month'))
    out_dir = Path('/scratch/wmtsai/temp_mcs/output_stats/')
    ds_month.to_netcdf(out_dir / 'BLmap_clim.test.nc')

