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
import warnings
warnings.filterwarnings('ignore')

def process_buoyancy(track):

    meanlon = data_stats.sel(tracks=track).isel(times=0).meanlon
    meanlat = data_stats.sel(tracks=track).isel(times=0).meanlat
    timestamp = str(data_stats.isel(tracks=track,times=0).base_time.values)
    month = timestamp[5:7]
    day = timestamp[8:10]
    hour = timestamp[11:13]
    
    tmp = lsmask.sel(longitude=meanlon, latitude=meanlat, method='nearest') # landsea info
    if tmp == 100: # ocean
        ls_flag = 0
    else:
        ls_flag = 1
        
    if ((hour == '00') or (hour == '03') or (hour == '06') or (hour == '09') or 
       (hour == '12') or (hour == '15') or (hour == '18') or (hour == '21') and (abs(meanlat) <= 30)):

        #print('track: {} date: {}-{}-{}'.format(track,month,day,hour))
        try:
            # get buoyancy data
            data_buoy = xr.open_dataset(buoy_dir / 'era5_2layers_thetae_{}_{}_{}.nc'.format(year,month,day))
            data_buoy = data_buoy.sel(time=timestamp, method='nearest') # files contain random seconds ...
            # change longitude from 0-360 to -180-180
            data_buoy.coords['lon'] = (data_buoy.coords['lon'] + 180) % 360 - 180
            data_buoy = data_buoy.sortby(data_buoy.lon)
            thetae_bl = data_buoy.thetae_bl
            thetae_sat_lt = data_buoy.thetae_sat_lt
            thetae_lt = data_buoy.thetae_lt
            sp = data_buoy.SP/100 # hPa


            delta_pl=sp-100-400
            delta_pb=100
            wb=(delta_pb/delta_pl)*np.log((delta_pl+delta_pb)/delta_pb)
            wl=1-wb

            # calculate buoyancy estimate (K)
            Buoy_CAPE = wb*((thetae_bl-thetae_sat_lt)/thetae_sat_lt) * 340 
            Buoy_SUBSAT = wl*((thetae_sat_lt-thetae_lt)/thetae_sat_lt) * 340

            Buoy_CAPE_regrid = Buoy_CAPE.interp(lat=lat_regrid, lon=lon_regrid)
            Buoy_SUBSAT_regrid = Buoy_SUBSAT.interp(lat=lat_regrid, lon=lon_regrid)

            # get the buoy component around nonMCS center
            BL_CAPE_mean = Buoy_CAPE_regrid.sel(lat=meanlat,
                                                lon=meanlon, method='nearest').values
            BL_SUBSAT_mean = Buoy_SUBSAT_regrid.sel(lat=meanlat,
                                                lon=meanlon, method='nearest').values
           
            return (ls_flag, BL_CAPE_mean, BL_SUBSAT_mean)

        except: # several error files without complete thetae vars.
            
            return (-999, -999, -999)

    return (-999, -999, -999)

##########################################
if __name__ == '__main__':

    from multiprocessing import Pool

    data_dir = Path('/neelin2020/mcs_flextrkr')
    buoy_dir = Path('/neelin2020/ERA-5_buoy/layer_thetae/')

    # bins for BL_CAPE and BL_SUBSAT
    bins_cape = np.arange(-15,10,0.2)
    bins_subsat = np.arange(-5,25,0.2)
    bins_samples = np.zeros((2,len(bins_cape)-1, len(bins_subsat)-1)) # (cape, subsat) for nonMCS, so no phase

    year = int(sys.argv[1]) # selected year
    print('processing year: {}'.format(year))

    # get nonMCS info from previous tracking
    data_stats = xr.open_dataset('/scratch/wmtsai/temp_mcs/mcs_stats/trackstats_{}0101.0000_{}0101.0000.nc'.format(year,year+1))
    # load landsea mask
    tmp = xr.open_dataset('/neelin2020/RGMA_feature_mask/ERA5_LandSeaMask_regrid.nc4').sel(latitude=slice(-40,40))
    lsmask = tmp.landseamask
    lsmask.coords['longitude'] = (lsmask.coords['longitude'] + 180) % 360 - 180
    lsmask = lsmask.sortby(lsmask.longitude)

    lon_regrid = np.arange(-180,180,0.25)
    lat_regrid = np.arange(-30,30+0.25,0.25)

    # assign tracks
    track_list =  data_stats.tracks.values[::10]
    print('total tracks processed: {}'.format(len(track_list)))
    
    # start multiprocessing
    pool = Pool(processes=10) # cpu numbers
    results = pool.map(process_buoyancy, track_list)
    pool.close()
    pool.join()
    print('length of results: {}'.format(len(results)))

    # convert result into numpy array
    landsea_1d = np.zeros(len(results))
    BL_CAPE_1d = np.copy(landsea_1d)
    BL_SUBSAT_1d = np.copy(landsea_1d)
    for i,result in enumerate(results):
        landsea_1d[i] = result[0]
        BL_CAPE_1d[i] = result[1]
        BL_SUBSAT_1d[i] = result[2]
    
    # 2-D joint histogram: BL_CAPE-BL_SUBSAT
    idx_land = np.where(landsea_1d == 1)[0]
    idx_ocean = np.where(landsea_1d == 0)[0]
    
    BL_CAPE_oce = BL_CAPE_1d[idx_ocean]
    BL_SUBSAT_oce = BL_SUBSAT_1d[idx_ocean]
    BL_CAPE_lnd = BL_CAPE_1d[idx_land]
    BL_SUBSAT_lnd = BL_SUBSAT_1d[idx_land]
    # filling bins for ocean samples
    for i in range(len(bins_cape)-1):
        idx = np.where(np.logical_and(BL_CAPE_oce >= bins_cape[i], BL_CAPE_oce < bins_cape[i+1]))[0]
        for j in range(len(bins_subsat)-1):
            idy = np.where(np.logical_and(BL_SUBSAT_oce >= bins_subsat[j], BL_SUBSAT_oce < bins_subsat[j+1]))[0]
            idx_com = np.intersect1d(idx,idy)
            bins_samples[0,i,j] = len(idx_com)
    # filling bins for land samples
    for i in range(len(bins_cape)-1):
        idx = np.where(np.logical_and(BL_CAPE_lnd >= bins_cape[i], BL_CAPE_lnd < bins_cape[i+1]))[0]
        for j in range(len(bins_subsat)-1):
            idy = np.where(np.logical_and(BL_SUBSAT_lnd >= bins_subsat[j], BL_SUBSAT_lnd < bins_subsat[j+1]))[0]
            idx_com = np.intersect1d(idx,idy)
            bins_samples[1,i,j] = len(idx_com)

    ds = xr.Dataset(data_vars=dict(samples=(['surface_type','bins_cape','bins_subsat'], bins_samples)),
                coords=dict(surface_type=(['surface_type'], ['ocean','land']),
                            bins_cape=(['bins_cape'], bins_cape[:-1]+0.25),
                            bins_subsat=(['bins_subsat'], bins_subsat[:-1]+0.25)))

    # writeout 
    out_dir = Path('/scratch/wmtsai/temp_mcs/output_stats/cape_subsat_hist/')
    ds.to_netcdf(out_dir / 'hist2d_cape_subsat_nonMCS.{}.alltracks.parallel.nc'.format(year))
    print('hist2d file saved..')

