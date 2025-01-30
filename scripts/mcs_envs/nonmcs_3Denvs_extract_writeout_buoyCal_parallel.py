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

import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.util import add_cyclic_point
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# calculations for thermodynamics
from metpy.calc import thermo
from metpy.units import units

# importing theta_calc module
sys.path.append('/neelin2020/mcs_flextrkr/scripts/modules') 
from theta_e_calc_mod import *

import warnings
warnings.filterwarnings('ignore')

def get_mcs_mask(track_number, phase_list):
   
    mask_sub_phase = []
    
    for idt_phase in phase_list:

        timestamp_phase = data_non2mcs_complete.base_time.sel(tracks=track_number, times=int(idt_phase))
        meanlon =  data_non2mcs_complete.meanlon.sel(tracks=track_number, times=int(idt_phase))
        meanlat =  data_non2mcs_complete.meanlat.sel(tracks=track_number, times=int(idt_phase))
        
        timestamp_str = str(timestamp_phase.values)
        year = timestamp_str[:4]
        month = timestamp_str[5:7]
        day = timestamp_str[8:10]
        hour = timestamp_str[11:13]

        mask_data = xr.open_dataset('/neelin2020/mcs_flextrkr/{}0101.0000_{}0101.0000/mcstrack_{}{}{}_{}30.nc'.format(year
                                                                                ,int(year)+1,year,month,day,hour))
        mcsnumber = data_non2mcs_complete.sel(tracks=track_number).tracks.values
        mask_sub = mask_data.cloudtracknumber_nomergesplit.isel(time=0)
        mask_sub = mask_sub.where(mask_sub == mcsnumber + 1, 0)
        mask_sub = mask_sub.where(mask_sub == 0, 1) # return 0, 1 binary mask
        mask_sub = mask_sub.sel(lon=slice(meanlon-5,meanlon+5), lat=slice(meanlat-5,meanlat+5)) # 10x10 domain
        
        # interpolate into 25 x 25 grids (equivalent to ~ 625 km)
        mask_sub_xy = mask_sub.interp(lon=np.linspace(mask_sub.lon.min(),mask_sub.lon.max(),40),
                                        lat=np.linspace(mask_sub.lat.min(),mask_sub.lat.max(),40))
        mask_sub_xy = mask_sub_xy.where(mask_sub_xy > 0, 0)
        mask_sub_xy = mask_sub_xy.where(mask_sub_xy == 0, 1)
        # converting lat-lon into x-y coordinates
        mask_sub_xy = mask_sub_xy.assign_coords(x=("lon", np.arange(0,40,1)), y=("lat", np.arange(0,40,1)))
        mask_sub_xy = mask_sub_xy.swap_dims({'lon':'x', 'lat': 'y'}).drop(['time','lat','lon'])
        
        mask_sub_phase.append(mask_sub_xy)
        
    mask_sub_phase_xr = xr.concat(mask_sub_phase, dim=pd.Index(['Init','Grow','Mature','Decay','End'], name='mcs_phase'))
    
    return mask_sub_phase_xr

def get_pr_estimates(track_number, phase_list):
    
    pr_sub_merge_phase = []
    
    for idt_phase in phase_list:

        timestamp_phase = data_non2mcs_complete.base_time.sel(tracks=track_number, times=int(idt_phase))
        meanlon = data_non2mcs_complete.meanlon.sel(tracks=track_number, times=int(idt_phase))
        meanlat = data_non2mcs_complete.meanlat.sel(tracks=track_number, times=int(idt_phase))
        
        # convert to era5 lon-lat
        if meanlon < 0:
            meanlon_era5 = meanlon + 360
        else:
            meanlon_era5 = meanlon
            
        timestamp_str = str(timestamp_phase.values)
        year = timestamp_str[:4]
        month = timestamp_str[5:7]
        day = timestamp_str[8:10]
        hour = timestamp_str[11:13]
        
        # 1. get ERA-5 precip. mtpr
        pr_data = xr.open_dataset(dir_era5 / '{}/era-5.mpr.{}.{}.nc'.format(year,year,month))
        pr_data = pr_data.reindex(latitude=list(reversed(pr_data.latitude))) # reverse latitude order
        pr_data = pr_data.sel(time=timestamp_phase, method='nearest')
        pr_sub = 3600*pr_data.mtpr.sel(longitude=slice(meanlon_era5-5,meanlon_era5+5), latitude=slice(meanlat-5,meanlat+5)) # [mm/hr]
        # interpolate into 25 x 25 grids (equivalent to ~ 625 km)
        pr_sub_xy = pr_sub.interp(longitude=np.linspace(pr_sub.longitude.min(), pr_sub.longitude.max(),40),
                                  latitude=np.linspace(pr_sub.latitude.min(), pr_sub.latitude.max(),40))
        # converting lat-lon into x-y coordinates
        pr_sub_xy = pr_sub_xy.assign_coords(x=("longitude", np.arange(0,40,1)), y=("latitude", np.arange(0,40,1)))
        pr_sub_xy = pr_sub_xy.swap_dims({'longitude':'x', 'latitude': 'y'}).drop(['time','longitude','latitude'])
        
        # 2. get GPM-IMERG 
        gpm_data = xr.open_dataset('/neelin2020/RGMA_feature_mask/GPM_ncfiles_{}/GPM_IMERGE_V06_{}{}{}_{}00.nc'.format(
                                    year, year, month, day, hour))
        gpm_data = gpm_data.sel(time=timestamp_phase, method='nearest')
        gpm_sub = gpm_data.precipitationCal.sel(lon=slice(meanlon-5,meanlon+5), lat=slice(meanlat-5,meanlat+5))
        # swap coordinate from (lon, lat) to (lat, lon) for consistency
        gpm_sub = gpm_sub.transpose("lat", "lon")
        
        # interpolate into 25 x 25 grids (equivalent to ~ 625 km)
        gpm_sub_xy = gpm_sub.interp(lon=np.linspace(gpm_sub.lon.min(), gpm_sub.lon.max(),40),
                                    lat=np.linspace(gpm_sub.lat.min(), gpm_sub.lat.max(),40))
        # converting lat-lon into x-y coordinates
        gpm_sub_xy = gpm_sub_xy.assign_coords(x=("lon", np.arange(0,40,1)), y=("lat", np.arange(0,40,1)))
        gpm_sub_xy = gpm_sub_xy.swap_dims({'lon':'x', 'lat': 'y'}).drop(['time','lon','lat'])
        
        # merge two precip data
        pr_sub_merge_xy = xr.merge([pr_sub_xy, gpm_sub_xy])
        
        pr_sub_merge_phase.append(pr_sub_merge_xy)
        
    pr_sub_merge_phase_xr = xr.concat(pr_sub_merge_phase, dim=pd.Index(['Init','Grow','Mature','Decay','End']
                                                                   , name='mcs_phase')).drop_vars(['tracks','times'])
    
    return pr_sub_merge_phase_xr

def get_2dera5_estimates(track_number, name, var_name, phase_list):
    
    var2d_sub_phase = []
    
    for idt_phase in phase_list:

        timestamp_phase = data_non2mcs_complete.base_time.sel(tracks=track_number, times=int(idt_phase))
        meanlon = data_non2mcs_complete.meanlon.sel(tracks=track_number, times=int(idt_phase))
        meanlat = data_non2mcs_complete.meanlat.sel(tracks=track_number, times=int(idt_phase))

        # convert track geolocation to era5 lon-lat
        if meanlon < 0:
            meanlon_era5 = meanlon + 360
        else:
            meanlon_era5 = meanlon
            
        timestamp_str = str(timestamp_phase.values)
        year = timestamp_str[:4]
        month = timestamp_str[5:7]
        day = timestamp_str[8:10]
        hour = timestamp_str[11:13]
        
        data = xr.open_dataset(dir_era5 / '{}/era-5.{}.{}.{}.nc'.format(year,name,year,month))
        
        data = data.reindex(latitude=list(reversed(data.latitude))) # reverse latitude order
        data = data.sel(time=timestamp_phase, method='nearest')
        data_sub = data[var_name].sel(longitude=slice(meanlon_era5-5,meanlon_era5+5), latitude=slice(meanlat-5,meanlat+5)) 
        # interpolate into 25 x 25 grids (equivalent to ~ 625 km)
        data_sub_xy = data_sub.interp(longitude=np.linspace(data_sub.longitude.min(), data_sub.longitude.max(),40),
                                  latitude=np.linspace(data_sub.latitude.min(), data_sub.latitude.max(),40))
        # converting lat-lon into x-y coordinates
        data_sub_xy = data_sub_xy.assign_coords(x=("longitude", np.arange(0,40,1)), y=("latitude", np.arange(0,40,1)))
        data_sub_xy = data_sub_xy.swap_dims({'longitude':'x', 'latitude': 'y'}).drop(['time','longitude','latitude'])
        
        var2d_sub_phase.append(data_sub_xy)
        
    var2d_sub_phase_xr = xr.concat(var2d_sub_phase, dim=pd.Index(['Init','Grow','Mature','Decay','End']
                                                                   , name='mcs_phase')).drop_vars(['tracks','times'])
    
    return var2d_sub_phase_xr

def get_3dera5_estimates(track_number, name, var_name, phase_list):
    
    var3d_sub_phase = []
    
    for idt_phase in phase_list:

        timestamp_phase = data_non2mcs_complete.base_time.sel(tracks=track_number, times=int(idt_phase))
        meanlon = data_non2mcs_complete.meanlon.sel(tracks=track_number, times=int(idt_phase))
        meanlat = data_non2mcs_complete.meanlat.sel(tracks=track_number, times=int(idt_phase))

        # convert to era5 lon-lat
        if meanlon < 0:
            meanlon_era5 = meanlon + 360
        else:
            meanlon_era5 = meanlon
            
        timestamp_str = str(timestamp_phase.values)
        year = timestamp_str[:4]
        month = timestamp_str[5:7]
        day = timestamp_str[8:10]
        hour = timestamp_str[11:13]
     
        # 1. get ERA-5 3d vars
        data = xr.open_dataset(dir_era5 / '{}/era-5.{}.{}.{}.nc'.format(year,name,year,month))
       
        data = data.reindex(latitude=list(reversed(data.latitude))) # reverse latitude order
        data = data.sel(time=timestamp_phase, method='nearest')
        data_sub = data[var_name].sel(longitude=slice(meanlon_era5-5,meanlon_era5+5), latitude=slice(meanlat-5,meanlat+5)).sel(level=slice(100,1000)) 
        # interpolate into 25 x 25 grids (equivalent to ~ 625 km)
        data_sub_xy = data_sub.interp(longitude=np.linspace(data_sub.longitude.min(), data_sub.longitude.max(),40),
                                  latitude=np.linspace(data_sub.latitude.min(), data_sub.latitude.max(),40))
        # converting lat-lon into x-y coordinates
        data_sub_xy = data_sub_xy.assign_coords(x=("longitude", np.arange(0,40,1)), y=("latitude", np.arange(0,40,1)))
        data_sub_xy = data_sub_xy.swap_dims({'longitude':'x', 'latitude': 'y'}).drop(['time','longitude','latitude'])
        
        var3d_sub_phase.append(data_sub_xy)
        
    var3d_sub_phase_xr = xr.concat(var3d_sub_phase, dim=pd.Index(['Init','Grow','Mature','Decay','End']
                                                                   , name='mcs_phase')).drop_vars(['tracks','times'])
    
    return var3d_sub_phase_xr

def get_tb_estimates(track_number, phase_list):
    
    tb_sub_phase = []
    
    for idt_phase in phase_list:

        timestamp_phase = data_non2mcs_complete.base_time.sel(tracks=track_number, times=int(idt_phase))
        meanlon = data_non2mcs_complete.meanlon.sel(tracks=track_number, times=int(idt_phase))
        meanlat = data_non2mcs_complete.meanlat.sel(tracks=track_number, times=int(idt_phase))
        
        # convert to era5 lon-lat
        if meanlon < 0:
            meanlon_era5 = meanlon + 360
        else:
            meanlon_era5 = meanlon
            
        timestamp_str = str(timestamp_phase.values)
        year = timestamp_str[:4]
        month = timestamp_str[5:7]
        day = timestamp_str[8:10]
        hour = timestamp_str[11:13]
        
        # 1. get regridded MERGE-IR 0.25-deg
        tb_data = xr.open_dataset('/neelin2020/RGMA_feature_mask/data_product/' + 
                                  '{}/MERGE-IR/Tb_MERGE_IR_{}_{}_hrly.compress.nc'.format(year,year,month))
        tb_data = tb_data.sel(time=timestamp_phase, method='nearest')
        tb_sub = tb_data.tb.sel(lon=slice(meanlon_era5-5,meanlon_era5+5), lat=slice(meanlat-5,meanlat+5)) 
        # interpolate into 25 x 25 grids (equivalent to ~ 625 km)
        tb_sub_xy = tb_sub.interp(lon=np.linspace(tb_sub.lon.min(), tb_sub.lon.max(),40),
                                  lat=np.linspace(tb_sub.lat.min(), tb_sub.lat.max(),40) )
        # converting lat-lon into x-y coordinates
        tb_sub_xy = tb_sub_xy.assign_coords(x=("lon", np.arange(0,40,1)), y=("lat", np.arange(0,40,1)))
        tb_sub_xy = tb_sub_xy.swap_dims({'lon':'x', 'lat': 'y'}).drop(['time','lon','lat'])
        
        tb_sub_phase.append(tb_sub_xy)
        
    tb_sub_phase_xr = xr.concat(tb_sub_phase, dim=pd.Index(['Init','Grow','Mature','Decay','End']
                                                                   , name='mcs_phase')).drop_vars(['tracks','times'])
    
    return tb_sub_phase_xr

def BL_estimates_cal_phase(T, q, sp, T2m, q2m):
    """
    function for calcultinig the low-trospospheric buoyancy estimates
    T, q : 3D xarray dataarray (level, lat, lon)
    sp: surface pressure (lat, lon), unit: hPa
    T2m, q2m: temperature, specific humidity at 2m (lat, lon) 
    """
    
    T = T.drop('mcs_phase')
    q = q.drop('mcs_phase')
    T2m = T2m.drop('mcs_phase')
    q2m = q2m.drop('mcs_phase')
    sp = sp.drop('mcs_phase')
    
    # constants
    Lv = 2.5e6 # (J/kg)
    g = 9.81 # (kg/m^2)
    cpd = 1004 # (J/kg/K)
    p0 = 1000  # (hPa)
    Rd = 287.15 # (J/kg)

    # find pbl top (100 hPa above the surface)
    pbl_top_level = find_pbl_top_level(sp, T, pbl_depth=100)
    
    thetae_bl_list = []
    thetae_lt_list = []
    thetae_sat_lt_list = []
    
    # loop for lat-lon grids
    for idx_lat in range(len(q2m.y)):
        for idx_lon in range(len(q2m.x)):
                        
            try:
                sf_p = sp.isel(y=idx_lat, x=idx_lon) # surface pressure 
                pbl_p = pbl_top_level.isel(y=idx_lat, x=idx_lon).pbl_top_level.values # pbl top pressure
                T_sf = T2m.isel(y=idx_lat, x=idx_lon).values
                q_sf = q2m.isel(y=idx_lat, x=idx_lon).values

                T_at_sf = T2m.isel(y=idx_lat, x=idx_lon).values
                q_at_sf = q2m.isel(y=idx_lat, x=idx_lon).values

                T_above_sf = T.isel(y=idx_lat, x=idx_lon).sel(level=slice(100,int(sf_p))).values
                q_above_sf = q.isel(y=idx_lat, x=idx_lon).sel(level=slice(100,int(sf_p))).values

                # reconstruct T, q profile by adding surface quantities
                T_1d = np.concatenate([np.array([T_at_sf]), np.flip(T_above_sf)])            
                q_1d = np.concatenate([np.array([q_at_sf]), np.flip(q_above_sf)])
                pressure_1d = np.concatenate([np.array([sf_p]), np.flip(T.sel(level=slice(100,int(sf_p))).level.values)])
                T_1d_xr = xr.DataArray(data=T_1d,
                                       dims=["level"],
                                       coords=dict(level=(["level"], pressure_1d)))
                q_1d_xr = xr.DataArray(data=q_1d,
                                       dims=["level"],
                                       coords=dict(level=(["level"], pressure_1d)))

                # splitting into boundary layer and lower free troposphere
                # 1. boundary layer, bl
                q_bl = q_1d_xr.sel(level=slice(int(sf_p), pbl_p))
                T_bl = T_1d_xr.sel(level=slice(int(sf_p), pbl_p))
                # 2. lower free troposphere, lt
                q_lt = q_1d_xr.sel(level=slice(pbl_p,500))
                T_lt = T_1d_xr.sel(level=slice(pbl_p,500))     

                # calculating layer-averaged thetae components
                thetae_bl = theta_e_calc(T_bl, q_bl).integrate('level')/(T_bl.level[0]-T_bl.level[-1]) # negative sign b.c. decreasing p
                thetae_lt = theta_e_calc(T_lt, q_lt).integrate('level')/(T_lt.level[0]-T_lt.level[-1])
                qsat_lt = qs_calc(T_lt)
                thetae_sat_lt = theta_e_calc(T_lt, qsat_lt).integrate('level')/(T_lt.level[0]-T_lt.level[-1]) 

                thetae_bl_list.append(thetae_bl.values)
                thetae_lt_list.append(thetae_lt.values)
                thetae_sat_lt_list.append(thetae_sat_lt.values)
               
            except:
                
                thetae_bl_list.append(np.nan)
                thetae_lt_list.append(np.nan)
                thetae_sat_lt_list.append(np.nan)
            
    # convert to geolocated xarray
    thetae_bl_xr = xr.DataArray(data = np.asarray(thetae_bl_list).reshape((len(T.y), len(T.x))),
                               dims = ["y",'x'],
                               coords = dict(y=(["y"], T.y.values),
                                           x=(["x"], T.x.values)))
    thetae_lt_xr = xr.DataArray(data = np.asarray(thetae_lt_list).reshape((len(T.y), len(T.x))),
                               dims = ["y",'x'],
                               coords = dict(y=(["y"], T.y.values),
                                           x=(["x"], T.x.values)))
    thetae_sat_lt_xr = xr.DataArray(data = np.asarray(thetae_sat_lt_list).reshape((len(T.y), len(T.x))),
                               dims = ["y",'x'],
                               coords = dict(y=(["y"], T.y.values),
                                           x=(["x"], T.x.values)))
    # calculate buoyancy estimates
    # 2-d weighting parameters for pbl and lt
    delta_pl=sp-100-500
    delta_pb=100
    wb=(delta_pb/delta_pl)*np.log((delta_pl+delta_pb)/delta_pb)
    wl=1-wb
    
    # calculate buoyancy estimate
    Buoy_CAPE = (9.81/(340*3)) * wb * ((thetae_bl_xr-thetae_sat_lt_xr)/thetae_sat_lt_xr) * 340
    Buoy_CAPE_xr = Buoy_CAPE.rename('Buoy_CAPE').to_dataset()
    Buoy_SUBSAT = (9.81/(340*3))* wl * ((thetae_sat_lt_xr-thetae_lt_xr)/thetae_sat_lt_xr) * 340
    Buoy_SUBSAT_xr = Buoy_SUBSAT.rename('Buoy_SUBSAT').to_dataset()
    Buoy_TOT = Buoy_CAPE - Buoy_SUBSAT
    Buoy_TOT_xr = Buoy_TOT.rename('Buoy_TOT').to_dataset()
    
    thetae_bl_xr  = thetae_bl_xr.rename('thetae_bl').to_dataset()
    thetae_lt_xr  = thetae_lt_xr.rename('thetae_lt').to_dataset()
    thetae_sat_lt_xr  = thetae_sat_lt_xr.rename('thetae_sat_lt').to_dataset()
    
    return xr.merge([Buoy_CAPE_xr, Buoy_SUBSAT_xr, Buoy_TOT_xr, thetae_bl_xr, thetae_lt_xr, thetae_sat_lt_xr])

def process_vars_env_writeout(i):

    track_number = i # cloudtracknumber in PyFLEXTRKR

    # MCS initial time centroid
    idt_init = data_non2mcs_complete.sel(tracks=track_number).idt_nonmcs_init.values
    meanlon = data_non2mcs_complete.sel(tracks=track_number, times=idt_init).meanlon
    meanlat = data_non2mcs_complete.sel(tracks=track_number, times=idt_init).meanlat

    if meanlon < 0: # if negative in longitude
        meanlon_era5 = meanlon + 360
    else:
        meanlon_era5 = meanlon

    # get phase_list
    phase_list = [
                  data_non2mcs_complete.sel(tracks=track_number).idt_nonmcs_init.values,
                  data_non2mcs_complete.sel(tracks=track_number).idt_nonmcs_grow.values,
                  data_non2mcs_complete.sel(tracks=track_number).idt_nonmcs_mature.values,
                  data_non2mcs_complete.sel(tracks=track_number).idt_nonmcs_decay.values,
                  data_non2mcs_complete.sel(tracks=track_number).idt_nonmcs_end.values]

    # setting var_name accords to the era-5 variable name
#    mcs_mask_phase_xr = get_mcs_mask(track_number, phase_list)
    prec_phase_xr = get_pr_estimates(track_number, phase_list)
    T3d_phase_xr = get_3dera5_estimates(track_number, name='T', var_name='t', phase_list=phase_list)
    q3d_phase_xr = get_3dera5_estimates(track_number, name='q', var_name='q', phase_list=phase_list)
    sp_phase_xr = get_2dera5_estimates(track_number, name='sp', var_name='SP', phase_list=phase_list)
    td2m_phase_xr = get_2dera5_estimates(track_number, name='2d', var_name='VAR_2D', phase_list=phase_list)
    t2m_phase_xr = get_2dera5_estimates(track_number, name='2t', var_name='VAR_2T', phase_list=phase_list)
    tb_phase_xr = get_tb_estimates(track_number, phase_list)

    # merge all variables into a single dataset
    vars_mcsenvs_xr = xr.merge([prec_phase_xr, sp_phase_xr, td2m_phase_xr,
                            t2m_phase_xr, tb_phase_xr,
                            T3d_phase_xr, q3d_phase_xr])

    # calculate buoyancy based on the variables contained
    data_T = vars_mcsenvs_xr.t
    data_q = vars_mcsenvs_xr.q
    data_sp = vars_mcsenvs_xr.SP/100
    data_t2m = vars_mcsenvs_xr.VAR_2T
    data_d2m = vars_mcsenvs_xr.VAR_2D
    data_q2m = thermo.specific_humidity_from_dewpoint(data_sp*100 * units.pascal, data_d2m * units.kelvin)

    BL_merged_list = []
    for phase in vars_mcsenvs_xr.mcs_phase:
        # # derive low-tropospheric buoyancy estimates
        BL_merged_sub = BL_estimates_cal_phase(data_T.sel(mcs_phase=phase), data_q.sel(mcs_phase=phase), data_sp.sel(mcs_phase=phase)   
                                     , data_t2m.sel(mcs_phase=phase), data_q2m.sel(mcs_phase=phase))
        BL_merged_list.append(BL_merged_sub)

    BL_merged_all = xr.concat(BL_merged_list, pd.Index(vars_mcsenvs_xr.mcs_phase.values,
                                                  name='mcs_phase'))
    vars_mcsenvs_all_xr = xr.merge([vars_mcsenvs_xr, BL_merged_all])

    # write out as netcdf file
    vars_mcsenvs_all_xr.to_netcdf(out_dir / 'nonmcs_era5_3D_envs_{}.{}.LD.nc'.format(year, str(track_number).zfill(7)))
    print('nonmcs_era5_3D_envs_{}.{}.LD.nc ....saved'.format(year, str(track_number).zfill(7)))

################# load processed non2mcs tracks ##################

if __name__ == '__main__':

    from multiprocessing import Pool

    year = sys.argv[1]
    opt_continue = str(sys.argv[2])

    # data directoies
    dir_mcs_track = Path('/scratch/wmtsai/temp_mcs/mcs_stats/nonmcs_tracks_samples')
    dir_era5 = Path('/neelin2020/ERA-5/NC_FILES/')
    data_non2mcs_complete = xr.open_dataset(dir_mcs_track / 'nonmcs_tracks_{}.tropics30NS.extend.nc'.format(year))

#    out_dir = Path('/neelin2020/mcs_flextrkr/mcs_stats/envs_track/{}/tropics'.format(year))
    out_dir = Path('/scratch/wmtsai/temp_mcs/mcs_stats/envs_track/{}/nonmcs_extend/'.format(year))
    if out_dir.exists() == False:
        os.system('mkdir -p {}'.format(out_dir))

    if opt_continue == 'continue':

        tracks_all = data_non2mcs_complete.tracks.values
       
        files = sorted(list(out_dir.glob('*.nc')))
        track_exist = [] # available mcs_envs
        for file in files:
            track_exist.append(int(file.name[-11:-6]))
        tracks_input = np.setdiff1d(tracks_all, np.asarray(track_exist))
        print('Tracks processed: {}/{}'.format(len(track_exist), len(tracks_all)))
        print('Tracks remained: {}'.format(len(tracks_input)))

        pool = Pool(processes=15) # cpu numbers
        pool.map(process_vars_env_writeout, tracks_input)

        pool.close()
        pool.join()

    elif opt_continue == 'new':
        
        tracks_all = data_non2mcs_complete.tracks.values

        # start multiprocessing
        pool = Pool(processes=15) # cpu numbers
        pool.map(process_vars_env_writeout, data_non2mcs_complete.tracks.values)

        pool.close()
        pool.join()

    elif opt_continue == 'single':

        track_number = 84        
        print("prcessing single track: {}".format(track_number))

        process_vars_env_writeout(track_number)

