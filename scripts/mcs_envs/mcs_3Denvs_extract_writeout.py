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
#from metpy.calc import thermo
#from metpy.units import units

# importing theta_calc module
sys.path.append('/neelin2020/mcs_flextrkr/scripts/modules') 
#from theta_e_calc_mod import *

import warnings
warnings.filterwarnings('ignore')

# data directoies
dir_mcs_track = Path('/neelin2020/mcs_flextrkr/mcs_stats/')
dir_era5 = Path('/neelin2020/ERA-5/NC_FILES/')
#dir_buoy = Path('/neelin2020/ERA-5_buoy/layer_thetae/')

def get_mcs_mask(phase_list):
    
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
        mask_sub = mask_sub.sel(lon=slice(meanlon-3,meanlon+3), lat=slice(meanlat-3,meanlat+3)) # 6-deg domain
        
        # interpolate into 25 x 25 grids (equivalent to ~ 625 km)
        mask_sub_xy = mask_sub.interp(lon=np.linspace(mask_sub.lon.min(),mask_sub.lon.max(),25),
                                        lat=np.linspace(mask_sub.lat.min(),mask_sub.lat.max(),25))
        mask_sub_xy = mask_sub_xy.where(mask_sub_xy > 0, 0)
        mask_sub_xy = mask_sub_xy.where(mask_sub_xy == 0, 1)
        # converting lat-lon into x-y coordinates
        mask_sub_xy = mask_sub_xy.assign_coords(x=("lon", np.arange(0,25,1)), y=("lat", np.arange(0,25,1)))
        mask_sub_xy = mask_sub_xy.swap_dims({'lon':'x', 'lat': 'y'}).drop(['time','lat','lon'])
        
        mask_sub_phase.append(mask_sub_xy)
        
    mask_sub_phase_xr = xr.concat(mask_sub_phase, dim=pd.Index(['Init','Grow','Mature','Decay','End'], name='mcs_phase'))
    
    return mask_sub_phase_xr

def get_buoy_estimates(phase_list):
    
    BL_merged_sub_phase = []
    
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
        
        BUOY_data = xr.open_dataset('/neelin2020/ERA-5_buoy/layer_thetae/era5_2layers_thetae_{}_{}_{}.nc'.format(year,month,day))
        BUOY_sub = BUOY_data.sel(time=timestamp_phase, method='nearest')
        BUOY_sub = BUOY_sub.sel(lon=slice(meanlon_era5-3,meanlon_era5+3), lat=slice(meanlat-3,meanlat+3))
        thetae_bl = BUOY_sub.thetae_bl
        thetae_sat_lt = BUOY_sub.thetae_sat_lt
        thetae_lt = BUOY_sub.thetae_lt
        
        # parameters
        delta_pl=400
        delta_pb=100
        wb=(delta_pb/delta_pl)*np.log((delta_pl+delta_pb)/delta_pb)
        wl=1-wb
        
        BL_tot_sub = 9.81*(wb*(thetae_bl-thetae_sat_lt)/thetae_sat_lt-wl*(thetae_sat_lt-thetae_lt)/thetae_sat_lt)
        BL_tot_sub = BL_tot_sub.to_dataset(name='BL_TOT')
        BL_cape_sub = 9.81*(wb*(thetae_bl-thetae_sat_lt)/thetae_sat_lt)
        BL_cape_sub = BL_cape_sub.to_dataset(name='BL_CAPE')
        BL_subsat_sub = 9.81*(wl*(thetae_sat_lt-thetae_lt)/thetae_sat_lt)
        BL_subsat_sub = BL_subsat_sub.to_dataset(name='BL_SUBSAT')
        BL_merged_sub = xr.merge([BL_tot_sub, BL_cape_sub, BL_subsat_sub])
                
        # interpolate into 25 x 25 grids (equivalent to ~ 625 km)
        BL_merged_sub_xy = BL_merged_sub.interp(lon=np.linspace( BL_merged_sub.lon.min(), BL_merged_sub.lon.max(),25),
                                        lat=np.linspace(BL_merged_sub.lat.min(), BL_merged_sub.lat.max(),25))
        # converting lat-lon into x-y coordinates
        BL_merged_sub_xy = BL_merged_sub_xy.assign_coords(x=("lon", np.arange(0,25,1)), y=("lat", np.arange(0,25,1)))
        BL_merged_sub_xy = BL_merged_sub_xy.swap_dims({'lon':'x', 'lat': 'y'})
        
        BL_merged_sub_phase.append(BL_merged_sub_xy)
        
    BL_merged_sub_xr = xr.concat(BL_merged_sub_phase, dim=pd.Index(['Init','Grow','Mature','Decay','End']
                                                                   , name='mcs_phase')).drop_vars(['tracks','time','times','lat','lon'])
    
    return BL_merged_sub_xr

def get_pr_estimates(phase_list):
    
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
        pr_sub = 3600*pr_data.mtpr.sel(longitude=slice(meanlon_era5-3,meanlon_era5+3), latitude=slice(meanlat-3,meanlat+3)) # [mm/hr]
        # interpolate into 25 x 25 grids (equivalent to ~ 625 km)
        pr_sub_xy = pr_sub.interp(longitude=np.linspace(pr_sub.longitude.min(), pr_sub.longitude.max(),25),
                                  latitude=np.linspace(pr_sub.latitude.min(), pr_sub.latitude.max(),25))
        # converting lat-lon into x-y coordinates
        pr_sub_xy = pr_sub_xy.assign_coords(x=("longitude", np.arange(0,25,1)), y=("latitude", np.arange(0,25,1)))
        pr_sub_xy = pr_sub_xy.swap_dims({'longitude':'x', 'latitude': 'y'}).drop(['time','longitude','latitude'])
        
        # 2. get GPM-IMERG 
        gpm_data = xr.open_dataset('/neelin2020/RGMA_feature_mask/GPM_ncfiles_{}/GPM_IMERGE_V06_{}{}{}_{}00.nc'.format(
                                    year, year, month, day, hour))
        gpm_data = gpm_data.sel(time=timestamp_phase, method='nearest')
        gpm_sub = gpm_data.precipitationCal.sel(lon=slice(meanlon-3,meanlon+3), lat=slice(meanlat-3,meanlat+3))
        # swap coordinate from (lon, lat) to (lat, lon) for consistency
        gpm_sub = gpm_sub.transpose("lat", "lon")
        
        # interpolate into 25 x 25 grids (equivalent to ~ 625 km)
        gpm_sub_xy = gpm_sub.interp(lon=np.linspace(gpm_sub.lon.min(), gpm_sub.lon.max(),25),
                                    lat=np.linspace(gpm_sub.lat.min(), gpm_sub.lat.max(),25))
        # converting lat-lon into x-y coordinates
        gpm_sub_xy = gpm_sub_xy.assign_coords(x=("lon", np.arange(0,25,1)), y=("lat", np.arange(0,25,1)))
        gpm_sub_xy = gpm_sub_xy.swap_dims({'lon':'x', 'lat': 'y'}).drop(['time','lon','lat'])
        
        # merge two precip data
        pr_sub_merge_xy = xr.merge([pr_sub_xy, gpm_sub_xy])
        
        pr_sub_merge_phase.append(pr_sub_merge_xy)
        
    pr_sub_merge_phase_xr = xr.concat(pr_sub_merge_phase, dim=pd.Index(['Init','Grow','Mature','Decay','End']
                                                                   , name='mcs_phase')).drop_vars(['tracks','times'])
    
    return pr_sub_merge_phase_xr

def get_3dera5_estimates(name, var_name, phase_list):
    
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
        
        # 1. get ERA-5 precip. mtpr
        data = xr.open_dataset(dir_era5 / '{}/era-5.{}.{}.{}.nc'.format(year,name,year,month))
        
        data = data.reindex(latitude=list(reversed(data.latitude))) # reverse latitude order
        data = data.sel(time=timestamp_phase, method='nearest')
        data_sub = data[var_name].sel(longitude=slice(meanlon_era5-3,meanlon_era5+3), latitude=slice(meanlat-3,meanlat+3)) 
        # interpolate into 25 x 25 grids (equivalent to ~ 625 km)
        data_sub_xy = data_sub.interp(longitude=np.linspace(data_sub.longitude.min(), data_sub.longitude.max(),25),
                                  latitude=np.linspace(data_sub.latitude.min(), data_sub.latitude.max(),25))
        # converting lat-lon into x-y coordinates
        data_sub_xy = data_sub_xy.assign_coords(x=("longitude", np.arange(0,25,1)), y=("latitude", np.arange(0,25,1)))
        data_sub_xy = data_sub_xy.swap_dims({'longitude':'x', 'latitude': 'y'}).drop(['time','longitude','latitude'])
        
        var3d_sub_phase.append(data_sub_xy)
        
    var3d_sub_phase_xr = xr.concat(var3d_sub_phase, dim=pd.Index(['Init','Grow','Mature','Decay','End']
                                                                   , name='mcs_phase')).drop_vars(['tracks','times'])
    
    return var3d_sub_phase_xr

def get_tb_estimates(phase_list):
    
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
        tb_sub = tb_data.tb.sel(lon=slice(meanlon_era5-3,meanlon_era5+3), lat=slice(meanlat-3,meanlat+3)) 
        # interpolate into 25 x 25 grids (equivalent to ~ 625 km)
        tb_sub_xy = tb_sub.interp(lon=np.linspace(tb_sub.lon.min(), tb_sub.lon.max(),25),
                                  lat=np.linspace(tb_sub.lat.min(), tb_sub.lat.max(),25) )
        # converting lat-lon into x-y coordinates
        tb_sub_xy = tb_sub_xy.assign_coords(x=("lon", np.arange(0,25,1)), y=("lat", np.arange(0,25,1)))
        tb_sub_xy = tb_sub_xy.swap_dims({'lon':'x', 'lat': 'y'}).drop(['time','lon','lat'])
        
        tb_sub_phase.append(tb_sub_xy)
        
    tb_sub_phase_xr = xr.concat(tb_sub_phase, dim=pd.Index(['Init','Grow','Mature','Decay','End']
                                                                   , name='mcs_phase')).drop_vars(['tracks','times'])
    
    return tb_sub_phase_xr

# load processed non2mcs tracks

if __name__ == '__main__':

    year = sys.argv[1]

    data_non2mcs_complete = xr.open_dataset(dir_mcs_track / 'mcs_tracks_non2mcs_{}.IndoPacific.amp.nc'.format(year))

    out_dir = Path('/neelin2020/mcs_flextrkr/mcs_stats/envs_track/{}'.format(year))
    if out_dir.exists() == False:
        os.system('mkdir -p {}'.format(out_dir))

    for n, track in enumerate(data_non2mcs_complete.tracks.values):

        track_number = track # cloudtracknumber in PyFLEXTRKR

        # get phase_list 
        phase_list = [data_non2mcs_complete.sel(tracks=track_number).idt_mcs_init.values, 
                      data_non2mcs_complete.sel(tracks=track_number).idt_mcs_grow.values,
                      data_non2mcs_complete.sel(tracks=track_number).idt_mcs_mature.values,
                      data_non2mcs_complete.sel(tracks=track_number).idt_mcs_decay.values,
                      data_non2mcs_complete.sel(tracks=track_number).idt_mcs_end.values]

        mcs_mask_phase_xr = get_mcs_mask(phase_list)
        try:
            BL_phase_xr = get_buoy_estimates(phase_list)
            prec_phase_xr = get_pr_estimates(phase_list)
            T3d_phase_xr = get_3dera5_estimates(name='T', var_name='t', phase_list=phase_list)
            q3d_phase_xr = get_3dera5_estimates(name='q', var_name='q', phase_list=phase_list)
            u3d_phase_xr = get_3dera5_estimates(name='ua', var_name='u', phase_list=phase_list)
            v3d_phase_xr = get_3dera5_estimates(name='va', var_name='v', phase_list=phase_list)
            w3d_phase_xr = get_3dera5_estimates(name='omega', var_name='w', phase_list=phase_list)
            tb_phase_xr = get_tb_estimates(phase_list)

            # merge all variables into a single dataset
            vars_mcsenvs_xr = xr.merge([mcs_mask_phase_xr, prec_phase_xr, BL_phase_xr, tb_phase_xr,
                                    T3d_phase_xr, q3d_phase_xr, u3d_phase_xr, v3d_phase_xr, w3d_phase_xr])

            # write out as netcdf file
            vars_mcsenvs_xr.to_netcdf(out_dir / 'mcs_era5_3D_envs_{}.{}.amp.nc'.format(year, str(track_number).zfill(5)))
            print('mcs_era5_3D_envs_{}.{}.amp.nc ...({} / {}) saved'.format(year, str(track_number).zfill(5),
                                                                        n+1, len(data_non2mcs_complete.tracks)))
        except:
            print('error due to missing variables: mcs_era5_3D_envs_{}.{}.amp.nc'.format(year, str(track_number).zfill(5)))


