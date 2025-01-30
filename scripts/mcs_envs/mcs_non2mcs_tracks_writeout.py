import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

def data_tracks_add_LSflag(data_tracks):
    """
    add landsea flag
    """

    lsflag_array = np.zeros((len(data_tracks.tracks),len(data_tracks.times)))*np.nan
    # ERA-5 lat-lon coordinate, 0.25-deg.
    data_lsmask = xr.open_dataset('/neelin2020/RGMA_feature_mask/ERA5_LandSeaMask_regrid.nc4')
    landseamask = data_lsmask.landseamask

    for (n,track) in enumerate(data_tracks.tracks.values):

        meanlon = data_tracks.sel(tracks=track).meanlon.values
        meanlat = data_tracks.sel(tracks=track).meanlat.values

        for t,(lon, lat) in enumerate(zip(meanlon, meanlat)):

            if np.isnan(lon) == 0: # not NaN

                if (lon < 0):
                    lon = lon + 360 # converting to 0-360
                
                landsea_flag = landseamask.sel(longitude=lon, latitude=lat, method='nearest').values

                if landsea_flag == 100: # ocean area (longitude, latitude)
                    lsflag_array[n,t] = 0
                else: # land area
                    lsflag_array[n,t] = 1
                    
    # write a new variable
    data_tracks['landsea_flag'] = data_tracks.meanlon*0 + lsflag_array
    data_tracks['landsea_flag'] = data_tracks['landsea_flag'].assign_attrs(units="0 = ocean; 1 = land", description="landsea flag")

    return data_tracks

########################## Main Code ################################

if __name__ == '__main__':

    year = int(sys.argv[1]) # get year integer

    # data directoies
    dir_mcs_track = Path('/neelin2020/mcs_flextrkr/mcs_stats/')
    # read data
    data_track = xr.open_dataset(dir_mcs_track / 'mcs_tracks_final_extc_{}0101.0000_{}0101.0000.nc'.format(year,year+1))
                                 
    # convection over the tropics [30S-30N]
    meanlat = data_track.meanlat.sel(times=0)
    idx_lat = meanlat.where((meanlat > -30) & (meanlat < 30)).dropna(dim='tracks').tracks
    meanlon = data_track.meanlon.sel(times=0)
    idx_lon = meanlon.where((meanlon > -180) & (meanlon < 180)).dropna(dim='tracks').tracks
    idx_reg = np.intersect1d(idx_lat, idx_lon) # tracks starting in the selected region

    data_sub = data_track.sel(tracks=idx_reg)

    ############## non2mcs options: CCS for at least 3hrs; MCS duration >= 5 hrs                              
    nonmcs_hours = data_sub.mcs_status.sel(times=[0,1,2]).sum(dim='times') 
    mcs_duration = data_sub.mcs_duration
    idx = np.where(np.logical_and(nonmcs_hours == 0, mcs_duration >=5))[0]
    data_non2mcs = data_sub.isel(tracks=idx)
    ##############################
                                
    ## generate time indices for tracks showing complete MCS lifetimes
    track_list = []

    for track in data_non2mcs.tracks.values:

        tmp = data_non2mcs.sel(tracks=track).mcs_status
        tmp2 = data_non2mcs.sel(tracks=track).total_rain
        idt_mcs_init = np.where(tmp == 1)[0][0]
        idt_mcs_mature = np.where(tmp2 == tmp2.max('times'))[0][0]
        idt_mcs_end = np.where(tmp == 1)[0][-1]

        mcs_duration = data_non2mcs.sel(tracks=track).mcs_duration.values
        
        cond1 = ((idt_mcs_end - idt_mcs_init + 1) == mcs_duration)
        cond2 = (idt_mcs_end > idt_mcs_mature) 
        cond3 = (idt_mcs_init < idt_mcs_mature)
        cond4 = (tmp.sel(times=idt_mcs_end+1) == 0)

        if (cond1 & cond2 & cond3 & cond4):
        
            idt_ccs_init = 0 # start as CCS        
            idt_mcs_grow = idt_mcs_init + (idt_mcs_mature - idt_mcs_init)//2
            idt_mcs_decay = idt_mcs_mature + (idt_mcs_end - idt_mcs_mature)//2

            if (idt_mcs_mature > idt_mcs_init + 1) & (idt_mcs_end > idt_mcs_mature + 1):

                ds = xr.Dataset(data_vars=dict(
                       idt_ccs_init=(['tracks'], [idt_ccs_init]),
                       idt_mcs_init=(['tracks'], [idt_mcs_init]),
                       idt_mcs_grow=(['tracks'], [idt_mcs_grow]),
                       idt_mcs_mature=(['tracks'], [idt_mcs_mature]),
                       idt_mcs_decay=(['tracks'], [idt_mcs_decay]),
                       idt_mcs_end=(['tracks'], [idt_mcs_end])
                       ),
                       coords=dict(tracks=(['tracks'],[track])))

                track_list.append(ds)

    data_non2mcs_phase = xr.concat(track_list, dim='tracks')                           
    data_non2mcs_complete = data_non2mcs.sel(tracks=data_non2mcs_phase.tracks)

    # 1. merge original data variables and phase timestamps
    ds_tracks_merged = xr.merge([data_non2mcs_complete, data_non2mcs_phase])
    # 2. add landsea flag
    ds_tracks_merged_flag = data_tracks_add_LSflag(ds_tracks_merged)

    # save merged dataset into the directory
    dir_out = Path('/scratch/wmtsai/temp_mcs/mcs_stats/mcs_tracks_non2mcs')
    ds_tracks_merged_flag.to_netcdf(dir_out / 'mcs_tracks_non2mcs_{}.tropics30NS.full.nc'.format(year))

