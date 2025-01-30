import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    year = int(sys.argv[1]) # get year integer

    # data directoies
    dir_mcs_track = Path('/neelin2020/mcs_flextrkr/mcs_stats/')
    # read data
    data_track = xr.open_dataset(dir_mcs_track / 'mcs_tracks_final_extc_{}0101.0000_{}0101.0000.nc'.format(year,year+1))
                                 
    ##############################
    # 1. first detected over the tropics [30S-30N] 
    meanlat = data_track.meanlat.sel(times=0)
    idx_lat = meanlat.where((meanlat > -30) & (meanlat < 30)).dropna(dim='tracks').tracks.values
    meanlon = data_track.meanlon.sel(times=0)
    data_sub = data_track.sel(tracks=idx_lat)

    # 2. non2mcs options: CCS for at least 3hrs; MCS duration >= 5 hrs                              
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
      
        # 3. stable MCS status (uninterrupted mcs_status == 1) throghout its all life time
        #    np.sum(mcs_status) == mcs_duration
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

    data_stablemcs_phase = xr.concat(track_list, dim='tracks') # timestamp information of stable MCSs
    # select stable MCSs from non2mcs 
    data_stablemcs_complete = data_non2mcs.sel(tracks=data_stablemcs_phase.tracks) 
    # merge two datasets into one as output   
    ds_tracks_merged = xr.merge([data_stablemcs_complete, data_stablemcs_phase])

    # save merged dataset into the directory
    dir_out = Path('/scratch/wmtsai/temp_mcs/mcs_stats/mcs_tracks_non2mcs')
    ds_tracks_merged.to_netcdf(dir_out / 'mcs_tracks_non2mcs_{}.tropics30NS.stablemcs.nc'.format(year))

