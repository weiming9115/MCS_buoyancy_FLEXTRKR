import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import random
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    year = int(sys.argv[1])

    # data directoies
    dir_ccs_track = Path('/scratch/wmtsai/temp_mcs/mcs_stats')
    # read data
    data_track = xr.open_dataset(dir_ccs_track / 'trackstats_{}0101.0000_{}0101.0000.nc'.format(year,year+1))

    ##############################
    # 1. first detected over the tropics [30S-30N]
    meanlat = data_track.meanlat.sel(times=0)
    idx_lat = meanlat.where((meanlat > -30) & (meanlat < 30)).dropna(dim='tracks').tracks.values
    meanlon = data_track.meanlon.sel(times=0)
    data_sub = data_track.sel(tracks=idx_lat)

    # 2. non2mcs options: CCS for at least 3hrs; MCS duration >= 5 hrs
    start_status = data_sub.start_status
    end_status = data_sub.end_status
    track_duration = data_sub.track_duration
    idx = np.where(np.logical_and(start_status == 1, track_duration >=5))[0]
    idy = np.where(end_status == 0)[0]
    idx_comb = np.intersect1d(idx,idy)
    data_nonmcs = data_sub.isel(tracks=idx_comb)

    ## generate time indices for tracks showing complete MCS lifetimes
    track_list = []

    for track in data_nonmcs.tracks.values:

        tmp = data_nonmcs.sel(tracks=track).track_status
        tmp2 = data_nonmcs.sel(tracks=track).area
        idt_init = np.where(tmp >= 0)[0][0]
        idt_mature = np.where(tmp2 == tmp2.max('times'))[0][0]
        idt_end = np.where(tmp >= 0)[0][-1]

        track_duration = data_nonmcs.sel(tracks=track).track_duration.values

        # 3. stable MCS status (uninterrupted mcs_status == 1) throghout its all life time
        #    np.sum(mcs_status) == mcs_duration
        cond1 = ((idt_end - idt_init + 1) == track_duration)
        cond2 = (idt_end > idt_mature)
        cond3 = (idt_init < idt_mature)
        #cond4 = (tmp.sel(times=idt_end+1) == 0)

        if (cond1 & cond2 & cond3):
            
            idt_init = 0 # start as CCS
            idt_grow = idt_init + (idt_mature - idt_init)//2
            idt_decay = idt_mature + (idt_end - idt_mature)//2

            if (idt_mature > idt_init + 1) & (idt_end > idt_mature + 1):

                ds = xr.Dataset(data_vars=dict(
                        idt_ccs_init=(['tracks'], [idt_init]),
                        idt_ccs_grow=(['tracks'], [idt_grow]),
                        idt_ccs_mature=(['tracks'], [idt_mature]),
                        idt_ccs_decay=(['tracks'], [idt_decay]),
                        idt_ccs_end=(['tracks'], [idt_end])
                        ),
                        coords=dict(tracks=(['tracks'],[track])))

                track_list.append(ds)

    data_stableccs_phase = xr.concat(track_list, dim='tracks') # timestamp information of stable MCSs
    # select stable MCSs from non2mcs
    data_stableccs_complete = data_nonmcs.sel(tracks=data_stableccs_phase.tracks)
    # merge two datasets into one as output
    ds_tracks_merged = xr.merge([data_stableccs_complete, data_stableccs_phase])

    # save merged dataset into the directory, chopping into several files 
    dir_out = Path('/scratch/wmtsai/temp_mcs/mcs_stats/mcs_tracks_non2mcs')
    tracks_sel = ds_tracks_merged.tracks.values 
    random.shuffle(ds_tracks_merged.tracks.values) # randomly selecting 5000 tracks and saved for each year
    ds_tracks_sampled = ds_tracks_merged.sel(tracks=tracks_sel[:5000]).sortby('tracks')
    ds_tracks_sampled.to_netcdf(dir_out / 'nonmcs_tracks_{}.tropics30NS.nc'.format(year))
    print(dir_out / 'nonmcs_tracks_{}.tropics30NS.nc'.format(year))
