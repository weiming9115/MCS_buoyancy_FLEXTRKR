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

import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.util import add_cyclic_point
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":

    #### connecting to other ERA-5 variables provided from PNNL
    # 3-deg area-averaged values

    era5_env_dir = Path('/neelin2020/mcs_flextrkr/era5_envs')
    mcs_dir = Path('/scratch/wmtsai/temp_mcs/mcs_stats/mcs_tracks_non2mcs')
    year = int(sys.argv[1])

    print('processing year:', year)
    data_era5_env = xr.open_dataset(era5_env_dir / 'mcs_era5_mean_envs_{}0101.0000_{}0101.0000.nc'.format(year,year+1))
    # get tropical tracks according to mcs_stats
    data_stats = xr.open_dataset(mcs_dir / 'mcs_tracks_non2mcs_{}.tropics30NS.full.nc'.format(year))
    track_sel = data_stats.tracks
    era5_env_tropics = data_era5_env.sel(tracks=track_sel)
    # get mcs_phase time indices
    ds_env_phase = []
    for track in era5_env_tropics.tracks:
        ds_track_env = era5_env_tropics.sel(tracks=track)
        ds_track_stat = data_stats.sel(tracks=track)
        
        times_phase = [ds_track_stat.idt_ccs_init.values, ds_track_stat.idt_mcs_init.values, ds_track_stat.idt_mcs_grow.values,
                    ds_track_stat.idt_mcs_mature.values, ds_track_stat.idt_mcs_decay.values, ds_track_stat.idt_mcs_end.values]

        tmp = ds_track_env.sel(rel_times=times_phase).rename({'rel_times':'mcs_phase'}) # defined MCS life stages
        tmp['mcs_phase'] = ["CCS", "Init", "Grow", "Mature", "Decay", "End"]
        ds_env_phase.append(tmp)

    ds_env_merged = xr.concat(ds_env_phase, dim='tracks')

    out_dir = Path('/scratch/wmtsai/temp_mcs/mcs_stats/mcs_tracks_non2mcs/tracks_stats_phase')
    ds_env_merged.to_netcdf(out_dir / 'envERA5_tracks_non2mcs_{}.tropics30NS.full.nc'.format(year))
    print(out_dir / 'envERA5_tracks_non2mcs_{}.tropics30NS.full.nc'.format(year))
