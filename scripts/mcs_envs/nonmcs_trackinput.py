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

    year = int(sys.argv[1]) # get year 
    dir_in = Path('/scratch/wmtsai/temp_mcs/mcs_stats/mcs_tracks_non2mcs')
    dir_out = Path('/scratch/wmtsai/temp_mcs/mcs_stats/mcs_tracks_non2mcs/featenv_track_input')
    ds = xr.open_dataset(dir_in / 'nonmcs_tracks_{}.tropics30NS.nc'.format(year), decode_times=False)
   
    ds_list = []
    for track in ds.tracks:
        ds_sub = ds.sel(tracks=track)
        idt_ccs_phase = [ds_sub.idt_ccs_init, ds_sub.idt_ccs_grow,
                         ds_sub.idt_ccs_mature, ds_sub.idt_ccs_decay, ds_sub.idt_ccs_end]
        ds_sub_phase = ds_sub.isel(times=idt_ccs_phase)
        ds_sub_extract = ds_sub_phase[['base_time','meanlat','meanlon']]
        ds_sub_extract.coords['times'] = np.arange(5)
        ds_sub_extract = ds_sub_extract.rename({'times': 'time'})
        ds_list.append(ds_sub_extract)

    # merge all tracks
    ds_merged = xr.concat(ds_list, dim=pd.Index(ds.tracks.values, name='tracks'))
    # converting longitude into 0-360
    meanlon = ds_merged.meanlon
    ds_merged['meanlon'] = meanlon.where(meanlon >=0, meanlon + 360)
    ds_merged.to_netcdf(dir_out / 'nonmcs_tracks_input.{}.nc'.format(year))
    print(dir_out / 'nonmcs_tracks_input.{}.nc'.format(year))

