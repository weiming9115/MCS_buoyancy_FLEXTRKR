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
    dir_out = Path('/pscratch/sd/w/wmtsai/featenv_analysis/data/EC-Earth3P-HR_highresSST-present_r3i1p1f1/mcs_inputs/')
    ds = xr.open_dataset(dir_out / 'mcs_tracks_non2mcs_{}.tropics30NS.stablemcs.nc'.format(year), decode_times=False)
   
    ds_list = []
    for track in ds.tracks:
        ds_sub = ds.sel(tracks=track)
        idt_mcs_phase = [ds_sub.idt_ccs_init, ds_sub.idt_mcs_init, ds_sub.idt_mcs_grow,
                         ds_sub.idt_mcs_mature, ds_sub.idt_mcs_decay, ds_sub.idt_mcs_end]
        ds_sub_phase = ds_sub.isel(times=idt_mcs_phase)
        ds_sub_extract = ds_sub_phase[['base_time','meanlat','meanlon']]
        ds_sub_extract.coords['times'] = np.arange(6)
        ds_sub_extract = ds_sub_extract.rename({'times': 'time'})
        ds_list.append(ds_sub_extract)

    # merge all tracks
    ds_merged = xr.concat(ds_list, dim=pd.Index(ds.tracks.values, name='tracks'))
    ds_merged.to_netcdf(dir_out / 'mcs_tracks_input.{}.nc'.format(year))
    print(dir_out / 'mcs_tracks_input.{}.nc'.format(year))
