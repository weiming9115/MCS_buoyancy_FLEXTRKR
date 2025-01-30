import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

year = int(sys.argv[1]) # get year integer

# data directoies
dir_mcs_track = Path('/neelin2020/mcs_flextrkr/mcs_stats/')
dir_era5 = Path('/neelin2020/ERA-5/NC_FILES/')
dir_buoy = Path('/neelin2020/ERA-5_buoy/layer_thetae/')

data_track = xr.open_dataset(dir_mcs_track / 'mcs_tracks_final_extc_{}0101.0000_{}0101.0000.nc'.format(year,year+1))
                                 
# convection over Indian Ocean [50E-90E, 20S-20N]
meanlat = data_track.meanlat.sel(times=0)
idx_lat = meanlat.where((meanlat > -20) & (meanlat < 20)).dropna(dim='tracks').tracks
meanlon = data_track.meanlon.sel(times=0)
idx_lon = meanlon.where((meanlon > 50) & (meanlon < 180)).dropna(dim='tracks').tracks
idx_reg = np.intersect1d(idx_lat, idx_lon) # tracks starting in the selected region

data_sub = data_track.sel(tracks=idx_reg)
                                 
nonmcs_hours = data_sub.mcs_status.sel(times=slice(0,4)).sum(dim='times') 
mcs_hours = data_sub.mcs_status.sel(times=slice(5,400)).sum(dim='times')
idx = np.where(nonmcs_hours == 0)[0]
data_non2mcs = data_sub.isel(tracks=idx)
                                 
## generate time indices for tracks showing complete MCS lifetimes
track_list = []

for track in data_non2mcs.tracks.values:

    tmp = data_non2mcs.sel(tracks=track).mcs_status
    tmp2 = data_non2mcs.sel(tracks=track).total_rain/(data_non2mcs.sel(tracks=track).area/100)
    idt_mcs_init = np.where(tmp == 1)[0][0]
    idt_mcs_mature = np.where(tmp2 == tmp2.max('times'))[0][0]
    idt_mcs_end = np.where(tmp == 1)[0][-1]

    mcs_duration = data_non2mcs.sel(tracks=track).mcs_duration.values
        
    cond1 = ((idt_mcs_end - idt_mcs_init + 1) == mcs_duration)
    cond2 = (idt_mcs_end > idt_mcs_mature) 
    cond3 = (idt_mcs_init < idt_mcs_mature)
    cond4 = (tmp.sel(times=idt_mcs_end+1) == 0)

    if (cond1 & cond2 & cond3 & cond4):
                
        idt_mcs_grow = idt_mcs_init + (idt_mcs_mature - idt_mcs_init)//2
        idt_mcs_decay = idt_mcs_mature + (idt_mcs_end - idt_mcs_mature)//2

        ds = xr.Dataset(data_vars=dict(
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

########## merge phase indices and area-averaged total rain rate from ERA-5
ds_tracks_list = []

for i, track_number in enumerate(data_non2mcs_complete.tracks.values):

    track_duration = data_non2mcs_complete.sel(tracks=track_number).track_duration.values
    base_time = data_non2mcs_complete.sel(tracks=track_number).base_time.values
    meanlat = data_non2mcs_complete.sel(tracks=track_number).meanlat.values
    meanlon = data_non2mcs_complete.sel(tracks=track_number).meanlon.values

    mpr_area_mn = np.zeros(400)*np.nan
    saved_list = 0

    for n, (timestamp, mnlat, mnlon) in enumerate(zip(base_time, meanlat, meanlon)):

        timestamp_str = str(timestamp)

        if timestamp_str != 'NaT': # excluding nan

            year = timestamp_str[:4]
            month = timestamp_str[5:7]
            day = timestamp_str[8:10]
            hour = timestamp_str[11:13]

            # read the file containing the binary mask
            dir_mask = Path('/neelin2020/mcs_flextrkr/{}0101.0000_{}0101.0000/'.format(year,int(year)+1))
            file = list(dir_mask.glob('*{}{}{}_{}*.nc'.format(year,month,day,hour)))[0] # find the corresponding file at the given time
            data_mcsmask = xr.open_dataset(file)
            lon_reset = data_mcsmask.lon
            lon_reset = lon_reset.where(lon_reset >= 0, 360+lon_reset) # converting lon as 0 to 359.75
            data_mcsmask.coords['lon'] = lon_reset # converting lon as -180 to 180
            data_mcsmask= data_mcsmask.sortby('lon')        

            # read the file containing era-5 mean total rain rate 
            dir_mpr = dir_era5 / '{}'.format(year)
            file = list(dir_mpr.glob('era-5.mpr.{}.{}.nc'.format(year, month)))[0]
            data_mpr = xr.open_dataset(file)
            # match the latitude range of two datasets
            data_mpr = data_mpr.reindex(latitude=list(reversed(data_mpr.latitude))) # reverse the order of latitude 
            data_mpr = data_mpr.mtpr.sel(latitude=slice(-60,60))
            data_mpr = data_mpr.sel(time=datetime(int(year),int(month),int(day),int(hour)), method='nearest').drop('time')
            # interpolating into mcs grids
            data_mpr = data_mpr.interp(longitude=data_mcsmask.lon, latitude=data_mcsmask.lat)

#            mcstracknumber = data_mcsmask.cloudtracknumber_nomergesplit.sel(lat=mnlat, lon=mnlon, method='nearest').values 
            # calculate the area mean of ERA-5 total rain rate
            mcs_mask = data_mcsmask.cloudtracknumber_nomergesplit.drop('time')
            mpr_area_mn[n] = 3600*data_mpr.where(mcs_mask == track_number + 1).mean().values # averaging values within the coarse-grained mask (mm/hr)

            saved_list += 1

    if saved_list == int(track_duration):
        print('nubmer of values matching track_duration... OK! ({} / {})'.format(i, len(data_non2mcs_complete.tracks)))
    else:
        raise ValueError('the number of values does not match the track duration...check')

    # create xarray 
    ds_single_track = xr.Dataset(data_vars=dict(mean_total_rain_era5 = (['times'], mpr_area_mn)),
                               coords=dict(times = (['times'], np.arange(400)))
                              )
    
    ds_tracks_list.append(ds_single_track)
 
ds_mpr_tracks_xr = xr.concat(ds_tracks_list, dim=pd.Index(data_non2mcs_complete.tracks.values, name='tracks'))

# save merged dataset into the directory
dir_out = Path('/neelin2020/mcs_flextrkr/mcs_stats/')

### revised --> can be calculated based on full data: mcs_3D_envs....nc
ds_tracks_merged = xr.merge([data_non2mcs_complete, data_non2mcs_phase, ds_mpr_tracks_xr])
#ds_tracks_merged = xr.merge([data_non2mcs_complete, data_non2mcs_phase])
ds_tracks_merged.to_netcdf(dir_out / 'mcs_tracks_non2mcs_{}.IndoPacific.amp.nc'.format(year))

