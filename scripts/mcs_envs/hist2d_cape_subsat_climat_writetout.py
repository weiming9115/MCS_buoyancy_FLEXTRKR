import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def get_files_HCC(year, corr_temp_cri, corr_space_cri):
    
    data = xr.open_dataset(featstats_dir / 'featstats_tracks_non2mcs_{}.tropics30NS.nc'.format(year))
    
    corr_coeff_temp = data.corr_coeff_temp
    corr_coeff_space = data.corr_coeff_space.mean('mcs_phase')
    
    cond_1 = corr_coeff_temp > corr_temp_cri
    cond_2 = corr_coeff_space > corr_space_cri
    track_sel = data.isel(tracks=np.where(cond_1 & cond_2)[0]).tracks
    
    files_HCC = []
    for track in track_sel.values:
        files_HCC.extend(list(dir_envs_track.glob('mcs_era5_3D_envs_{}.{}.LD.nc'.format(year
                                                                    , str(track).zfill(5)))))
    return files_HCC

def get_files_duration(year, duration_min, duration_max):
    
    data = xr.open_dataset(featstats_dir / 'featstats_tracks_non2mcs_{}.tropics30NS.nc'.format(year))
    
    mcs_duration = data.mcs_duration
    
    cond_1 = mcs_duration >= duration_min
    cond_2 = mcs_duration < duration_max
    track_sel = data.isel(tracks=np.where(cond_1 & cond_2)[0]).tracks
    
    files_HCC = []
    for track in track_sel.values:
        files_HCC.extend(list(dir_envs_track.glob('mcs_era5_3D_envs_{}.{}.LD.nc'.format(year
                                                                    , str(track).zfill(5)))))
    return files_HCC


def cape_subsat_hist(files, multi_year=False):
    
    # bins for BL_CAPE and BL_SUBSAT
    bins_cape = np.linspace(-20,30,101)
    bins_subsat = np.linspace(-20,30,101)
    bins_samples = np.zeros((5, len(bins_cape)-1, len(bins_subsat)-1)) # (mcs_phase, cape, subsat)
    
    n = 0
    track_list = []
    for file in files: # total files read (single year or multiple years)
        
        if multi_year == True:
            n += 1 # total track numbers merged
            
        track_list.append(int(file.name[-11:-6])) # save track number 
      
        track_info =  data_track.sel(tracks=int(file.name[-11:-6]))
        idt_phase = [track_info.idt_mcs_init.values, track_info.idt_mcs_grow.values,
                     track_info.idt_mcs_mature.values, track_info.idt_mcs_decay.values,
                     track_info.idt_mcs_end.values]
        timestamp_phase = track_info.sel(times=idt_phase).base_time.values # get numpy64.datetime info
        meanlon_phase = track_info.sel(times=idt_phase).meanlon.values # get meanlon at phases
        meanlat_phase = track_info.sel(times=idt_phase).meanlat.values # get meanlat at phases

        for p, (timestamp,meanlon,meanlat) in enumerate(zip(timestamp_phase, meanlon_phase, meanlat_phase)):

            month = pd.to_datetime(timestamp).month 
            if meanlon < 0:
                meanlon = meanlon + 360

            # read monthly buoyancy file 
            data_BL = data_BL_monthly.sel(year=year,month=month)
            # parameters converting the unit from (m/s^2) to (Kelvin)
            delta_pl=400
            delta_pb=100
            wb=(delta_pb/delta_pl)*np.log((delta_pl+delta_pb)/delta_pb)
            wl=1-wb

            data_BL = data_BL.sel(lat=slice(meanlat-2.5,meanlat+2.5), lon=slice(meanlon-2.5,meanlon+2.5)).mean(('lat','lon'))
            BL_CAPE_phase_month = (340*3)/9.8/wb*data_BL.Buoy_CAPE  # (K)
            BL_SUBSAT_phase_month = (340*3)/9.8/wl*data_BL.Buoy_SUBSAT # (K)

            # get 1-D CAPE and SUBSAT values of the monthly averages as climatology
            cape_value = BL_CAPE_phase_month.values
            subsat_value = BL_SUBSAT_phase_month.values

            idx = np.argmin(abs(cape_value-(bins_cape+0.25)))
            idy = np.argmin(abs(subsat_value-(bins_subsat+0.25)))

            bins_samples[p,idx,idy] += 1

    if multi_year == False:
    
        ds_bins = xr.Dataset(data_vars = dict(samples = (['phase','bins_cape','bins_subsat'], bins_samples)),
                 coords = dict(tracks = track_list,
                               phase = (['phase'], ['Initial', 'Grow', 'Mature', 'Decay', 'End']),
                               bins_cape = (['bins_cape'], bins_cape[:-1]),
                               bins_subsat = (['bins_subsat'], bins_subsat[:-1])),
                 attrs = dict(description = 'cape-subsat histogram. monthly mean reference 5-deg average. BL measures of the monthly mean at that 5-deg box are counted as reference'))
    else:
        ds_bins = xr.Dataset(data_vars = dict(samples = (['phase','bins_cape','bins_subsat'], bins_samples)),
                 coords = dict(tracks = np.arange(n),
                               phase = (['phase'], ['Initial', 'Grow', 'Mature', 'Decay', 'End']),
                               bins_cape = (['bins_cape'], bins_cape[:-1]),
                               bins_subsat = (['bins_subsat'], bins_subsat[:-1])),
                 attrs = dict(description = 'cape-subsat histogram. monthly mean reference 5-deg average. BL measures of the monthly mean at that 5-deg box are counted as reference'))
    
    return ds_bins

if __name__ == '__main__':

    # separate by mcs duration groups
    year = int(sys.argv[1])

    track_dir = Path('/scratch/wmtsai/temp_mcs/mcs_stats/mcs_tracks_non2mcs')
    featstats_dir = Path('/scratch/wmtsai/temp_mcs/mcs_stats/mcs_tracks_non2mcs/tracks_area_mean/')
    dir_envs_track = Path('/scratch/wmtsai/temp_mcs/mcs_stats/envs_track/{}/tropics'.format(year))
    buoy_dir = Path('/neelin2020/mcs_flextrkr/mcs_stats/output_stats')

    # load monthly average BL measures 
    data_BL_monthly = xr.open_dataset(buoy_dir / 'era5_BL_measures_monthly_avg.2002-2014.nc')

    data_bins_merged = []
    for (dmin, dmax, duration_type) in zip([5,6,12,18,24],[6,12,18,24,72],
                                           ['SL','ML','LL','UL','UUL']):

        # selecting files 
        files_HCC = get_files_HCC(year, corr_temp_cri=0, corr_space_cri=0)
        files_duration = get_files_duration(year, duration_min=dmin, duration_max=dmax)
        files_comb = list(set(files_HCC).intersection(files_duration))
        print('processsing year: {}'.format(year))
        print('number of selected tracks: {}'.format(len(files_comb)))

        # non2mcs track data 
        data_track = xr.open_dataset(track_dir / 'mcs_tracks_non2mcs_{}.tropics30NS.nc'.format(year))
        data_bins_dtype = cape_subsat_hist(files_comb, multi_year=False)
        data_bins_merged.append(data_bins_dtype)

    data_bins_duration = xr.concat(data_bins_merged, pd.Index(['SL','ML','LL','UL','UUL'], name='duration_type'))

    out_dir = Path('/scratch/wmtsai/temp_mcs/output_stats')
    data_bins_duration.to_netcdf(out_dir / 'hist2d_cape_subsat_dtype.{}.all.climat.nc'.format(year))
    print('hist2d_cape_subsat_dtype.{}.all.climat.nc .... done'.format(year))
