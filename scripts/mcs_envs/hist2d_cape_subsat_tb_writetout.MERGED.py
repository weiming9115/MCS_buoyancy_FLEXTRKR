import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def data_filterby_duration(data_stats, duration_min, duration_max):
    
    mcs_duration = data_stats.mcs_duration   
    cond_1 = mcs_duration >= duration_min
    cond_2 = mcs_duration < duration_max
    track_sel = data_stats.isel(tracks=np.where(cond_1 & cond_2)[0]).tracks.values
    
    # return subset of tracks matching the duration range
    return data_stats.sel(tracks=track_sel)

def data_filterby_landsea(data_stats, sampling_opt='all'):
   
    """
    sampling option to filter out MCS tracks by genesis locations: 'all', 'ocean', 'land'
    """
    idt_mcs_init = data_stats.idt_mcs_init
    # determined by the location when MCS is identified
    landsea_flag = data_stats.landsea_flag.sel(times=idt_mcs_init)

    if sampling_opt == 'all':
        track_sel = data_stats.tracks.values
    elif sampling_opt == 'land':
        idx_sel = np.where(landsea_flag == 1)[0]
        track_sel = data_stats.isel(tracks=idx_sel).tracks.values
    elif sampling_opt == 'ocean':
        idx_sel = np.where(landsea_flag == 0)[0]
        track_sel = data_stats.isel(tracks=idx_sel).tracks.values
    
    # return a subset of tracks matching the landsea filter
    return data_stats.sel(tracks=track_sel)

def BL_mcs_2dmap(data_merged_phase):
    """
    input: processed envs_track file containing MCS feature mask and 2D/3D variables on ERA-5 coordinates
    return BL componets within the tracked MCS feature
    """
    BL_TOT = data_merged_phase.Buoy_TOT
    BL_CAPE = data_merged_phase.Buoy_CAPE
    BL_SUBSAT = data_merged_phase.Buoy_SUBSAT
    
    # BL associated with mcs / non-mcs grids
    mcs_mask = data_merged_phase.cloudtracknumber_nomergesplit.fillna(0) # mask (track_number or 0)
    BL_TOT_mcs = BL_TOT.where(mcs_mask > 0).rename('BL_TOT_mcs')
    BL_CAPE_mcs = BL_CAPE.where(mcs_mask > 0).rename('BL_CAPE_mcs')
    BL_SUBSAT_mcs = BL_SUBSAT.where(mcs_mask > 0).rename('BL_SUBSAT_mcs')
    
    BL_TOT_env = BL_TOT.where(mcs_mask == 0).rename('BL_TOT_env')
    BL_CAPE_env = BL_CAPE.where(mcs_mask == 0).rename('BL_CAPE_env')
    BL_SUBSAT_env = BL_SUBSAT.where(mcs_mask == 0).rename('BL_SUBSAT_env')
    
    return xr.merge([BL_TOT_mcs, BL_CAPE_mcs, BL_SUBSAT_mcs, BL_TOT_env, BL_CAPE_env, BL_SUBSAT_env])

def cape_subsat_hist(data_merged_phase):
    
    # bins for BL_CAPE and BL_SUBSAT
    bins_cape = np.arange(-15,10,0.5)
    bins_subsat = np.arange(-5,25,0.5)
    bins_samples = np.zeros((3, 6, len(bins_cape)-1, len(bins_subsat)-1)) # (area_type, mcs_phase, cape, subsat)
    tb_sum = np.copy(bins_samples)
        
    data_BLenvmcs_phase = BL_mcs_2dmap(data_merged_phase)

    for p, phase in enumerate(["CCS","Init", "Grow", "Mature", "Decay", "End"]):

        # ===== for inside mcs ======
        BL_CAPE_phase_mcs = data_BLenvmcs_phase.BL_CAPE_mcs.sel(mcs_phase=phase) # 9.81/(340*3)*wb*((thetae_bl-thetae_sat_lt)/thetae_sat_lt)
        BL_CAPE_phase_mcs = (340*3/9.81)*BL_CAPE_phase_mcs # wb * (theta....)
        BL_SUBSAT_phase_mcs = data_BLenvmcs_phase.BL_SUBSAT_mcs.sel(mcs_phase=phase) # 9.81/(340*3)*wl*((thetae_sat_lt-thetae_lt)/thetae_sat_lt)
        BL_SUBSAT_phase_mcs = (340*3/9.81)*BL_SUBSAT_phase_mcs # wl * (theta...)
        
        # get 1-D CAPE and SUBSAT values associated with MCS from all tracks
        cape_1d = BL_CAPE_phase_mcs.values.ravel()
        subsat_1d = BL_SUBSAT_phase_mcs.values.ravel()
        tb = data_merged_phase.tb.sel(mcs_phase=phase)
        mcs_mask = data_merged_phase.cloudtracknumber_nomergesplit.sel(mcs_phase=phase).fillna(0)
        tb_mcs = tb.where(mcs_mask > 0)
        tmp = tb_mcs.values.ravel()
        tb_1d = tmp[~np.isnan(tmp)]
        cape_1d = cape_1d[~np.isnan(tmp)]
        subsat_1d = subsat_1d[~np.isnan(tmp)]

        for i in range(len(bins_cape)-1):
            idx = np.where(np.logical_and(cape_1d >= bins_cape[i], cape_1d < bins_cape[i+1]))[0]
            for j in range(len(bins_subsat)-1):
                idy = np.where(np.logical_and(subsat_1d >= bins_subsat[j], subsat_1d < bins_subsat[j+1]))[0]

                idx_com = np.intersect1d(idx,idy)
                bins_samples[0,p,i,j] += len(idx_com)
                tb_sum[0,p,i,j] += np.sum(tb_1d[idx_com])

        # ===== for outside mcs (environment) ======
        BL_CAPE_phase_env = data_BLenvmcs_phase.BL_CAPE_env.sel(mcs_phase=phase) # 9.81*(wb*(thetae_bl-thetae_sat_lt)/thetae_sat_lt)
        BL_CAPE_phase_env = (340*3/9.81)*BL_CAPE_phase_env # wb * (theta..)
        BL_SUBSAT_phase_env = data_BLenvmcs_phase.BL_SUBSAT_env.sel(mcs_phase=phase) # 9.81*(wl*(thetae_sat_lt-thetae_lt)/thetae_sat_lt)
        BL_SUBSAT_phase_env = (340*3/9.81)*BL_SUBSAT_phase_env # wl * (theta...)

        # get 1-D CAPE and SUBSAT values associated with the environment
        cape_1d = BL_CAPE_phase_env.values.ravel()
        subsat_1d = BL_SUBSAT_phase_env.values.ravel()
        tb = data_merged_phase.tb.sel(mcs_phase=phase)
        mcs_mask = data_merged_phase.cloudtracknumber_nomergesplit.sel(mcs_phase=phase).fillna(0)
        tb_mcs = tb.where(mcs_mask == 0)
        tmp = tb_mcs.values.ravel()
        tb_1d = tmp[~np.isnan(tmp)]
        cape_1d = cape_1d[~np.isnan(tmp)]
        subsat_1d = subsat_1d[~np.isnan(tmp)]

        for i in range(len(bins_cape)-1):
            idx = np.where(np.logical_and(cape_1d >= bins_cape[i], cape_1d < bins_cape[i+1]))[0]
            for j in range(len(bins_subsat)-1):
                idy = np.where(np.logical_and(subsat_1d >= bins_subsat[j], subsat_1d < bins_subsat[j+1]))[0]

                idx_com = np.intersect1d(idx,idy)
                bins_samples[1,p,i,j] += len(idx_com)
                tb_sum[1,p,i,j] += np.sum(tb_1d[idx_com])
                
        # ===== for 3-deg box averaged BL meatures, including MCS / non-MCS grids
        BL_CAPE_phase_mean = (data_BLenvmcs_phase.BL_CAPE_mcs.fillna(0) + data_BLenvmcs_phase.BL_CAPE_env.fillna(0)).sel(mcs_phase=phase, x=slice(14,26),
                                y=slice(14,26)).mean(('x','y'))
        BL_CAPE_phase_mean = (340*3/9.81)*BL_CAPE_phase_mean
        BL_SUBSAT_phase_mean = (data_BLenvmcs_phase.BL_SUBSAT_mcs.fillna(0) + data_BLenvmcs_phase.BL_SUBSAT_env.fillna(0)).sel(mcs_phase=phase, x=slice(14,26),
                                y=slice(14,26)).mean(('x','y'))
        BL_SUBSAT_phase_mean = (340*3/9.81)*BL_SUBSAT_phase_mean
    
        # get CAPE and SUBSAT 3-deg mean
        cape_1d = BL_CAPE_phase_mean.values.ravel()
        subsat_1d = BL_SUBSAT_phase_mean.values.ravel()
        tb = data_merged_phase.tb.sel(mcs_phase=phase)
        tb_mean = tb.sel(x=slice(14,26),y=slice(14,26)).mean(('x','y'))
        tmp = tb_mean.values.ravel()
        tb_1d = tmp[~np.isnan(tmp)]
        cape_1d = cape_1d[~np.isnan(tmp)]
        subsat_1d = subsat_1d[~np.isnan(tmp)]

        for i in range(len(bins_cape)-1):
            idx = np.where(np.logical_and(cape_1d >= bins_cape[i], cape_1d < bins_cape[i+1]))[0]
            for j in range(len(bins_subsat)-1):
                idy = np.where(np.logical_and(subsat_1d >= bins_subsat[j], subsat_1d < bins_subsat[j+1]))[0]

                idx_com = np.intersect1d(idx,idy)
                bins_samples[2,p,i,j] += len(idx_com)
                tb_sum[2,p,i,j] += np.sum(tb_1d[idx_com])

    ds_bins = xr.Dataset(data_vars = dict(samples = (['area_type','phase','bins_cape','bins_subsat'], bins_samples),
                                          tb_sum = (['area_type','phase','bins_cape','bins_subsat'], tb_sum)),
                         coords = dict(tracks = data_merged_phase.tracks.values,
                                       area_type = (['area_type'],['mcs','env','amean']),
                                       phase = (['phase'], ['CCS','Initial', 'Grow', 'Mature', 'Decay', 'End']),
                                       bins_cape = (['bins_cape'], bins_cape[:-1]),
                                       bins_subsat = (['bins_subsat'], bins_subsat[:-1])),
                         attrs = dict(description = 'cape-subsat histogram. amean = 3-deg average')
                         )
    
    return ds_bins

if __name__ == '__main__':

    # separate by mcs duration groups
    year = sys.argv[1]
    sampling_opt = sys.argv[2] # MCS geolocation: 'all','ocean','land'
    print('processing year: {}'.format(year))
    print('sampling_opt: {}'.format(sampling_opt))

    mcs_dir = Path('/scratch/wmtsai/temp_mcs/mcs_stats/mcs_tracks_non2mcs/')
    data_stats = xr.open_dataset(mcs_dir / 'mcs_tracks_non2mcs_{}.tropics30NS.full.nc'.format(year))

    buoy_dir = Path('/scratch/wmtsai/temp_mcs/output_stats/buoy_meregd_2001_2020/{}/environment_catalogs/VARS_derived'.format(year))
    mask_dir = Path('/scratch/wmtsai/temp_mcs/output_stats/buoy_meregd_2001_2020/{}/environment_catalogs/VARS_2D'.format(year))
    tb_dir = Path('/scratch/wmtsai/temp_mcs/output_stats/buoy_meregd_2001_2020/{}/environment_catalogs/VARS_2D'.format(year))
    data_buoy = xr.open_dataset(buoy_dir / 'MCS_FLEXTRKR_tropics_buoyancy.merged.nc')
    data_mask = xr.open_dataset(mask_dir / 'MCS_FLEXTRKR_tropics_cloudtracknumber_nomergesplit.merged.nc')
    data_tb = xr.open_dataset(tb_dir / 'MCS_FLEXTRKR_tropics_tb.merged.nc')
    data_merged_phase = xr.merge([data_buoy, data_mask, data_tb])  
    print('total number of tracks in {}: {}'.format(year, len(data_merged_phase.tracks)))

    data_bins_merged = []
    for (dmin, dmax, duration_type) in zip([5,6,12,18,24],[6,12,18,24,72],
                                           ['SL','ML','LL','UL','UUL']):

        # selecting files
        # 2. filtered by genesis location: 'all','ocean','land'
        tracks_landsea = data_filterby_landsea(data_stats, sampling_opt).tracks.values
        # 3. filtered by MCS duration
        tracks_duration = data_filterby_duration(data_stats, duration_min=dmin, duration_max=dmax).tracks.values
        tracks_combined = np.intersect1d(tracks_landsea, tracks_duration)
        data_merged_sub = data_merged_phase.sel(tracks=tracks_combined) # updating the selected tracks matching the conditions
        # rename time into mcs_phase
        data_merged_sub = data_merged_sub.rename({'time':'mcs_phase'})
        data_merged_sub['mcs_phase'] = ['CCS','Init','Grow','Mature','Decay','End']
     
        print('number of selected tracks: {}'.format(len(data_merged_sub.tracks)))

        data_bins_dtype = cape_subsat_hist(data_merged_sub)
        data_bins_merged.append(data_bins_dtype)

    data_bins_duration = xr.concat(data_bins_merged, pd.Index(['SL','ML','LL','UL','UUL'], name='duration_type'))

    out_dir = Path('/scratch/wmtsai/temp_mcs/output_stats/cape_subsat_hist')
    data_bins_duration.to_netcdf(out_dir / 'hist2d_cape_subsat_dtype.{}.{}.tb.full.nc'.format(year,sampling_opt))
    print('hist2d_cape_subsat_dtype.{}.{}.tb.full.nc'.format(year,sampling_opt))
