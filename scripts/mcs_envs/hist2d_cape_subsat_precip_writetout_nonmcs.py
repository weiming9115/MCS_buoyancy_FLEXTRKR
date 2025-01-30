import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def get_files_landsea(year, sampling_opt='all'):
    
    """
    sampling option to filter out MCS tracks by genesis locations: 'all', 'ocean', 'land'
    """

    data = xr.open_dataset(nonmcs_dir / 'nonmcs_tracks_{}.tropics30NS.extend.nc'.format(year))

    idt_nonmcs_init = data.idt_nonmcs_init
    landsea_flag = data.landsea_flag.sel(times=idt_nonmcs_init)
    if sampling_opt == 'all':
        track_sel = data.tracks
    elif sampling_opt == 'land':
        idx_sel = np.where(landsea_flag == 1)[0]
        track_sel = data.isel(tracks=idx_sel).tracks
    elif sampling_opt == 'ocean':
        idx_sel = np.where(landsea_flag == 0)[0]
        track_sel = data.isel(tracks=idx_sel).tracks

    files_HCC = []
    for track in track_sel.values:
        files_HCC.extend(list(dir_envs_track.glob('nonmcs_era5_3D_envs_{}.{}.LD.nc'.format(year
                                                                    , str(track).zfill(7)))))
    return files_HCC

def BL_mcs_2dmap(fid_envs_track):
    """
    input: processed envs_track file containing MCS feature mask and 2D/3D variables on ERA-5 coordinates
    return BL componets within the tracked MCS feature
    """
    
    data = xr.open_dataset(fid_envs_track)
    thetae_bl = data.thetae_bl
    thetae_lt = data.thetae_lt
    thetae_sat_lt = data.thetae_sat_lt
    sp = data.SP/100 # hPa

    delta_pl=sp-100-400
    delta_pb=100
    wb=(delta_pb/delta_pl)*np.log((delta_pl+delta_pb)/delta_pb)
    wl=1-wb

    BL_CAPE = wb*(thetae_bl - thetae_sat_lt)/thetae_sat_lt*340 # (K)
    BL_SUBSAT = wl*(thetae_sat_lt - thetae_lt)/thetae_sat_lt*340 # (K)
    BL_TOT = BL_CAPE - BL_SUBSAT  # (K)

    # BL associated with mcs / non-mcs grids
    # use Tb < 241 K to define CCS 
    tb = data.tb 
    BL_TOT_mcs = BL_TOT.where(tb < 241)
    BL_CAPE_mcs = BL_CAPE.where(tb < 241)
    BL_SUBSAT_mcs = BL_SUBSAT.where(tb < 241)
    
    BL_TOT_env = BL_TOT.where(tb >= 241)
    BL_CAPE_env = BL_CAPE.where(tb >= 241)
    BL_SUBSAT_env = BL_SUBSAT.where(tb >= 241)
    
    return BL_TOT_mcs, BL_CAPE_mcs, BL_SUBSAT_mcs, BL_TOT_env, BL_CAPE_env, BL_SUBSAT_env

def prec_mcs_2dmap(fid_envs_track):
    """
    input: processed envs_track file containing MCS feature mask and 2D/3D variables on ERA-5 coordinates
    return precipitation componets within the tracked MCS feature
    """

    data = xr.open_dataset(fid_envs_track)
    prec_gpm = data.precipitationCal # regridded gpm-precipitation (mm/hr)
    prec_era5 = data.mtpr # era-5 precipitation (mm/hr)

    # GPMprec associated with mcs / non-mcs grids
    # use Tb < 241 K to define CCS 
    tb = data.tb # binary mask
    prec_gpm_mcs = prec_gpm.where(tb < 241)
    prec_gpm_env = prec_gpm.where(tb >= 241)
    prec_era5_mcs = prec_era5.where(tb < 241)
    prec_era5_env = prec_era5.where(tb >= 241)

    return prec_gpm_mcs, prec_gpm_env, prec_era5_mcs, prec_era5_env

def cape_subsat_hist(files, multi_year=False):
    
    # bins for BL_CAPE and BL_SUBSAT
    bins_cape = np.arange(-15,10,0.25)
    bins_subsat = np.arange(-5,25,0.25)
    bins_samples = np.zeros((4, 5, len(bins_cape)-1, len(bins_subsat)-1)) # (area_type, mcs_phase, cape, subsat)
    prec_gpm_sum = np.copy(bins_samples)
    prec_era5_sum = np.copy(bins_samples)
    
    n = 0
    track_list = []
    for file in files: # total files read (single year or multiple years)
        
        if multi_year == True:
            n += 1 # total track numbers merged
            
        track_list.append(int(file.name[-13:-6])) # save track number 
        
        (BL_TOT_mcs, BL_CAPE_mcs, BL_SUBSAT_mcs, BL_TOT_env, BL_CAPE_env, BL_SUBSAT_env) = BL_mcs_2dmap(file)
        (prec_gpm_mcs, prec_gpm_env, prec_era5_mcs, prec_era5_env) = prec_mcs_2dmap(file)

        for p, phase in enumerate(["Init", "Grow", "Mature", "Decay", "End"]):

            # ===== for inside mcs ======
            BL_CAPE_phase_mcs = BL_CAPE_mcs.sel(mcs_phase=phase) # ((thetae_bl-thetae_sat_lt)/thetae_sat_lt*340)
            BL_SUBSAT_phase_mcs = BL_SUBSAT_mcs.sel(mcs_phase=phase) # ((thetae_sat_lt-thetae_lt)/thetae_sat_lt*340)
            prec_gpm_phase_mcs = prec_gpm_mcs.sel(mcs_phase=phase)
            prec_era5_phase_mcs = prec_era5_mcs.sel(mcs_phase=phase)
             
            # get 1-D CAPE and SUBSAT values associated with MCS
            cape_1d = BL_CAPE_phase_mcs.values.ravel()
            subsat_1d = BL_SUBSAT_phase_mcs.values.ravel()
            prec_gpm_1d = prec_gpm_phase_mcs.values.ravel()
            prec_era5_1d = prec_era5_phase_mcs.values.ravel()

            for i in range(len(bins_cape)-1):
                idx = np.where(np.logical_and(cape_1d >= bins_cape[i], cape_1d < bins_cape[i+1]))[0]
                for j in range(len(bins_subsat)-1):
                    idy = np.where(np.logical_and(subsat_1d >= bins_subsat[j], subsat_1d < bins_subsat[j+1]))[0]

                    idx_com = np.intersect1d(idx,idy)
                    bins_samples[0,p,i,j] += len(idx_com)
                    prec_gpm_sum[0,p,i,j] += np.sum(prec_gpm_1d[idx_com])
                    prec_era5_sum[0,p,i,j] += np.sum(prec_era5_1d[idx_com])

            # ===== for outside mcs (environment) ======
            BL_CAPE_phase_env = BL_CAPE_env.sel(mcs_phase=phase) # (thetae_bl-thetae_sat_lt)/thetae_sat_lt*340
            BL_SUBSAT_phase_env = BL_SUBSAT_env.sel(mcs_phase=phase) # (thetae_sat_lt-thetae_lt)/thetae_sat_lt*340
            prec_gpm_phase_env = prec_gpm_env.sel(mcs_phase=phase)
            prec_era5_phase_env = prec_era5_env.sel(mcs_phase=phase)

            # get 1-D CAPE and SUBSAT values associated with the environment
            cape_1d = BL_CAPE_phase_env.values.ravel()
            subsat_1d = BL_SUBSAT_phase_env.values.ravel()
            prec_gpm_1d = prec_gpm_phase_env.values.ravel()
            prec_era5_1d = prec_era5_phase_env.values.ravel()

            for i in range(len(bins_cape)-1):
                idx = np.where(np.logical_and(cape_1d >= bins_cape[i], cape_1d < bins_cape[i+1]))[0]
                for j in range(len(bins_subsat)-1):
                    idy = np.where(np.logical_and(subsat_1d >= bins_subsat[j], subsat_1d < bins_subsat[j+1]))[0]

                    idx_com = np.intersect1d(idx,idy)
                    bins_samples[1,p,i,j] += len(idx_com)
                    prec_gpm_sum[1,p,i,j] += np.sum(prec_gpm_1d[idx_com])
                    prec_era5_sum[1,p,i,j] += np.sum(prec_era5_1d[idx_com])
                    
            # ===== for 3-deg box averaged BL meatures, including MCS / non-MCS grids
            BL_CAPE_phase_mean = (BL_CAPE_mcs.fillna(0) + BL_CAPE_env.fillna(0)).sel(mcs_phase=phase, x=slice(14,26),
                                  y=slice(14,26)).mean(('x','y'))
            BL_SUBSAT_phase_mean = (BL_SUBSAT_mcs.fillna(0) + BL_SUBSAT_env.fillna(0)).sel(mcs_phase=phase, x=slice(14,26),
                                  y=slice(14,26)).mean(('x','y'))
            prec_gpm_phase_mean = (prec_gpm_mcs.fillna(0) + prec_gpm_env.fillna(0)).sel(mcs_phase=phase, x=slice(14,26),
                                  y=slice(14,26)).mean(('x','y'))
            prec_era5_phase_mean = (prec_era5_mcs.fillna(0) + prec_era5_env.fillna(0)).sel(mcs_phase=phase, x=slice(14,26),
                                  y=slice(14,26)).mean(('x','y'))
        
            # get  CAPE and SUBSAT 3-deg mean
            cape_1d = BL_CAPE_phase_mean.values.ravel()
            subsat_1d = BL_SUBSAT_phase_mean.values.ravel()
            prec_gpm_1d = prec_gpm_phase_mean.values.ravel()
            prec_era5_1d = prec_era5_phase_mean.values.ravel()

            for i in range(len(bins_cape)-1):
                idx = np.where(np.logical_and(cape_1d >= bins_cape[i], cape_1d < bins_cape[i+1]))[0]
                for j in range(len(bins_subsat)-1):
                    idy = np.where(np.logical_and(subsat_1d >= bins_subsat[j], subsat_1d < bins_subsat[j+1]))[0]

                    idx_com = np.intersect1d(idx,idy)
                    bins_samples[2,p,i,j] += len(idx_com)
                    prec_gpm_sum[2,p,i,j] += np.sum(prec_gpm_1d[idx_com])
                    prec_era5_sum[2,p,i,j] += np.sum(prec_era5_1d[idx_com])

            # ===== for the mean BL meature of MCS grids
            BL_CAPE_phase_mean = BL_CAPE_mcs.sel(mcs_phase=phase).mean(('x','y'))
            BL_SUBSAT_phase_mean = BL_SUBSAT_mcs.sel(mcs_phase=phase).mean(('x','y'))
            prec_gpm_phase_mean = prec_gpm_mcs.sel(mcs_phase=phase).mean(('x','y'))
            prec_era5_phase_mean = prec_era5_mcs.sel(mcs_phase=phase).mean(('x','y'))

            # get  CAPE and SUBSAT 3-deg mean
            cape_1d = BL_CAPE_phase_mean.values.ravel()
            subsat_1d = BL_SUBSAT_phase_mean.values.ravel()
            prec_gpm_1d = prec_gpm_phase_mean.values.ravel()
            prec_era5_1d = prec_era5_phase_mean.values.ravel()

            for i in range(len(bins_cape)-1):
                idx = np.where(np.logical_and(cape_1d >= bins_cape[i], cape_1d < bins_cape[i+1]))[0]
                for j in range(len(bins_subsat)-1):
                    idy = np.where(np.logical_and(subsat_1d >= bins_subsat[j], subsat_1d < bins_subsat[j+1]))[0]

                    idx_com = np.intersect1d(idx,idy)
                    bins_samples[3,p,i,j] += len(idx_com)
                    prec_gpm_sum[3,p,i,j] += np.sum(prec_gpm_1d[idx_com])
                    prec_era5_sum[3,p,i,j] += np.sum(prec_era5_1d[idx_com])

    if multi_year == False:
    
        ds_bins = xr.Dataset(data_vars = dict(samples = (['area_type','phase','bins_cape','bins_subsat'], bins_samples),
                                              prec_gpm_sum = (['area_type','phase','bins_cape','bins_subsat'], prec_gpm_sum),
                                              prec_era5_sum = (['area_type','phase','bins_cape','bins_subsat'], prec_era5_sum)),
                 coords = dict(tracks = track_list,
                               area_type = (['area_type'],['mcs','env','amean','mcs_mean']),
                               phase = (['phase'], ['Initial', 'Grow', 'Mature', 'Decay', 'End']),
                               bins_cape = (['bins_cape'], bins_cape[:-1]),
                               bins_subsat = (['bins_subsat'], bins_subsat[:-1])),
                 attrs = dict(description = 'cape-subsat histogram. amean = 3-deg average'))
    else:
        ds_bins = xr.Dataset(data_vars = dict(samples = (['area_type','phase','bins_cape','bins_subsat'], bins_samples),
                                              prec_gpm_sum = (['area_type','phase','bins_cape','bins_subsat'], prec_gpm_sum),
                                              prec_era5_sum = (['area_type','phase','bins_cape','bins_subsat'], prec_era5_sum)),
                 coords = dict(tracks = np.arange(n),
                               area_type = (['area_type'],['mcs','env','amean','mcs_mean']),
                               phase = (['phase'], ['Initial', 'Grow', 'Mature', 'Decay', 'End']),
                               bins_cape = (['bins_cape'], bins_cape[:-1]),
                               bins_subsat = (['bins_subsat'], bins_subsat[:-1])),
                 attrs = dict(description = 'cape-subsat histogram. amean = 3-deg average'))
    
    return ds_bins

if __name__ == '__main__':

    # separate by mcs duration groups
    year = sys.argv[1]

    ########  parameters for filtering MCS tracks  ########
    sampling_opt = 'all' # non-MCS geolocation: 'all','ocean','land'
    ######################################################33

    print('processing year: {}'.format(year))
    print('sampling_opt: {}'.format(sampling_opt))

    nonmcs_dir = Path('/scratch/wmtsai/temp_mcs/mcs_stats/nonmcs_tracks_samples/')
    dir_envs_track = Path('/scratch/wmtsai/temp_mcs/mcs_stats/envs_track/{}/nonmcs_extend'.format(year))

    data_bins_merged = []

    # selecting files 
    files_geoloc = get_files_landsea(year, sampling_opt)
    print('number of selected tracks: {}'.format(len(files_geoloc)))
    data_bins = cape_subsat_hist(files_geoloc, multi_year=False)

    out_dir = Path('/scratch/wmtsai/temp_mcs/output_stats/cape_subsat_hist')
    data_bins.to_netcdf(out_dir / 'hist2d_cape_subsat_dtype.{}.{}.FNL.nonmcs.nc'.format(year,sampling_opt))
    print('hist2d_cape_subsat_dtype.{}.{}.FNL.nonmcs.nc'.format(year,sampling_opt))
