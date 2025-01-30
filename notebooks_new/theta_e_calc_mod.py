# Module for theta_e calculation

import xarray as xr
import numpy as np

def find_pbl_top_level(sp, temp, pbl_depth=100):
    
    if len(sp.squeeze().dims) == 0: # 1-dimension, grid data
    
        p_level = temp.level # get era-5 standard pressure levels
        idx = np.argmin(abs(level.values- (sp.values - pbl_depth)))
        
        pbl_top_level_xr = p_level[idx]
                
    elif len(sp.squeeze().dims) == 2: # 2-dimension (lat, lon)
        
        p_level_2d = (temp - temp + temp.level) 
        idx = np.argmin(abs(p_level_2d.values- (sp.values - pbl_depth)), axis=0)
        
        pbl_list = []
        for n in idx.ravel():
            pbl_list.append(p_level_2d.level[n])
        pbl_list = np.asarray(pbl_list)
        
        pbl_top_level = pbl_list.reshape((len(sp.y), len(sp.x)))
        #convert back into xarray
        pbl_top_level_xr = xr.Dataset(data_vars=dict(pbl_top_level=(['y','x'],pbl_top_level)),
                                      coords=dict(latitude=(['y'], sp.y.values),
                                                  longitude=(['x'], sp.x.values)))
            
    return pbl_top_level_xr

def find_pbl_top_level_latlon(sp, temp, pbl_depth=100):

    if len(sp.squeeze().dims) == 0: # 1-dimension, grid data

        p_level = temp.level # get era-5 standard pressure levels
        idx = np.argmin(abs(level.values- (sp.values - pbl_depth)))

        pbl_top_level_xr = p_level[idx]

    elif len(sp.squeeze().dims) == 2: # 2-dimension (lat, lon)

        p_level_2d = (temp - temp + temp.level)
        idx = np.argmin(abs(p_level_2d.values- (sp.values - pbl_depth)), axis=0)

        pbl_list = []
        for n in idx.ravel():
            pbl_list.append(p_level_2d.level[n])
        pbl_list = np.asarray(pbl_list)

        pbl_top_level = pbl_list.reshape((len(sp.latitude), len(sp.longitude)))
        #convert back into xarray
        pbl_top_level_xr = xr.Dataset(data_vars=dict(pbl_top_level=(['latitude','longitude'],pbl_top_level)),
                                      coords=dict(latitude=(['latitude'], sp.latitude.values),
                                                  longitude=(['longitude'], sp.longitude.values)))

    return pbl_top_level_xr

def es_calc_bolton(temp):
    # in hPa

    tmelt  = 273.15
    tempc = temp - tmelt 
    es = 6.112*np.exp(17.67*tempc/(243.5+tempc))
    
    return es

def es_calc(temp):

    tmelt  = 273.15
    
    c0=0.6105851e+03
    c1=0.4440316e+02
    c2=0.1430341e+01
    c3=0.2641412e-01
    c4=0.2995057e-03
    c5=0.2031998e-05
    c6=0.6936113e-08
    c7=0.2564861e-11
    c8=-.3704404e-13

    tempc = temp - tmelt 
    tempcorig = tempc
        
    #if tempc < -80:
    es_ltn80c = es_calc_bolton(temp)
    es_ltn80c = es_ltn80c.where(tempc < -80, 0)
    #else:
    es = c0+tempc*(c1+tempc*(c2+tempc*(c3+tempc*(c4+tempc*(c5+tempc*(c6+tempc*(c7+tempc*c8)))))))
    es_gtn80c = es/100
    es_gtn80c = es_gtn80c.where(tempc >= -80, 0)
    
    # complete es
    es = es_ltn80c + es_gtn80c
    
    return es

def qs_calc(temp):

    tmelt  = 273.15
    RV=461.5
    RD=287.04

    EPS=RD/RV

    press = temp.level * 100. # in Pa
    tempc = temp - tmelt 

    es=es_calc(temp) 
    es=es * 100. #hPa
    qs = (EPS * es) / (press + ((EPS-1.)*es))
    
    return qs

def theta_e_calc(temp, q):
    
    # following the definitions in Bolton (1980): the calculation of equivalent potential temperature
    
    pref = 100000.
    tmelt  = 273.15
    CPD=1005.7
    CPV=1870.0
    CPVMCL=2320.0
    RV=461.5
    RD=287.04
    EPS=RD/RV
    ALV0=2.501E6

    press = temp.level * 100. # in Pa
    tempc = temp - tmelt # in C

    r = q / (1. - q) 

    # get ev in hPa 
    ev_hPa = temp.level * r / (EPS + r) # hpa

    #get TL
    TL = (2840. / ((3.5*np.log(temp)) - (np.log(ev_hPa)) - 4.805)) + 55.

    #calc chi_e:
    chi_e = 0.2854 * (1. - (0.28 * r))

    theta_e = temp * np.power((pref / press),chi_e) * np.exp(((3.376/TL) - 0.00254) * r * 1000 * (1. + (0.81 * r)))
    
    return theta_e


