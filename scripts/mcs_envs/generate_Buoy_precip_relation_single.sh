#!/bin/bash

for year in {2003,2004,2005,2006,2007,2008,2009}
    do 
    python Buoy_precip_relation_regional_gridtype.py $year
    done 
