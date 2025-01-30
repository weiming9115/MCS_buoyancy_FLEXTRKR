#!/bin/bash

for year in {2003,2004,2005,2006}
do
#    echo 'processing mcs-env extraction: '$year
    python hist1d_BLtot_precip_nonMCSvsMCS_writeout.py $year 
    sleep 5
done
