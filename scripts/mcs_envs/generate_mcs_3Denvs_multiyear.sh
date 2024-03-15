#!/bin/bash

for year in {2010,2011,2012,2013}
do
    echo 'processing mcs-env extraction: '$year
    python mcs_3Denvs_extract_writeout_buoyCal_parallel.py $year new
    sleep 5
done
