#!/bin/bash

for year in {2011,2012,2013,2014,2015,2016,2017,2018,2019,2020}
do
    echo 'processing mcs-env extraction: '$year
    python mcs_3Denvs_extract_yearly_merge.py $year new
    sleep 5
done
