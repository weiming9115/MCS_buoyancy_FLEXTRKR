#!/bin/bash

area_type='ocean'

python hist2d_cape_subsat_tb_writetout.MERGED.py 2011 $area_type &
python hist2d_cape_subsat_tb_writetout.MERGED.py 2012 $area_type &
python hist2d_cape_subsat_tb_writetout.MERGED.py 2013 $area_type &
python hist2d_cape_subsat_tb_writetout.MERGED.py 2014 $area_type &
python hist2d_cape_subsat_tb_writetout.MERGED.py 2015 $area_type &
python hist2d_cape_subsat_tb_writetout.MERGED.py 2016 $area_type &
python hist2d_cape_subsat_tb_writetout.MERGED.py 2017 $area_type &
python hist2d_cape_subsat_tb_writetout.MERGED.py 2018 $area_type &
python hist2d_cape_subsat_tb_writetout.MERGED.py 2019 $area_type &
python hist2d_cape_subsat_tb_writetout.MERGED.py 2020 $area_type &
