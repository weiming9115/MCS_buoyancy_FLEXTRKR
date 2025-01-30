#!/bin/bash

area_type='land'

python hist2d_cape_subsat_writetout.MERGED.py 2001 $area_type &
python hist2d_cape_subsat_writetout.MERGED.py 2002 $area_type &
python hist2d_cape_subsat_writetout.MERGED.py 2003 $area_type &
python hist2d_cape_subsat_writetout.MERGED.py 2004 $area_type &
python hist2d_cape_subsat_writetout.MERGED.py 2005 $area_type &
python hist2d_cape_subsat_writetout.MERGED.py 2006 $area_type &
python hist2d_cape_subsat_writetout.MERGED.py 2007 $area_type &
python hist2d_cape_subsat_writetout.MERGED.py 2008 $area_type &
python hist2d_cape_subsat_writetout.MERGED.py 2009 $area_type &
python hist2d_cape_subsat_writetout.MERGED.py 2010 $area_type &

