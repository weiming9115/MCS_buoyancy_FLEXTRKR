import os
import sys
from pathlib import Path
import pandas as pd
import xarray as xr
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":

    # available MCS data 2001-2020
    year_list = range(2001,2021)
    
    # open an text file to which missing times will be written into
    out_dir = Path('/neelin2020/mcs_flextrkr')
    with open(out_dir / 'MCSflextrkr_unavailable_times.txt', 'w') as f:

        f.write('FLEXTRKR MCS unavailable data files\n')

        for year_sel in year_list:
            print('writing year: {}'.format(year_sel))
            f.write('---------------------------------\n')
            f.write('year: {}\n'.format(year_sel))
            f.write('---------------------------------\n')
            mcs_dir = Path('/neelin2020/mcs_flextrkr/{}'.format(year_sel))
            # writeout a list of unavailable FLEXTRKR MCS times 
            timestamps = pd.date_range(start='1/1/{}'.format(year_sel), end='12/31/{}'.format(year_sel), freq='1H')
            for timestamp in timestamps:
                
                datetime_str = str(timestamp.to_datetime64())
                year = datetime_str[:4]
                month = datetime_str[5:7]
                day = datetime_str[8:10]
                hour = datetime_str[11:13]

                file = mcs_dir / 'mcstrack_{}{}{}_{}30.nc'.format(year,month,day,hour)
                if file.exists() == False:
                    f.write('mcstrack_{}{}{}_{}30.nc\n'.format(year,month,day,hour))
                
        f.close()

        


    



