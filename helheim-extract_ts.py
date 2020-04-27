## Extract and plot Helheim time series with IceUtils
## 19 Mar 2020  EHU
import numpy as np
import matplotlib.pyplot as plt
import iceutils as ice
import sys

## Set up combined hdf5 stack
fpath='/Users/lizz/Documents/Research/Gld-timeseries/Stack/'
hel_stack = ice.MagStack(files=[fpath+'vx.h5', fpath+'vy.h5'])
data_key = 'igram' # B. Riel convention for access to datasets in hdf5 stack

## Extract time series at a point
xy_1 = (308000., -2580000.) #polar stereo coordinates of a point near Helheim 2009 terminus, in m
d = hel_stack.timeseries(xy=xy_1, key=data_key)
print(type(d))

plt.figure()
plt.plot(hel_stack.tdec, d, marker='o', lw=0.5)
plt.axes().set_xlabel('Year')
plt.axes().set_ylabel('Speed [m/a]')
plt.title('Helheim Glacier speed at {}'.format(0.001*np.array(xy_1)))
plt.show()