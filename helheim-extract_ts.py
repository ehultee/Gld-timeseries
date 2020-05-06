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

## Extract time series at selected points
xy_1 = (308103., -2577200.) #polar stereo coordinates of a point near Helheim 2009 terminus, in m
xy_2 = (302026., -2566770.) # point up on North branch
xy_3 = (297341., -2571490.) # point upstream on main branch
xy_4 = (294809., -2577580.) # point on southern tributary
xys = (xy_1, xy_2, xy_3, xy_4)
labels = ('Near terminus', 'North branch', 'Main branch', 'South branch')

series = [hel_stack.timeseries(xy=xyi, key=data_key) for xyi in xys]

s = hel_stack.slice(index=10, key=data_key)

fig1, ax1 = plt.subplots()
for i, ser in enumerate(series):
	ax1.plot(hel_stack.tdec, ser, marker='o', lw=0.5, label=labels[i])
ax1.set_ylabel('Speed [km/a]')
ax1.legend(loc='best')
plt.show()


fig2, ax2 = plt.subplots()
ax2.contourf(hel_stack.stacks[0]._datasets['x'], hel_stack.stacks[0]._datasets['y'], s)
for xy in (xy_1, xy_2, xy_3, xy_4):
	ax2.plot(xy[0], xy[1], marker='*')
plt.show()