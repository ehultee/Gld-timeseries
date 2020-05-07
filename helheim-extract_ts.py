## Extract and plot Helheim time series with IceUtils
## 19 Mar 2020  EHU
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
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

## Make some plots to show at EGU
colors = cm.get_cmap('Dark2')(np.linspace(0, 1, num=len(labels)))
fig1, ax1 = plt.subplots()
for i, ser in enumerate(series):
	ax1.plot(hel_stack.tdec, ser, marker='o', lw=0.5, label=labels[i], color=colors[i])
ax1.set(xlabel='Year', ylabel='Speed [km/a]')
ax1.legend(loc='best')
plt.show()


fig2, ax2 = plt.subplots()
ax2.contourf(hel_stack.stacks[0]._datasets['x'], hel_stack.stacks[0]._datasets['y'], s, 50)
for i, xy in enumerate(xys):
	ax2.plot(xy[0], xy[1], marker='*', markersize=10, color=colors[i])
ax2.set(
	xlim=(292000, 320500), ylim=(-2581400, -2555000),
	xticks=[295000, 300000, 305000, 310000, 315000, 320000], xticklabels=['295', '300', '305', '310', '315', '320'], xlabel='Easting [km]',
	yticks=[-2580000, -2575000, -2570000, -2565000, -2560000, -2555000], yticklabels=['-2580', '-2575', '-2570', '-2565', '-2560', '-2555'], ylabel='Northing [km]'
	)
ax2.set_aspect(1)
plt.tight_layout()
plt.show()