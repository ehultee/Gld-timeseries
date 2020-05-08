## Time series decomposition on Helheim velocity
## 6 May 2020  EHU
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import iceutils as ice


## Set up combined hdf5 stack
fpath='/Users/lizz/Documents/Research/Gld-timeseries/Stack/'
hel_stack = ice.MagStack(files=[fpath+'vx.h5', fpath+'vy.h5'])
data_key = 'igram' # B. Riel convention for access to datasets in hdf5 stack

## Extract time series at selected points
xy_1 = (308103., -2577200.) # polar stereo coordinates of a point near Helheim 2009 terminus, in m
xy_2 = (302026., -2566770.) # point up on North branch
xy_3 = (297341., -2571490.) # point upstream on main branch
xy_4 = (294809., -2577580.) # point on southern tributary
xys = (xy_1, xy_3, xy_4, xy_2) #order by mean velocity magnitude (for nice stacked plotting)
labels = ('Near terminus', 'Main branch', 'South branch', 'North branch')
series = [hel_stack.timeseries(xy=xyi, key=data_key) for xyi in xys]

## Set up design matrix and perform lasso regression, according to Bryan's documentation
def build_collection(dates):
    """
    Function that creates a list of basis functions for a given datetime vector dates.
    """
    # Get date bounds
    tstart, tend = dates[0], dates[-1]

    # Initalize a collection and relevant basis functions
    collection = ice.tseries.timefn.TimefnCollection()
    periodic = ice.tseries.timefn.fnmap['periodic']
    ispl = ice.tseries.timefn.fnmap['isplineset']
    poly = ice.tseries.timefn.fnmap['poly']

    # Add polynomial first for secular components
    collection.append(poly(tref=tstart, order=1, units='years'))

    # Add seasonal terms
    collection.append(periodic(tref=tstart, units='years', period=0.5,
                               tmin=tstart, tmax=tend))
    collection.append(periodic(tref=tstart, units='years', period=1.0,
                               tmin=tstart, tmax=tend))
    
    # Integrated B-slines for transient signals
    # In general, we don't know the timescales of transients a prior
    # Therefore, we add integrated B-splines of various timescales where the
    # timescale is controlled by the 'nspl' value (this means to divide the time
    # vector into 'nspl' equally spaced spline center locations)
    for nspl in [128, 64, 32, 16, 8, 4]:
        collection.append(ispl(order=3, num=nspl, units='years', tmin=tstart, tmax=tend))
    
    # Done
    return collection

# First convert the time vector to a list of datetime
dates = ice.tdec2datestr(hel_stack.tdec, returndate=True)

# Build the collection
collection = build_collection(dates)

# Instantiate a model
model = ice.tseries.Model(dates, collection=collection)

# Create lasso regression solvers (see time_series_inversion.ipynb for ridge vs lasso performance)
lasso = ice.tseries.select_solver('lasso', reg_indices=model.itransient, penalty=0.2)

## Perform inversion for each time series extracted
decomps = []
for ser in series:    
    SUCCESS, m_lasso, Cm = lasso.invert(model.G, ser)
    pred = model.predict(m_lasso)
    decomps.append(pred)


## Plot output in nice figures
colors = cm.get_cmap('Dark2')(np.linspace(0, 1, num=len(labels)))

# ## Plot single continuous series
# fig, ax = plt.subplots(figsize=(12,6))
# ax.plot(hel_stack.tdec, series[0], '.', color=colors[0], markersize=10, alpha=0.5)
# ax.plot(hel_stack.tdec, pred['full'], color=colors[0])
# ax.set_xlabel('Year')
# ax.set_ylabel('Velocity')
# plt.show()

# ## Plot secular, transient, and seasonal signals
# fig1, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(12,6), sharex=True)
# ax1.plot(hel_stack.tdec, pred['secular'], color=colors[0])
# ax1.set_title('Secular')
# ax2.plot(hel_stack.tdec, pred['seasonal'], color=colors[0])
# ax2.set_title('Seasonal')
# ax3.plot(hel_stack.tdec, pred['transient'], color=colors[0])
# ax3.set_title('Transient')
# plt.show()

## Plot fits of all extracted series
fig2, ax4 = plt.subplots(figsize=(12,6))
for i in range(len(series)):
    ax4.plot(hel_stack.tdec, series[i], '.', color=colors[i], markersize=10, alpha=0.5, label=labels[i])
    ax4.plot(hel_stack.tdec, decomps[i]['full'], color=colors[i])
ax4.legend(loc='center right')
ax4.set(xlabel='Year', ylim=(0, 11), yticks=[0, 2, 4, 6, 8, 10], ylabel='Velocity [km/a]')
plt.show()

## Plot stacked secular signals
fig3, ax5 = plt.subplots(figsize=(12,6))
for i in range(len(series)):
    ax5.plot(hel_stack.tdec, decomps[i]['secular'], color=colors[i], label='{} secular'.format(labels[i]))
ax5.legend(loc='best')
ax5.set(xlabel='Year', ylim=(0, 11), yticks=[0, 2, 4, 6, 8, 10], ylabel='Velocity [km/a]')
plt.show()

## Stacked seasonal signals (check they correspond)
# fig4, ax6 = plt.subplots(figsize=(12,6))
# for i in range(len(series)):
#     ax6.plot(hel_stack.tdec, decomps[i]['seasonal'], color=colors[i], label='{} seasonal'.format(labels[i]))
# ax6.legend(loc='best')
# # ax6.set(xlabel='Year', ylim=(0, 11), yticks=[0, 2, 4, 6, 8, 10], ylabel='Velocity [km/a]')
# plt.show()

## Plot stacked transient signals
fig5, ax7 = plt.subplots(figsize=(12,6))
for i in range(len(series)):
    ax7.plot(hel_stack.tdec, decomps[i]['transient'], color=colors[i], label='{} transient'.format(labels[i]))
ax7.legend(loc='best')
ax7.set(xlabel='Year', ylim=(-2.25, 1.25), yticks=[-2, -1, 0, 1], ylabel='Speed [km/a]')
plt.show()

## Plot secular and transient for EGU
fig6, (ax8, ax9) = plt.subplots(nrows=2, figsize=(12,6), sharex=True)
for i in range(len(series)):
    ax8.plot(hel_stack.tdec, decomps[i]['secular'], color=colors[i])
    ax9.plot(hel_stack.tdec, decomps[i]['transient'], color=colors[i], label=labels[i])
ax9.legend(loc='best')
ax8.set(ylim=(0, 11), yticks=[0, 2, 4, 6, 8, 10], ylabel='Speed [km/a]', title='Secular signal')
ax9.set(xlabel='Year', ylim=(-2.25, 1.25), yticks=[-2, -1, 0, 1], ylabel='Speed [km/a]', title='Transient signal')
plt.show()