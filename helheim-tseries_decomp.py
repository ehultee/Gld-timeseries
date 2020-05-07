## Time series decomposition on Helheim velocity
## 6 May 2020  EHU
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
series_m = [np.ma.masked_invalid(ser) for ser in series]
masks = [np.ma.getmask(ser) for ser in series_m]


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

## Access the design matrix for plotting
G = model.G
# plt.plot(hel_stack.tdec, G)
# plt.xlabel('Year')
# plt.ylabel('Amplitude')
# plt.show()

# Create a ridge regression solver that damps out the transient spline coefficients
ridge = ice.tseries.select_solver('ridge', reg_indices=model.itransient, penalty=20)
# Create lasso regression object
lasso = ice.tseries.select_solver('lasso', reg_indices=model.itransient, penalty=2)

## Perform inversion to get coefficient vector and coefficient covariance matrix
# SUCCESS, m, Cm = ridge.invert(model.G, series[0]) # fit near-terminus (series[0]) first
SUCCESS, m_lasso, Cm = lasso.invert(model.G, series[0])
print(SUCCESS)

## Model will perform predictions
pred = model.predict(m_lasso)

print(len(pred['full']), len(series[0]))
print(sum(np.isnan(pred['full'])), sum(np.isnan(series[0])))
print(np.nanmean(pred['full']), np.nanmean(series[0]))

## Plotting
# fig, ax = plt.subplots(figsize=(12,6))
# ax.plot(hel_stack.tdec, series[0], '.')
# ax.plot(hel_stack.tdec, pred['full'])
# ax.set_xlabel('Year')
# ax.set_ylabel('Velocity')
# plt.show()