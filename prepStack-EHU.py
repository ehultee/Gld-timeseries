#!/usr/bin/env python3

import numpy as np
# import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import map_coordinates
from sklearn.linear_model import RANSACRegressor
import datetime
from tqdm import tqdm
import glob
import gdal
import h5py
import sys
import os

def main():

    # Traverse data path
    dpath = '/home/ehultee/data/nsidc0481_MEASURES_greenland_V01/Ecoast-66.50N'
    dates = []
    vx_files = []; vy_files = []
    ex_files = []; ey_files = []
    for root, dirs, files in os.walk(dpath):
        for fname in files:

            if not fname.endswith('v1.2.meta'):
                continue

            # Get the dates for the igram pair
            first_date, second_date, nominal_dt = parseMeta(os.path.join(root, fname))

            # Compute middle date
            dt = 0.5 * (second_date - first_date).total_seconds()
            mid_date = first_date + datetime.timedelta(seconds=dt)
            mid_date += datetime.timedelta(seconds=nominal_dt)
            dates.append(mid_date)

            # Find the data files
            vx_file = glob.glob(os.path.join(root, '*vx*.tif'))[0]
            vy_file = glob.glob(os.path.join(root, '*vy*.tif'))[0]
            ex_file = glob.glob(os.path.join(root, '*ex*.tif'))[0]
            ey_file = glob.glob(os.path.join(root, '*ey*.tif'))[0]

            # Append the filenames
            vx_files.append(vx_file)
            vy_files.append(vy_file)
            ex_files.append(ex_file)
            ey_files.append(ey_file)

    # Create array for dates and files
    dates = np.array(dates)
    N_dates = len(dates)
    vx_files = np.array(vx_files, dtype='S')
    vy_files = np.array(vy_files, dtype='S')
    ex_files = np.array(ex_files, dtype='S')
    ey_files = np.array(ey_files, dtype='S')

    # Construct array of decimal year
    tdec = np.zeros(N_dates)
    for i in tqdm(range(N_dates)):
        date = dates[i]
        year_start = datetime.datetime(date.year, 1, 1)
        if date.year % 4 == 0:
            ndays = 366.0
        else:
            ndays = 365.0
        tdec[i] = date.year + (date - year_start).total_seconds() / (ndays * 86400.0)

    # Sort the dates and files
    indsort = np.argsort(tdec)
    tdec = tdec[indsort]
    vx_files = vx_files[indsort]
    vy_files = vy_files[indsort]
    ex_files = ex_files[indsort]
    ey_files = ey_files[indsort]

    # Read first file to get dimensions and geo transform
    ds = gdal.Open(vx_files[0], gdal.GA_ReadOnly)
    Ny, Nx = ds.RasterYSize, ds.RasterXSize
    x_start, dx, _, y_start, _, dy = ds.GetGeoTransform()
    ds = None

	# Get DEM
#     dem = load_interpolated_dem()

    # Allocate arrays for velocities and errors
    vx = np.zeros((N_dates, Ny, Nx), dtype=np.float32)
    vy = np.zeros((N_dates, Ny, Nx), dtype=np.float32)
    ex = np.zeros((N_dates, Ny, Nx), dtype=np.float32)
    ey = np.zeros((N_dates, Ny, Nx), dtype=np.float32)
    heading = np.zeros(N_dates)
    counts = np.zeros((Ny, Nx))

    # Loop over rasters
    for i in tqdm(range(len(vx_files))):

        # Load vx
        ds = gdal.Open(vx_files[i], gdal.GA_ReadOnly)
        vx_dat = ds.GetRasterBand(1).ReadAsArray()
        ds = None

        # Load vy
        ds = gdal.Open(vy_files[i], gdal.GA_ReadOnly)
        vy_dat = ds.GetRasterBand(1).ReadAsArray()
        ds = None

        # Load vx
        ds = gdal.Open(ex_files[i], gdal.GA_ReadOnly)
        ex_dat = ds.GetRasterBand(1).ReadAsArray()
        ds = None

        # Load vy
        ds = gdal.Open(ey_files[i], gdal.GA_ReadOnly)
        ey_dat = ds.GetRasterBand(1).ReadAsArray()
        ds = None

        # Compute heading
        try:
            heading[i] = compute_heading(vx_dat, skip=15)
        except ValueError:
            heading[i] = np.nan
            continue

        # Mask out bad values
        mask = (np.abs(vx_dat) > 1e6) + (ex_dat < 0.0) + (ex_dat > 100.0)
        vx_dat[mask] = np.nan
        vy_dat[mask] = np.nan
        ex_dat[mask] = np.nan
        ey_dat[mask] = np.nan

        # Scale and save
        vx[i,:,:] = 1.0e-3 * vx_dat
        vy[i,:,:] = 1.0e-3 * vy_dat
        ex[i,:,:] = 1.0e-3 * ex_dat
        ey[i,:,:] = 1.0e-3 * ey_dat

        # Update counters
        counts[np.invert(mask)] += 1

    # Only keep good headings
    mask = np.isfinite(heading)
    vx, vy, ex, ey = vx[mask], vy[mask], ex[mask], ey[mask]
    heading = heading[mask]
    tdec = tdec[mask]
    vx_files = vx_files[mask]
    vy_files = vy_files[mask]
    ex_files = ex_files[mask]
    ey_files = ey_files[mask]
    N_dates = len(heading)

    # Create arrays for coordinates
    x = x_start + dx * np.arange(Nx)
    y = y_start + dy * np.arange(Ny)
    X, Y = np.meshgrid(x, y)

    # Initialize stack directory
    if not os.path.isdir('Stack'):
        os.mkdir('Stack')

    # Convert errors into weights
    wx = 1.0 / (25.0 * np.sqrt(ex))
    wy = 1.0 / (25.0 * np.sqrt(ey))
    del ex, ey

    # Spatially subset
    islice = slice(120, 580)
    jslice = slice(240, 878)
    vx = vx[:,islice,jslice]
    vy = vy[:,islice,jslice]
    wx = wx[:,islice,jslice]
    wy = wy[:,islice,jslice]
#     dem = dem[islice,jslice]
    X = X[islice,jslice]
    Y = Y[islice,jslice]
    Ny, Nx = X.shape

    # Create stack for Vx data
    with h5py.File('Stack/vx.h5', 'w') as fid:
        chunks = (1, 128, 128)
        fid.create_dataset('igram', (N_dates, Ny, Nx), dtype='f', data=vx, chunks=chunks)
        fid.create_dataset('weights', (N_dates, Ny, Nx), dtype='f', data=wx, chunks=chunks)
        fid['tdec'] = tdec
        fid['x'] = X
        fid['y'] = Y
#         fid['z'] = dem
        fid['chunk_shape'] = list(chunks)
        fid['vx_files'] = vx_files
        fid['vy_files'] = vy_files
        fid['heading'] = heading

    # Create stack for Vy data
    with h5py.File('Stack/vy.h5', 'w') as fid:
        chunks = (1, 128, 128)
        fid.create_dataset('igram', (N_dates, Ny, Nx), dtype='f', data=vy, chunks=chunks)
        fid.create_dataset('weights', (N_dates, Ny, Nx), dtype='f', data=wy, chunks=chunks)
        fid['tdec'] = tdec
        fid['x'] = X
        fid['y'] = Y
#         fid['z'] = dem
        fid['chunk_shape'] = list(chunks)
        fid['vx_files'] = vx_files
        fid['vy_files'] = vy_files
        fid['heading'] = heading


def parseMeta(filename):
    """
    Parse the metadata for dates.
    """
    with open(filename, 'r') as fid:
        for line in fid:
            if line.startswith('First Image Date'):
                dstr = line.strip().split('=')[-1].strip()
                first_date = datetime.datetime.strptime(dstr, '%b:%d:%Y')
            elif line.startswith('Second Image Date'):
                dstr = line.strip().split('=')[-1].strip()
                second_date = datetime.datetime.strptime(dstr, '%b:%d:%Y')
            elif line.startswith('Product Center Latitude'):
                vstr = line.strip().split('=')[-1].strip()
                clat = float(vstr)
            elif line.startswith('Product Center Longitude'):
                vstr = line.strip().split('=')[-1].strip()
                clon = float(vstr)
            elif line.startswith('Nominal Time'):
                tstr = line.strip().split('=')[-1].strip()
                hh, mm, ss = [int(val) for val in tstr.split(':')]
                nominal_dt = hh * 3600.0 + mm * 60.0 + ss

    return first_date, second_date, nominal_dt


# def load_interpolated_dem():
# 
#     # Get hdr information from random velocity tif file
#     vhdr = load_gdal('/data0/briel/measures/nsidc0481_MEASURES_greenland_V01/Wcoast-69.10N/TSX_Sep-11-2012_Sep-22-2012_20-41-24/TSX_W69.10N_11Sep12_22Sep12_20-41-24_ex_v1.2.tif', hdr_only=True)
# 
#     # Load DEM
#     dem, dhdr = load_gdal('arcticdem_crop.dem')
# 
#     # Velocity grid meshgrid coordinates
#     x = vhdr.x0 + vhdr.dx * np.arange(vhdr.nx)
#     y = vhdr.y0 + vhdr.dy * np.arange(vhdr.ny)
#     X, Y = np.meshgrid(x, y)
# 
#     # Interpolate DEM to velocity grid
#     dem = interpolate_raster(dem, dhdr, X.ravel(), Y.ravel())
#     dem = dem.reshape(vhdr.ny, vhdr.nx)
#     return dem


def load_gdal(filename, hdr_only=False):
    hdr = GenericClass()
    dset = gdal.Open(filename, gdal.GA_ReadOnly)
    hdr.x0, hdr.dx, _, hdr.y0, _, hdr.dy = dset.GetGeoTransform()
    hdr.ny = dset.RasterYSize
    hdr.nx = dset.RasterXSize
    if hdr_only:
        return hdr
    else:
        d = dset.GetRasterBand(1).ReadAsArray()
        return d, hdr


def interpolate_raster(data, hdr, x, y):
    row = (y - hdr.y0) / hdr.dy
    col = (x - hdr.x0) / hdr.dx
    coords = np.vstack((row, col))
    values = map_coordinates(data, coords, order=3, prefilter=False)
    return values


def compute_heading(v, skip=10):

    dy = -100.0
    dx = 100.0
    ny, nx = v.shape

    # Get left edge
    ycoords = dy * np.arange(0, ny, skip)
    xcoords = np.full(ycoords.shape, np.nan)
    for cnt, i in enumerate(range(0, ny, skip)):
        good_ind = (v[i,:] > -20000).nonzero()[0]
        if len(good_ind) < 10:
            continue
        xcoords[cnt] = dx * good_ind[-1]

    # Solve linear
    mask = np.isfinite(xcoords)
    ycoords, xcoords = ycoords[mask], xcoords[mask]
    X = np.column_stack((ycoords, np.ones_like(ycoords)))
    solver = RANSACRegressor().fit(X, xcoords)
    fit = solver.predict(X)

    # Compute heading
    slope = solver.estimator_.coef_[0]
    heading = np.degrees(np.arctan(slope))

    return heading


class GenericClass:
    pass


if __name__ == '__main__':
    main()

# end of file
