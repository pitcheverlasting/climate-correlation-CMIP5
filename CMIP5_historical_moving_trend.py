__author__ = 'lpeng'
import sys
import time
import numpy as np
import scipy.stats.mstats as mstats
import pandas as pd
from pandas import Series, DataFrame
import gc, IO
from pylab import *
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap, maskoceans
gc.collect()

# input path
root = '/home/air2/jherrera/Data/CMIP5/historical/'
figdir = '/home/water5/lpeng/Figure/CMIP5/201509/'
res = ['annual', 'monthly']
# models = IO.Retrieve_Foldername(root)
models = ('BNU-ESM', 'CCSM4', 'CSIRO-Mk3-6-0', 'CESM1-BGC', 'IPSL-CM5A-LR', 'MPI-ESM-LR')
vars = ('tas', 'rlds', 'rsds', 'rlus', 'rsus', 'pr', 'hfls', 'hfss')
limits = ([-1.2, 1.2], [-3., 3.], [-3., 3.], [-3., 3.], [-2., 2.], [-0.2, 0.2], [-1., 1.], [-0.5, 0.5])
filename2 = 'historical_r1i1p1_185001-200512.nc'
styr = 1900
edyr = 2005
stmon = (styr - 1850) * 12
# tstep = len(dates)  # tstep = (edyr-styr+1)*12

for m, model in enumerate(models):
	for v, var in enumerate(vars):
		data = Dataset('%s%s/%s/%s_Amon_%s_%s' % (root, model, res[1], var, model, filename2)).variables[var][stmon:, :, :]
		mask = IO.Land_Mask(root, model)
		data_land_monthly = nanmean(nanmean(data*mask, axis=2), axis=1)
		# calculate global annual mean
		data_land_annual = vstack([sum(data_land_monthly[mon:mon+12]) for mon in xrange(0, len(data_land_monthly), 12)])

		# calculate moving trend
		slope = np.empty((edyr-styr-9, edyr-styr-9))
		slope.fill(np.nan)
		for st in xrange(0, edyr - styr - 9):
			for ed in xrange(st+10, edyr - styr + 1):
				slope[ed - 10, st] = mstats.theilslopes(data_land_annual[st:ed], alpha=0.95)[0]

		# Mapping
		print model, var
		x = np.arange(styr, edyr-9, 1.)
		y = np.arange(styr+10, edyr+1, 1.)
		X, Y = np.meshgrid(x, y)
		# create figure
		clevs = arange(limits[v][0], limits[v][1]+0.01, (limits[v][1]-limits[v][0])/100)
		cblevs = arange(limits[v][0], limits[v][1]+0.01, round((limits[v][1]-limits[v][0])/10, 2))
		fig = plt.figure(figsize=(12, 8), dpi=100, facecolor="white")
		font = {'family': 'serif', 'color': 'darkred', 'weight': 'normal', 'size': 50}
		im = plt.contourf(X, Y, slope, clevs, cmap=plt.cm.seismic)
		cb = plt.colorbar(im, ticks=cblevs)
		plt.xlabel("STARTING YEAR")
		plt.ylabel("ENDING YEAR")
		plt.title('%s %s Moving Trend %s-%s' % (model, var, str(styr), str(edyr)))
		savefig('%s/global_moving_trend_%s_%s_%s-%s.png' %(figdir, model, var, str(styr), str(edyr)))
		plt.clf()


exit()

# data = [Dataset('%s/%s_Amon_GFDL-ESM2G_historical_r1i1p1_%s01-%s12.nc' % (root, forcing[i], str(year), str(year+4))).variables[forcing[i]][:] for year in xrange(styr, edyr+1, 5)]


# # wind (uas,vas)
# uas = [Dataset('%s/uas_Amon_GFDL-ESM2G_historical_r1i1p1_%s01-%s12.nc' % (root, str(year), str(year+4))).variables[forcing[5]][:] for year in xrange(styr, edyr+1, 5)]
# uas = np.vstack(uas)
# vas = [Dataset('%s/vas_Amon_GFDL-ESM2G_historical_r1i1p1_%s01-%s12.nc' % (root, str(year), str(year+4))).variables[forcing[6]][:] for year in xrange(styr, edyr+1, 5)]
# vas = np.vstack(vas)
# data = sqrt(uas**2 + vas**2)
