__author__ = 'lpeng'
import sys
import time
import numpy as np
import scipy.stats.mstats as mstats
import pandas as pd
from pandas import Series, DataFrame
import gc
from pylab import *
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap, maskoceans
gc.collect()

# input path
homedir = '/home/latent1/lpeng/Data/justin/historical/'
figdir = '/home/water5/lpeng/Figure/CMIP5/'
forcing = ('tas','huss','ps','rlds','rsds','uas','vas')

# resolution: 2 * 2.5
glon = 144
glat = 90
styr = 1861
edyr = 2005
dates = pd.date_range((str(styr)), (str(edyr+1)), freq='A')
tstep = len(dates)  # tstep = (edyr-styr+1)*12
print tstep

# loop for
for i in xrange(0, len(forcing)-1):
	for j in xrange(i+1, len(forcing)):
		data1 = [Dataset('%s/%s_Amon_GFDL-ESM2G_historical_r1i1p1_%s01-%s12.nc' % (homedir, forcing[i], str(year), str(year+4))).variables[forcing[i]][:] for year in xrange(styr, edyr+1, 5)]
		data2 = [Dataset('%s/%s_Amon_GFDL-ESM2G_historical_r1i1p1_%s01-%s12.nc' % (homedir, forcing[j], str(year), str(year+4))).variables[forcing[j]][:] for year in xrange(styr, edyr+1, 5)]
		data1 = np.vstack(data1)
		data2 = np.vstack(data2)
		# get global monthly 3D datasets
		ts1 = np.mean(np.mean(data1, axis=2), axis=1)
		ts2 = np.mean(np.mean(data2, axis=2), axis=1)
		# calculate global annual mean
		ann1 = []
		ann2 = []
		for mon in xrange(0, len(ts1), 12):
			ann1.append(np.sum(ts1[mon:mon+12]))
			ann2.append(np.sum(ts2[mon:mon+12]))
		# calculate moving correlation
		# Mapping for the year
		x = np.arange(styr, edyr+1, 1.)
		y = np.arange(1., edyr-styr+1, 1.)
		X, Y = np.meshgrid(x, y)

		stat = np.empty((tstep-1, tstep))
		stat.fill(np.nan)

		for wind in xrange(1, tstep):
			data1 = Series(ann1, index=dates)
			data2 = Series(ann2, index=dates)
			print pd.rolling_corr(data1, data2, window=wind)
			stat[wind-1, :] = pd.rolling_corr(data1, data2, window=wind)  # tstep-wind-1
			print wind


		# create figure
		fig = plt.figure(figsize=(12, 8), dpi=100, facecolor="white")
		font = {'family': 'serif', 'color': 'darkred', 'weight': 'normal', 'size': 20}

		clevs = np.arange(-1.0, 1.1, 0.1)
		im = plt.contourf(X, Y, stat, clevs, cmap=plt.cm.jet)
		del clevs
		cb = plt.colorbar(im, ticks=np.arange(-1.0, 1.1, 0.2)) #, "right", size="5%", pad='2%',
		plt.xlabel("ENDING YEAR")
		plt.ylabel("WINDOW LENGTH")
		plt.title('CMIP5 Global Moving COR(%s,%s) %s-%s' % (forcing[i], forcing[j], str(styr), str(edyr)))
		# plt.show()
		savefig('%s/global_moving_cor_%s_%s_%s-%s.png' %(figdir, forcing[i], forcing[j], str(styr), str(edyr)))
		plt.clf()

		# # mask out ocean grids
		# fig = plt.figure(figsize=(12, 8), dpi=100, facecolor="white")
		# m = Basemap(projection='cyl', llcrnrlat=-60, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='l')
		# # draw parraels, coastlines, edge of map
		# m.drawcoastlines()
		# m.drawmeridians(np.arange(-180., 181., 60.), labels=[0, 0, 0, 1])
		# m.drawparallels(np.arange(-60., 91., 30.), labels=[1, 0, 0, 0])
		# clevs = np.arange(0, 361, 60)
		# print np.mean(ts,axis=0).shape
		# z = maskoceans(X, Y, np.mean(ts,axis=0))
		# print z
		# im = m.contourf(X, Y, z, clevs, cmap=plt.cm.jet)
		# plt.show()

