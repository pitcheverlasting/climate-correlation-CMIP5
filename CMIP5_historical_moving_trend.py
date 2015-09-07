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
forcing = ('tas', 'rlds', 'huss', 'ps', 'rsds', 'uas', 'vas')

# resolution: 2 * 2.5
glon = 144
glat = 90
styr = 1861
edyr = 2005
dates = pd.date_range((str(styr)), (str(edyr+1)), freq='A')
tstep = len(dates)  # tstep = (edyr-styr+1)*12
print tstep

for i in xrange(5, 6):  # len(forcing)):
	data = [Dataset('%s/%s_Amon_GFDL-ESM2G_historical_r1i1p1_%s01-%s12.nc' % (homedir, forcing[i], str(year), str(year+4))).variables[forcing[i]][:] for year in xrange(styr, edyr+1, 5)]
	data = np.vstack(data)
	# get global monthly 3D datasets
	ts = np.mean(np.mean(data, axis=2), axis=1)
	# calculate global annual mean
	ann = []
	[ann.append(np.sum(ts[mon:mon+12])) for mon in xrange(0, len(ts), 12)]

	# calculate moving trend
	slope = np.empty((edyr-styr-9, edyr-styr-9))
	slope.fill(np.nan)
	for st in xrange(0, edyr - styr - 9):
		for ed in xrange(st+10, edyr - styr + 1):
			slope[ed - 10, st] = mstats.theilslopes(ann[st:ed])[0]
	# print slope[0,0]
	# Mapping
	x = np.arange(styr, edyr-9, 1.)
	y = np.arange(styr+10, edyr+1, 1.)
	X, Y = np.meshgrid(x, y)

	# create figure
	clevs = np.arange(-1.2, 1.3, 0.01)
	tclevs = np.arange(-1.2, 1.3, 0.2)
	fig = plt.figure(figsize=(12, 8), dpi=100, facecolor="white")
	font = {'family': 'serif', 'color': 'darkred', 'weight': 'normal', 'size': 20}
	im = plt.contourf(X, Y, slope*10, clevs, cmap=plt.cm.jet)
	cb = plt.colorbar(im, ticks=tclevs)
	# del clevs
	plt.xlabel("STARTING YEAR")
	plt.ylabel("ENDING YEAR")
	plt.title('CMIP5 Global %s Moving Trend %s-%s' % (forcing[i], str(styr), str(edyr)))
	savefig('%s/global_moving_trend_%s_%s-%s.png' %(figdir, forcing[i], str(styr), str(edyr)))
	plt.clf()
	#plt.show()


# # wind (uas,vas)
# uas = [Dataset('%s/uas_Amon_GFDL-ESM2G_historical_r1i1p1_%s01-%s12.nc' % (homedir, str(year), str(year+4))).variables[forcing[5]][:] for year in xrange(styr, edyr+1, 5)]
# uas = np.vstack(uas)
# vas = [Dataset('%s/vas_Amon_GFDL-ESM2G_historical_r1i1p1_%s01-%s12.nc' % (homedir, str(year), str(year+4))).variables[forcing[6]][:] for year in xrange(styr, edyr+1, 5)]
# vas = np.vstack(vas)
# data = sqrt(uas**2 + vas**2)
# # get global monthly 3D datasets
# ts = np.mean(np.mean(data, axis=2), axis=1)
# # calculate global annual mean
# ann = []
# [ann.append(np.sum(ts[mon:mon+12])) for mon in xrange(0, len(ts), 12)]
#
# # calculate moving trend
# slope = np.empty((edyr-styr-9, edyr-styr-9))
# slope.fill(np.nan)
# for st in xrange(0, edyr - styr - 9):
# 	for ed in xrange(st+10, edyr - styr + 1):
# 		slope[ed - 10, st] = mstats.theilslopes(ann[st:ed])[0]
# # print slope[0,0]
# # Mapping
# x = np.arange(styr, edyr-9, 1.)
# y = np.arange(styr+10, edyr+1, 1.)
# X, Y = np.meshgrid(x, y)
#
# # create figure
# clevs = np.arange(-0.15, 0.16, 0.01)
# tclevs = np.arange(-0.15, 0.16, 0.05)
# fig = plt.figure(figsize=(12, 8), dpi=100, facecolor="white")
# font = {'family': 'serif', 'color': 'darkred', 'weight': 'normal', 'size': 20}
# im = plt.contourf(X, Y, slope, clevs, cmap=plt.cm.jet)
# cb = plt.colorbar(im, ticks=tclevs)
# # del clevs
# plt.xlabel("STARTING YEAR")
# plt.ylabel("ENDING YEAR")
# plt.title('CMIP5 Global Wind Moving Trend %s-%s' % (str(styr), str(edyr)))
# savefig('%s/global_moving_trend_wind_%s-%s.png' % (figdir, str(styr), str(edyr)))
# plt.clf()
#
# exit()

