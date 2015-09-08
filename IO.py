__author__ = 'lpeng'
import os, glob
import numpy as np
from netCDF4 import Dataset

def Retrieve_Foldername(root):

	##### ALL files without path #####
	#---------------------------------
	# walk through all the subfolder names under one directory
	# For path = i - 1 level, where i is the deepest level
	folder = next(os.walk(root))[1]

	return folder

def Retrieve_Filename(root):

	##### ALL files without path #####
	#---------------------------------
	# walk through all the file names under one directory
	# For path = i level, where i is the deepest level
	nfile = next(os.walk(root))[2]

	return nfile

def Retrieve_File_by_Extension(root, extension):

	##### filename matching #####
	#----------------------------
	# filter the files with certain conditions
	# with full path+filename pattern
	for filename in glob.glob(root + '/*.' + extension):
		print filename
		base = os.path.basename(filename)
		print base

	return

def Land_Mask(root, model):

	filename2 = 'historical_r1i1p1_185001-200512.nc'
	var = 'mrso'
	res = 'monthly'
	data = Dataset('%s%s/%s/%s_Lmon_%s_%s' % (root, model, res, var, model, filename2)).variables[var][:]
	mask = np.mean(data, axis=0)
	mask[mask>0] = 1.0
	mask[mask==0] = np.nan

	return mask

# split the filename in the name and its extension
# baselist = []  # definition
# for base in nfile:
# 	basename = os.path.splitext(base)[0]
