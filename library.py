## High Resolution co-registration ##
## Library ##

import os,sys
import numpy as np
import scipy as sp
from scipy import ndimage
import osgeo.gdal
import osgeo.ogr
from gdalconst import *
import cv2
import time
import collections
import subprocess
import random
import shutil
from sklearn import linear_model
import gdal

from operator import itemgetter, attrgetter
from numpy.fft import fft2, ifft2, fftshift


def data_type2gdal_data_type(data_type):

	'''Conversion from numpy data type to GDAL data type

	:param data_type: numpy type (e.g. np.uint8, np.int32; 0 for default: np.uint16) (numpy type).
	:returns: corresponding GDAL data type
	:raises: AttributeError, KeyError

	Author: Daniele De Vecchi - Mostapha Harb
	Last modified: 18/03/2014
	'''
	#Function needed when it is necessary to write an output file
	if data_type == np.uint16:
		return GDT_UInt16
	if data_type == np.uint8:
		return GDT_Byte
	if data_type == np.int32:
		return GDT_Int32
	if data_type == np.float32:
		return GDT_Float32
	if data_type == np.float64:
		return GDT_Float64


def read_image(input_raster,data_type,band_selection):

	'''Read raster using GDAL

	:param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string).
	:param data_type: numpy type used to read the image (e.g. np.uint8, np.int32; 0 for default: np.uint16) (numpy type).
	:param band_selection: number associated with the band to extract (0: all bands, 1: blue, 2: green, 3:red, 4:infrared) (integer).
	:returns:  a list containing the desired bands as ndarrays (list of arrays).
	:raises: AttributeError, KeyError

	Author: Daniele De Vecchi - Mostapha Harb
	Last modified: 18/03/2014
	'''

	band_list = []

	if data_type == 0: #most of the images (MR and HR) can be read as uint16
		data_type = np.uint16

	inputimg = osgeo.gdal.Open(input_raster, GA_ReadOnly)
	cols=inputimg.RasterXSize
	rows=inputimg.RasterYSize
	nbands=inputimg.RasterCount

	if band_selection == 0:
		#read all the bands
		for i in range(1,nbands+1):
			inband = inputimg.GetRasterBand(i)
			#mat_data = inband.ReadAsArray(0,0,cols,rows).astype(data_type)
			mat_data = inband.ReadAsArray().astype(data_type)
			band_list.append(mat_data)
	else:
		#read the single band
		inband = inputimg.GetRasterBand(band_selection)
		mat_data = inband.ReadAsArray(0,0,cols,rows).astype(data_type)
		band_list.append(mat_data)

	inputimg = None
	return band_list


def read_image_parameters(input_raster):

	'''Read raster parameters using GDAL

	:param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string).
	:returns:  a list containing rows, columns, number of bands, geo-transformation matrix and projection.
	:raises: AttributeError, KeyError

	Author: Daniele De Vecchi - Mostapha Harb
	Last modified: 18/03/2014
	'''

	inputimg = osgeo.gdal.Open(input_raster, GA_ReadOnly)
	cols=inputimg.RasterXSize
	rows=inputimg.RasterYSize
	nbands=inputimg.RasterCount
	geo_transform = inputimg.GetGeoTransform()
	projection = inputimg.GetProjection()

	inputimg = None
	return rows,cols,nbands,geo_transform,projection


def world2pixel(geo_transform, long, lat):

	'''Conversion from geographic coordinates to matrix-related indexes

	:param geo_transform: geo-transformation matrix containing coordinates and resolution of the output (array of 6 elements, float)
	:param long: longitude of the desired point (float)
	:param lat: latitude of the desired point (float)
	:returns: A list with matrix-related x and y indexes (x,y)
	:raises: AttributeError, KeyError

	Author: Daniele De Vecchi - Mostapha Harb
	Last modified: 18/03/2014
	'''

	ulX = geo_transform[0] #starting longitude
	ulY = geo_transform[3] #starting latitude
	xDist = geo_transform[1] #x resolution
	yDist = geo_transform[5] #y resolution

	pixel_x = int((long - ulX) / xDist)
	pixel_y = int((ulY - lat) / abs(yDist))
	return (pixel_x, pixel_y)


def write_image(band_list,data_type,band_selection,output_raster,rows,cols,geo_transform,projection):

	'''Write array to file as raster using GDAL

	:param band_list: list of arrays containing the different bands to write (list of arrays).
	:param data_type: numpy data type of the output image (e.g. np.uint8, np.int32; 0 for default: np.uint16) (numpy type)
	:param band_selection: number associated with the band to write (0: all, 1: blue, 2: green, 3: red, 4: infrared) (integer)
	:param output_raster: path and name of the output raster to create (*.TIF, *.tiff) (string)
	:param rows: rows of the output raster (integer)
	:param cols: columns of the output raster (integer)
	:param geo_transform: geo-transformation matrix containing coordinates and resolution of the output (array of 6 elements, float)
	:param projection: projection of the output image (string)
	:returns: An output file is created
	:raises: AttributeError, KeyError

	Author: Daniele De Vecchi - Mostapha Harb
	Last modified: 18/03/2014
	'''

	if data_type == 0:
		gdal_data_type = GDT_UInt16 #default data type
	else:
		gdal_data_type = data_type2gdal_data_type(data_type)

	driver = osgeo.gdal.GetDriverByName('GTiff')

	if band_selection == 0:
		nbands = len(band_list)
	else:
		nbands = 1
	outDs = driver.Create(output_raster, cols, rows,nbands, gdal_data_type)
	if outDs is None:
		print 'Could not create output file'
		sys.exit(1)

	if band_selection == 0:
		#write all the bands to file
		for i in range(0,nbands):
			outBand = outDs.GetRasterBand(i+1)
			outBand.WriteArray(band_list[i], 0, 0)
	else:
		#write the specified band to file
		outBand = outDs.GetRasterBand(1)
		outBand.WriteArray(band_list[band_selection-1], 0, 0)
	#assign geomatrix and projection
	outDs.SetGeoTransform(geo_transform)
	outDs.SetProjection(projection)
	outDs = None


def get_coordinate_limit(input_raster):

	'''Get corner cordinate from a raster

	:param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string)
	:returs: minx,miny,maxx,maxy: points taken from geomatrix (string)

	Author: Daniel Aurelio Galeazzo - Daniele De Vecchi - Mostapha Harb
	Last modified: 23/05/2014
	'''
	dataset = osgeo.gdal.Open(input_raster, GA_ReadOnly)
	if dataset is None:
		print 'Could not open'
		sys.exit(1)
	driver = dataset.GetDriver()
	band = dataset.GetRasterBand(1)

	width = dataset.RasterXSize
	height = dataset.RasterYSize
	geoMatrix = dataset.GetGeoTransform()
	minx = geoMatrix[0]
	miny = geoMatrix[3] + width*geoMatrix[4] + height*geoMatrix[5]
	maxx = geoMatrix[0] + width*geoMatrix[1] + height*geoMatrix[2]
	maxy = geoMatrix[3]

	dataset = None

	return minx,miny,maxx,maxy


def extract_tiles(input_raster,start_col_coord,start_row_coord,end_col_coord,end_row_coord,data_type):

	'''
	Extract a subset of a raster according to the desired coordinates

	:param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string)
	:param output_raster: path and name of the output raster file (*.TIF,*.tiff) (string)
	:param start_col_coord: starting longitude coordinate
	:param start_row_coord: starting latitude coordinate
	:param end_col_coord: ending longitude coordinate
	:param end_row_coord: ending latitude coordinate

	:returns: an output file is created and also a level of confidence on the tile is returned

	Author: Daniele De Vecchi
	Last modified: 20/08/2014
	'''

	#Read input image
	rows,cols,nbands,geotransform,projection = read_image_parameters(input_raster)
	band_list = read_image(input_raster,data_type,0)
	#Definition of the indices used to tile
	start_col_ind,start_row_ind = world2pixel(geotransform,start_col_coord,start_row_coord)
	end_col_ind,end_row_ind = world2pixel(geotransform,end_col_coord,end_row_coord)
	#print start_col_ind,start_row_ind
	#print end_col_ind,end_row_ind
	#New geotransform matrix
	new_geotransform = [start_col_coord,geotransform[1],0.0,start_row_coord,0.0,geotransform[5]]
	#Extraction
	data = band_list[1][start_row_ind:end_row_ind,start_col_ind:end_col_ind]

	band_list = []
	return data,start_col_coord,start_row_coord,end_col_coord,end_row_coord


def ORIGINAL_SURF(ref_band_mat,target_band_mat,output_as_array):

	'''
	SURF version taken from website example
	http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_surf_intro/py_surf_intro.html#surf
	https://gist.github.com/moshekaplan/5106221

	:param ref_band_mat: numpy 8 bit array containing reference image
	:param target_band_mat: numpy 8 bit array containing target image
	:param output_as_array: if True the output is converted to matrix for visualization purposes
	:returns: points from reference, points from target, result of matching function or array of points (depending on the output_as_array flag)

	'''

	detector = cv2.SURF(400) #Detector definition, Hessian threshold set to 400 (suggested between 300 and 500)
	detector.extended = True #Descriptor extended to 128
	detector.upright = True #Avoid orientation
	matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

	#Extraction of features from REFERENCE
	ref_mask_zeros = np.ma.masked_equal(ref_band_mat, 0).astype('uint8')
	kp_ref, des_ref = detector.detectAndCompute(ref_band_mat, mask=ref_mask_zeros)
	h_ref, w_ref = ref_band_mat.shape[:2]
	print h_ref, w_ref
	ref_band_mat = []

	#Extraction of features from TARGET
	target_mask_zeros = np.ma.masked_equal(target_band_mat, 0).astype('uint8')
	kp_target, des_target = detector.detectAndCompute(target_band_mat, mask=target_mask_zeros)
	h_target, w_target = target_band_mat.shape[:2]
	print h_target, w_target
	target_band_mat = []

	#Matching
	print des_ref
	print des_target
	if (des_target != None) and (des_ref != None):
		matches = matcher.match(des_ref,des_target)
	else:
		des_ref = None
		des_target = None
		matches = matcher.match(des_ref,des_target)

	if output_as_array == True:
		ext_points = np.zeros(shape=(len(matches),4))
		i = 0
		for m in matches:
			ext_points[i][:]= [int(kp_ref[m.queryIdx].pt[0]),int(kp_ref[m.queryIdx].pt[1]),int(kp_target[m.trainIdx].pt[0]),int(kp_target[m.trainIdx].pt[1])]
			i = i+1
		return kp_ref,kp_target,ext_points
	else:
		return kp_ref,kp_target,matches


def linear_quantization(input_mat,quantization_factor):
	
	'''Quantization of all the input bands cutting the tails of the distribution
	
	:param input_band_list: list of 2darrays (list of 2darrays)
	:param quantization_factor: number of levels as output (integer)
	:returns:  list of values corresponding to the quantized bands (list of 2darray)
	:raises: AttributeError, KeyError
	
	Author: Daniele De Vecchi - Mostapha Harb
	Last modified: 12/05/2014
	'''

	q_factor = quantization_factor - 1
	inmatrix = input_mat.reshape(-1)
	print np.min(inmatrix),np.max(inmatrix)
	out = np.bincount(inmatrix)
	tot = inmatrix.shape[0]
	freq = (out.astype(np.float32)/float(tot))*100 #frequency for each value
	cumfreqs = np.cumsum(freq)
	first = np.where(cumfreqs>1.49)[0][0] #define occurrence limits for the distribution
	last = np.where(cumfreqs>97.8)[0][0]
	input_mat[np.where(input_mat>last)] = last
	input_mat[np.where(input_mat<first)] = first

	k1 = float(q_factor)/float((last-first)) #k1 term of the quantization formula
	k2 = np.ones(input_mat.shape)-k1*first*np.ones(input_mat.shape) #k2 term of the quantization formula
	out_matrix = np.floor(input_mat*k1+k2) #take the integer part
	out_matrix2 = out_matrix-np.ones(out_matrix.shape)
	out_matrix2.astype(np.uint8)

	return out_matrix2


def ORIGINAL_SIFT(ref_band_mat,target_band_mat,output_as_array):

	'''
	SIFT algorithm

	:param ref_band_mat: numpy 8 bit array containing reference image
	:param target_band_mat: numpy 8 bit array containing target image
	:param output_as_array: if True the output is converted to matrix for visualization purposes
	:returns: points from reference, points from target, result of matching function or array of points (depending on the output_as_array flag)
	'''

	#detector = cv2.xfeatures2d.SIFT_create()
	detector = cv2.SIFT()
	matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

	#Extraction of features from REFERENCE
	ref_mask_zeros = np.ma.masked_equal(ref_band_mat, 0).astype('uint8') #mask borders
	kp_ref, des_ref = detector.detectAndCompute(ref_band_mat.astype(np.uint8), mask=ref_mask_zeros)
	h_ref, w_ref = ref_band_mat.shape[:2]
	print h_ref, w_ref
	ref_band_mat = []

	#Extraction of features from TARGET
	target_mask_zeros = np.ma.masked_equal(target_band_mat, 0).astype('uint8') #mask borders
	kp_target, des_target = detector.detectAndCompute(target_band_mat.astype('uint8'), mask=target_mask_zeros)
	h_target, w_target = target_band_mat.shape[:2]
	print h_target, w_target
	target_band_mat = []

	#Matching
	print des_ref
	print des_target
	if (des_target != None) and (des_ref != None):
		matches = matcher.match(des_ref,des_target)
	else:
		des_ref = None
		des_target = None
		matches = matcher.match(des_ref,des_target)

	if output_as_array == True:
		ext_points = np.zeros(shape=(len(matches),4))
		i = 0
		for m in matches:
			ext_points[i][:]= [int(kp_ref[m.queryIdx].pt[0]),int(kp_ref[m.queryIdx].pt[1]),int(kp_target[m.trainIdx].pt[0]),int(kp_target[m.trainIdx].pt[1])]
			i = i+1
		return kp_ref,kp_target,ext_points
	else:
		return kp_ref,kp_target,matches


def ORIGINAL_ORB(ref_band_mat,target_band_mat,output_as_array,enable_knn,enable_ratio_filter,enable_ransac,enable_custom_ransac):

	'''
	ORB algorithm

	:param ref_band_mat: numpy 8 bit array containing reference image
	:param target_band_mat: numpy 8 bit array containing target image
	:param output_as_array: if True the output is converted to matrix for visualization purposes
	:param enable_ransac: if True, a different matcher is used and a ransac filter is applied
	:returns: points from reference, points from target, result of matching function or array of points (depending on the output_as_array flag)
	'''

	detector = cv2.ORB()
	matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	#Extraction of features from REFERENCE
	ref_mask_zeros = np.ma.masked_equal(ref_band_mat, 0).astype('uint8') #mask borders
	kp_ref, des_ref = detector.detectAndCompute(ref_band_mat, mask=ref_mask_zeros)
	h_ref, w_ref = ref_band_mat.shape[:2]
	print h_ref, w_ref
	ref_band_mat = []

	#Extraction of features from TARGET
	target_mask_zeros = np.ma.masked_equal(target_band_mat, 0).astype('uint8') #mask borders
	kp_target, des_target = detector.detectAndCompute(target_band_mat, mask=target_mask_zeros)
	h_target, w_target = target_band_mat.shape[:2]
	print h_target, w_target
	target_band_mat = []

	#Matching
	#print des_ref
	#print des_target
	if (des_target != None) and (des_ref != None):
		if enable_knn == False:
			matches = matcher.match(des_ref,des_target)
		else:
			FLANN_INDEX_KDTREE = 0
			index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
			search_params = dict(checks = 50)
			flann = cv2.FlannBasedMatcher(index_params, search_params)
			if (len(kp_ref) >= 2) and (len(kp_target) >= 2):
				matches = flann.knnMatch(des_ref.astype(np.float32),des_target.astype(np.float32),k=2)
			else:
				des_ref = None
				des_target = None
				matches = matcher.match(des_ref,des_target)

		if enable_ratio_filter == False:
			if enable_knn == True:
				matches = sorted(matches, key = lambda x:x[0].distance)
			else:
				matches = sorted(matches, key = lambda x:x.distance)
			matches = matches[:1]
			p_ref, p_target, kp_pairs = convert_matches(kp_ref, kp_target, matches,enable_knn)
		else:
			p_ref, p_target, kp_pairs = filter_matches(kp_ref, kp_target, matches)

		

		if enable_ransac == True:
			if len(p_ref) >= 4:
				H, status = cv2.findHomography(p_ref, p_target, cv2.RANSAC, 2.0)
				print '%d / %d  inliers/matched' % (np.sum(status), len(status))
				# do not draw outliers (there will be a lot of them)
				kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]
				'''
				if output_as_array == True:
					ext_points = np.zeros(shape=(len(kp_pairs),4))
					i = 0
					for kpp in kp_pairs:
						ext_points[i][:] = [int(kpp[0].pt[0]),int(kpp[0].pt[1]),int(kpp[1].pt[0]),int(kpp[1].pt[1])]
						print ext_points[i][:]
						i +=1 
				'''
					#return kp_ref,kp_target,ext_points
			else:
				kp_pairs = np.zeros(shape=(0,4))
				#ext_points = np.zeros(shape=(0,4))
				#return kp_ref,kp_target,ext_points
				
	else:
		des_ref = None
		des_target = None
		matches = matcher.match(des_ref,des_target)
		p_ref, p_target, kp_pairs = convert_matches(kp_ref, kp_target, matches,enable_knn)
		#ext_points = [int(kp_pairs[0][0].pt[0]),int(kp_pairs[0][0].pt[1]),int(kp_pairs[0][1].pt[0]),int(kp_pairs[0][1].pt[1])]

	if output_as_array == True:
		#ext_points = np.zeros(shape=(len(kp_pairs),4))
		ext_points = np.zeros(4)
		#slope_array = np.zeros(len(kp_pairs))
		#distance_array = np.zeros(len(kp_pairs))
		if len(kp_pairs) !=0:
			ext_points = [int(kp_pairs[0][0].pt[0]),int(kp_pairs[0][0].pt[1]),int(kp_pairs[0][1].pt[0]),int(kp_pairs[0][1].pt[1])]
		print ext_points
		slope,distance = points2rect(ext_points,w_ref)
		#i=0
		'''
		for kpp in kp_pairs:
			ext_points[i][:] = [int(kpp[0].pt[0]),int(kpp[0].pt[1]),int(kpp[1].pt[0]),int(kpp[1].pt[1])]
			if enable_custom_ransac == True:
				deltax = np.float(ext_points[i][2]+w_ref-ext_points[i][0])
				deltay = np.float(ext_points[i][3]-ext_points[i][1])
				if deltax == 0 and deltay != 0:
					slope_array[i] = 90
				elif deltax != 0 and deltay == 0:
					slope_array[i] = 0
				else:
					slope_array[i] = (np.arctan(deltay/deltax)*360)/(2*np.pi)
				apoint = np.array((ext_points[i][0],ext_points[i][1]))
				bpoint = np.array((ext_points[i][2]+w_ref,ext_points[i][3]))
				distance_array[i] = np.sqrt(np.sum((apoint-bpoint)**2))
			print ext_points[i][:]
			print slope_array[i]
			print distance_array[i]
			i +=1
		'''
		#distance_array = np.sqrt(np.sum((ext_points[:][])**2))
		if enable_custom_ransac == True and len(slope_array) > 2 and len(distance_array) > 2:
			model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
			try:
				model_ransac.fit(slope_array.reshape(-1,1), distance_array.reshape(-1,1))
				inlier_mask = model_ransac.inlier_mask_
				print slope_array.reshape(-1,1)[inlier_mask]
				print distance_array.reshape(-1,1)[inlier_mask]
			except:
				print 'No inliers'
		

		return kp_ref,kp_target,ext_points,slope,distance
	else:
		return kp_ref,kp_target,matches
	'''
	if output_as_array == True and enable_ransac == False:
		ext_points = np.zeros(shape=(len(matches),4))
		i = 0
		for m in matches:
			ext_points[i][:]= [int(kp_ref[m.queryIdx].pt[0]),int(kp_ref[m.queryIdx].pt[1]),int(kp_target[m.trainIdx].pt[0]),int(kp_target[m.trainIdx].pt[1])]
			i = i+1
		return kp_ref,kp_target,ext_points
	else:
		return kp_ref,kp_target,matches
	'''


def filter_matches(kp1, kp2, matches, ratio = 0.75):
	mkp1, mkp2 = [], []
	for m in matches:
		if len(m) == 2 and m[0].distance < m[1].distance * ratio:
			m = m[0]
			mkp1.append( kp1[m.queryIdx] )
			mkp2.append( kp2[m.trainIdx] )
	p1 = np.float32([kp.pt for kp in mkp1])
	p2 = np.float32([kp.pt for kp in mkp2])
	kp_pairs = zip(mkp1, mkp2)
	return p1, p2, kp_pairs


def convert_matches(kp1,kp2,matches,enable_knn):
	mkp1, mkp2 = [], []
	for m in matches:
		if enable_knn == True:
			m = m[0]

		mkp1.append( kp1[m.queryIdx] )
		mkp2.append( kp2[m.trainIdx] )
	p1 = np.float32([kp.pt for kp in mkp1])
	p2 = np.float32([kp.pt for kp in mkp2])
	kp_pairs = zip(mkp1, mkp2)
	return p1, p2, kp_pairs


def otb_orthorectification(input_raster,output_raster):

	#Limitations: Supported sensors are Pleiades, SPOT5 (TIF format), Ikonos, Quickbird, Worldview2, GeoEye.
	#otbcli_OrthoRectification -io.in QB_TOULOUSE_MUL_Extract_500_500.tif -io.out QB_Toulouse_ortho.tif

	command = 'C:/OSGeo4W64/bin/otbcli_OrthoRectification -progress 1 -io.in {} -io.out {}'.format(input_raster,output_raster)
	proc = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stdin=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,).stdout
	print 'OTB Ortho-Rectification: ' + str(input_raster)
	for line in iter(proc.readline, ''):
		if '[*' in line:
			idx = line.find('[*')
			perc = int(line[idx - 4:idx - 2].strip(' '))
			if perc%10 == 0 and perc!=0:
				print str(perc) + '...',

	print '100'


def create_gdal_gcps(ext_points,geotransform_ref,geotransform_target):

	'''
	Function to convert the points extracted from SURF into GCPs compatible with the GDAL transform function

	:param ext_points: array with coordinates of extracted points
	:returns: an array of filtered points

	Author: Daniele De Vecchi
	Last modified: 10/09/2014
	'''
	gdal_gcp = []
	gdal_string = ''
	#ext points structure: x_ref,y_ref,x_target,y_target
	#gdal structure: pixel line easting northing
	gdal_gcp_list = []
	for p in range(0,len(ext_points)):
		north_target = geotransform_target[3] + ext_points[p][2]*geotransform_target[4] + ext_points[p][3]*geotransform_target[5]
		east_target = geotransform_target[0] + ext_points[p][2]*geotransform_target[1] + ext_points[p][3]*geotransform_target[2]
		#pixel_target,line_target =  world2pixel(geotransform_target_original, east_target, north_target)
		#pixel_target = east_target
		#line_target = north_target
		pixel_target = ext_points[p][2]
		line_target = ext_points[p][3]
		north_ref = geotransform_ref[3] + ext_points[p][0]*geotransform_ref[4] + ext_points[p][1]*geotransform_ref[5]
		east_ref = geotransform_ref[0] + ext_points[p][0]*geotransform_ref[1] + ext_points[p][1]*geotransform_ref[2]
		#gdal_gcp.append((pixel_target,line_target,east_ref,north_ref))
		gdal_gcp = osgeo.gdal.GCP(float(east_ref),float(north_ref),float(0.0),float(pixel_target),float(line_target))
		gdal_gcp_list.append(gdal_gcp) 
		gdal_string = gdal_string + '-gcp {} {} {} {} '.format(str(pixel_target),str(line_target),str(east_ref),str(north_ref))
		#print gdal_string
	return gdal_string,gdal_gcp_list


def tile_statistics(band_mat,start_col_coord,start_row_coord,end_col_coord,end_row_coord):

	'''
	Compute statistics related to the input tile

	:param band_mat: numpy 8 bit array containing the extracted tile
	:param start_col_coord: starting longitude coordinate
	:param start_row_coord: starting latitude coordinate
	:param end_col_coord: ending longitude coordinate
	:param end_row_coord: ending latitude coordinate

	:returns: a list of statistics (start_col_coord,start_row_coord,end_col_coord,end_row_coord,confidence, min frequency value, max frequency value, standard deviation value, distance among frequent values)

	Author: Daniele De Vecchi
	Last modified: 22/08/2014
	'''

	#Histogram definition
	data_flat = band_mat.flatten()
	data_counter = collections.Counter(data_flat)
	data_common = (data_counter.most_common(20)) #20 most common values
	data_common_sorted = sorted(data_common,key=itemgetter(0)) #reverse=True for inverse order
	hist_value = [elt for elt,count in data_common_sorted]
	hist_count = [count for elt,count in data_common_sorted]

	#Define the level of confidence according to the computed statistics
	min_value = hist_value[0]
	max_value = hist_value[-1]
	std_value = np.std(hist_count)
	diff_value = max_value - min_value
	min_value_count = hist_count[0]
	max_value_count = hist_count[-1]
	tot_count = np.sum(hist_count)
	min_value_freq = (float(min_value_count) / float(tot_count)) * 100
	max_value_freq = (float(max_value_count) / float(tot_count)) * 100

	if max_value_freq > 20.0 or min_value_freq > 20.0 or diff_value < 18 or std_value > 100000:
		confidence = 0
	elif max_value_freq > 5.0: #or std_value < 5.5: #or min_value_freq > 5.0:
		confidence = 0.5
	else:
		confidence = 1

	return (start_col_coord,start_row_coord,end_col_coord,end_row_coord,confidence,min_value_freq,max_value_freq,std_value,diff_value)


def window_output(window_name,ext_points,ref_band_mat,target_band_mat,output_file):

	'''
	Define a window to show the extracted points, useful to refine the method

	:param window_name: name of the output window
	:param ext_points: array with coordinates of extracted points
	:param ref_band_mat: numpy 8 bit array containing reference image
	:param target_band_mat: numpy 8 bit array containing target image
	:returns: a window is shown with the matching points

	'''
	#Height (rows) and width (cols) of the reference image
	h_ref, w_ref = ref_band_mat.shape[:2]
	#Height (rows) and width (cols) of the target image
	h_target, w_target = target_band_mat.shape[:2]

	vis = np.zeros((max(h_ref, h_target), w_ref+w_target), np.uint8)
	vis[:h_ref, :w_ref] = ref_band_mat
	vis[:h_target, w_ref:w_ref+w_target] = target_band_mat
	
	
	
	
	
	vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
	green = (0, 255, 0) #green
	print 'Total matching: ' + str(len(ext_points))
	'''
	for p in range(0,len(ext_points)):
		x_ref,y_ref,x_target,y_target = int(ext_points[p][0]),int(ext_points[p][1]),int(ext_points[p][2]),int(ext_points[p][3])
		#print x_ref,y_ref,x_target,y_target
		print x_ref-x_target,y_ref-y_target
		cv2.circle(vis, (x_ref, y_ref), 2, green, -1)
		cv2.circle(vis, (x_target+w_ref, y_target), 2, green, -1)
		cv2.line(vis, (x_ref, y_ref), (x_target+w_ref, y_target), green)
	'''
	x_ref,y_ref,x_target,y_target = int(ext_points[0]),int(ext_points[1]),int(ext_points[2]),int(ext_points[3])
	#print x_ref,y_ref,x_target,y_target
	print x_ref-x_target,y_ref-y_target
	cv2.circle(vis, (x_ref, y_ref), 2, green, -1)
	cv2.circle(vis, (x_target+w_ref, y_target), 2, green, -1)
	cv2.line(vis, (x_ref, y_ref), (x_target+w_ref, y_target), green)
	cv2.imwrite(output_file, vis)
	#vis0 = vis.copy()
	#cv2.imshow(window_name, vis)
	#cv2.waitKey()
	#cv2.destroyAllWindows()


def kp2matrix(kp_ref,kp_target,matches):

	'''
	Original filter from website example
	https://gist.github.com/moshekaplan/5106221

	:param matches: output of the matching operation
	:returns: list of extracted points

	'''

	#sel_matches = [m for m in matches if m[0].distance < m[1].distance * ratio]
	ext_points = np.zeros(shape=(len(matches),4))
	i = 0
	for m in matches:
		ext_points[i][:]= [int(kp_ref[m.queryIdx].pt[0]),int(kp_ref[m.queryIdx].pt[1]),int(kp_target[m.trainIdx].pt[0]),int(kp_target[m.trainIdx].pt[1])]
		i = i+1

	return ext_points

def FFT_coregistration(ref_band_mat,target_band_mat):

	'''
	Alternative method used to coregister the images based on the FFT

	:param ref_band_mat: numpy 8 bit array containing reference image
	:param target_band_mat: numpy 8 bit array containing target image
	:returns: the shift among the two input images 

	'''

	#Normalization - http://en.wikipedia.org/wiki/Cross-correlation#Normalized_cross-correlation 
	ref_band_mat = (ref_band_mat - ref_band_mat.mean()) / ref_band_mat.std()
	target_band_mat = (target_band_mat - target_band_mat.mean()) / target_band_mat.std() 

	#Check dimensions - they have to match
	rows_ref,cols_ref =  ref_band_mat.shape
	rows_target,cols_target = target_band_mat.shape



	if rows_target < rows_ref:
		print 'Rows - correction needed'

		diff = rows_ref - rows_target
		target_band_mat = np.vstack((target_band_mat,np.zeros((diff,cols_target))))
	elif rows_ref < rows_target:
		print 'Rows - correction needed'
		diff = rows_target - rows_ref
		ref_band_mat = np.vstack((ref_band_mat,np.zeros((diff,cols_ref))))
		
	rows_target,cols_target = target_band_mat.shape
	rows_ref,cols_ref = ref_band_mat.shape

	if cols_target < cols_ref:
		print 'Columns - correction needed'
		diff = cols_ref - cols_target
		target_band_mat = np.hstack((target_band_mat,np.zeros((rows_target,diff))))
	elif cols_ref < cols_target:
		print 'Columns - correction needed'
		diff = cols_target - cols_ref
		ref_band_mat = np.hstack((ref_band_mat,np.zeros((rows_ref,diff))))

	rows_target,cols_target = target_band_mat.shape   

	#translation(im_target,im_ref)
	freq_target = fft2(target_band_mat)   
	freq_ref = fft2(ref_band_mat)  
	inverse = abs(ifft2((freq_target * freq_ref.conjugate()) / (abs(freq_target) * abs(freq_ref))))   

	#Converts a flat index or array of flat indices into a tuple of coordinate arrays. would give the pixel of the max inverse value
	y_shift,x_shift = np.unravel_index(np.argmax(inverse),(rows_target,cols_target))

	if y_shift > rows_target // 2: # // used to truncate the division
		y_shift -= rows_target
	if x_shift > cols_target // 2: # // used to truncate the division
		x_shift -= cols_target
	
	return -x_shift, -y_shift


def points2rect(ext_points,width_reference): #width is used to compute distance in order to have values different from 0

	deltax = np.float(ext_points[2]+width_reference-ext_points[0])
	deltay = np.float(ext_points[3]-ext_points[1])
	if deltax == 0 and deltay != 0:
		#slope_array[i] = 90
		slope = 90
		distance = np.inf
	elif deltax != 0 and deltay == 0:
		#slope_array[i] = 0
		slope = 0
		distance = 0
	else:
		#slope_array[i] = (np.arctan(deltay/deltax)*360)/(2*np.pi)
		slope = (np.arctan(deltay/deltax)*360)/(2*np.pi)
		apoint = np.array((ext_points[0],ext_points[1]))
		bpoint = np.array((ext_points[2]+width_reference,ext_points[3]))
		distance = np.sqrt(np.sum((apoint-bpoint)**2))
	return slope,distance


def ransac_filter(slope_list,distance_list):

	model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
	try:
		model_ransac.fit(np.array(slope_list).reshape(-1,1), np.array(distance_list).reshape(-1,1))
		inlier_mask = model_ransac.inlier_mask_
		slope_list_inliers = np.array(slope_list).reshape(-1,1)[inlier_mask]
		distance_list_inliers = np.array(distance_list).reshape(-1,1)[inlier_mask]
		print np.array(slope_list).reshape(-1,1)[inlier_mask]
		print np.array(distance_list).reshape(-1,1)[inlier_mask]
	except:
		print 'No inliers'

	return slope_list_inliers,distance_list_inliers,inlier_mask