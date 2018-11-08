from library import *
from waveletTransformV1 import *
import numpy as np
#np.set_printoptions(threshold='nan')
import os
import time
import osgeo.gdal
import osgeo.ogr
from gdalconst import *
from operator import itemgetter, attrgetter
import subprocess
import shutil
from matplotlib import pyplot as plt
import csv
import pdb
import sys





#input_reference = "/home/pablo/Escritorio/DATOS/Opticos/Quickbird/puertoPrincipe.tif"


#Multiespectral cett
#input_reference = "/home/pablo/Escritorio/DATOS/Opticos/Spot 6/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_MS_001_A/IMG_SPOT6_MS_201803151357431_ORT_A1520000112510_R1C1.JP2"
input_reference_master  = "/home/pablo/Escritorio/DATOS/ROI/CarlosPaz_SPOT_ms2.tif"


#Pancromatica cett
#input_reference = "/home/pablo/Escritorio/DATOS/Opticos/Spot 6/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_P_001_A/IMG_SPOT6_P_201803151357431_ORT_A1520000112510_R1C1.JP2"
input_reference_slave  = "/home/pablo/Escritorio/DATOS/ROI/CarlosPaz_SPOT_pn.tif"




#input_reference = "/home/pablo/Escritorio/DATOS/sar/envisat_wsm_12sep2005_hh_pot_cal_cg_gk5wgs84_nn.tif"

#input_target    = "/home/ubuntumate/Escritorio/PortAuPrince/PortAuPrince/post_rs.tif"



#//////////////////////////////////////////////////////////////////////////
"""
print input_reference

rows_ref,cols_ref,nbands_ref,geo_transform_ref,projection_ref = read_image_parameters(input_reference)

print "Informacion del archivo seleccionado-----------------------"

print "Cant. de filas: ", rows_ref
print "Cant. de columnas: ", cols_ref
print "Cant. de bandas: ", nbands_ref
print "geo_transform_ref: ", geo_transform_ref
print "projection_ref: ", projection_ref

print "Fin-----------------------"

band_list= []

inputimg = osgeo.gdal.Open(input_reference, GA_ReadOnly)
cols = inputimg.RasterXSize
rows = inputimg.RasterYSize
nbands = inputimg.RasterCount



for i in range(1, nbands + 1):
    inband = inputimg.GetRasterBand(i)
    # mat_data = inband.ReadAsArray(0,0,cols,rows).astype(data_type)
    mat_data = inband.ReadAsArray()
    #mat_data = inband.ReadAsArray().astype(data_type)
    band_list.append(mat_data)
"""
#//////////////////////////////////////////////////////////////////////////

"""
#print band_list[0]
print type(band_list[0])
#np.set_printoptions(threshold='nan')
print band_list[0].shape
"""



# Lectura de datos de la imagen maestra.
band_list_master = read_bands(input_reference_master)

# Lectura de datos de la imagen esclava.
band_list_slave = read_bands(input_reference_slave)


print "Comenzo el procesamiento de la transformada wavelet -------------/////////////////////////////"
waveletTransform_master_slave(band_list_master[0],band_list_slave[0])

print "Finalizo el procesamiento de la transformada wavelet -------------/////////////////////////////"


# probar con la imagen del puerto principe!!!