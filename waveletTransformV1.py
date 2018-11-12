import numpy as np
import matplotlib.pyplot as plt

import pywt
import pywt.data
import sys


import osgeo.gdal
import osgeo.ogr
from gdalconst import *

import scipy
from scipy import ndimage, signal



"""

# Load image
original = pywt.data.camera()
#print original
print type(original)
print original.shape


# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(original, 'bior1.3')
LL, (LH, HL, HH) = coeffs2

print "-----------888---------------"
print type(coeffs2)
#print coeffs2.shape
print "-----------888---------------"


print LL
print type(LL)
print LL.shape
print LH.shape
print HL.shape
print HH.shape
#sys.exit()


fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()

"""

# Lectura de bandas y datos de la imagen entrante. (Master - Slave)
def read_bands(input_reference):

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


    return band_list

def resampling_2dArray(a):

    # Scale factor
    factor = 4

    # Input image
    #a = nup.arange(16).reshape((4,4))

    #print a.shape

    # Empty image enlarged by scale factor
    b = np.zeros((a.shape[0]*factor, a.shape[1]*factor))


    #print b.shape
    #sys.exit()

    # Fill the new array with the original values
    b[::factor,::factor] = a


    # Define the convolution kernel
    kernel_1d = scipy.signal.boxcar(factor)
    kernel_2d = np.outer(kernel_1d, kernel_1d)

    # Apply the kernel by convolution, seperately in each axis
    c = scipy.signal.convolve(b, kernel_2d, mode="valid")


    return c




def waveletTransform_master_slave(band_mat_master_original, band_mat_slave):


    print "* Dimension ms original"
    print band_mat_master_original.shape

    band_mat_master_original = resampling_2dArray(band_mat_master_original)
    #band_mat_master_resam = resampling_2dArray(band_mat_master_original)

    print "* Dimension ms original sobresampleada"
    print band_mat_master_original.shape



    #b = band_mat_master1.repeat(4, axis=0)
    #band_mat_master = b.repeat(4, axis=1)

    print "-------------------------------------------"



    #print "* Dimension ms resampling"
    #print band_mat_master_resam.shape

    print "* Dimension sl original"
    print band_mat_slave.shape

    # Elimina 3 filas
    #band_mat_slave = np.delete(band_mat_slave, range(-3,0), axis=0)
    band_mat_slave = np.delete(band_mat_slave, -1, axis=0)
    band_mat_slave = np.delete(band_mat_slave, -1, axis=0)
    band_mat_slave = np.delete(band_mat_slave, -1, axis=0)

    # Elimina 4 columnas
    #band_mat_slave = np.delete(band_mat_slave, range(-4,0), axis=1)
    band_mat_slave = np.delete(band_mat_slave, -1, axis=1)
    band_mat_slave = np.delete(band_mat_slave, -1, axis=1)
    band_mat_slave = np.delete(band_mat_slave, -1, axis=1)
    band_mat_slave = np.delete(band_mat_slave, -1, axis=1)

    print "* Dimension eliminando 4 filas y col de sl"
    print band_mat_slave.shape


    #fig = plt.figure()
    #fig.suptitle("Imagen original", fontsize=14)
    #plt.imshow(band_mat_master_original, interpolation="nearest", cmap=plt.cm.gray)

    #fig = plt.figure()
    #fig.suptitle("Imagen original resampleada", fontsize=14)
    #plt.imshow(band_mat_master_resam, interpolation="nearest", cmap=plt.cm.gray)

    #plt.show()


    #sys.exit()



    # Wavelet transform of image, and plot approximation and details
    titles = ['Approximation ms', ' Horizontal detail ms',
              'Vertical detail ms', 'Diagonal detail ms']
    coeffs2_master = pywt.dwt2(band_mat_master_original, 'bior1.3')
    LL_ms, (LH_ms, HL_ms, HH_ms) = coeffs2_master




    # Wavelet transform of image, and plot approximation and details
    titles2 = ['Approximation sl', ' Horizontal detail sl',
              'Vertical detail sl', 'Diagonal detail sl']
    coeffs2_slave = pywt.dwt2(band_mat_slave, 'bior1.3')
    LL_slv, (LH_slv, HL_slv, HH_slv) = coeffs2_slave





    """
    print "-----------888---------------"
    print type(coeffs2)
    print len(coeffs2)
    print "-----------888---------------"

    print type(coeffs2[0])
    print coeffs2[0].shape
    #print coeffs2[0]

    print type(coeffs2[1])
    print len(coeffs2[1])
    #print coeffs2[1]

    print type(coeffs2[1][0])
    print coeffs2[1][0].shape
    print coeffs2[1][0]

    print type(coeffs2[1][1])
    print coeffs2[1][1].shape
    print type(coeffs2[1][2])
    print coeffs2[1][1].shape


    sys.exit()



# Ploteo de todas las imagenes ///////////////////////////////////////////////

    # Now reconstruct and plot the original image
    reconstructed = pywt.idwt2(coeffs2, 'bior1.3')
    fig = plt.figure()
    fig.suptitle("Imagen reconstruida - dwt2", fontsize=14)
    plt.imshow(reconstructed, interpolation="nearest", cmap=plt.cm.gray)


   # Ploteo de cada imagen: Original, Recostruida, H, V y D.
    fig = plt.figure()
    fig.suptitle("Imagen original", fontsize=14)
    plt.imshow(original2, interpolation="nearest", cmap=plt.cm.gray)

    fig = plt.figure()
    fig.suptitle("Horizontal", fontsize=14)
    plt.imshow(LH, interpolation="nearest", cmap=plt.cm.gray)

    fig = plt.figure()
    fig.suptitle("Vertical", fontsize=14)
    plt.imshow(HL, interpolation="nearest", cmap=plt.cm.gray)

    fig = plt.figure()
    fig.suptitle("Diagonal", fontsize=14)
    plt.imshow(HH, interpolation="nearest", cmap=plt.cm.gray)

    plt.show()

    sys.exit()

#///////////////////////////////////////////////

    print LL_ms
    print type(LL_ms)
    print LL_ms.shape
    print LH_ms.shape
    print HL_ms.shape
    print HH_ms.shape



    fig = plt.figure(figsize=(12, 3))
    for i, a in enumerate([LL_ms, LH_ms, HL_ms, HH_ms]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()


    fig = plt.figure(figsize=(12, 3))
    for i, a in enumerate([LL_slv, LH_slv, HL_slv, HH_slv]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles2[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()

    plt.show()

    sys.exit()

"""

    coeffs2_fusion = LL_ms, (LH_slv, HL_slv, HH_slv)

    #print LL_ms.shape
    #print LH_slv.shape
    #print HL_slv.shape
    #print HH_slv.shape
    #sys.exit()

    print "Comienza la fusion..."

    # Now reconstruct and plot the original image
    reconstructed = pywt.idwt2(coeffs2_fusion, 'bior1.3')
    fig = plt.figure()
    fig.suptitle("Imagen fusionada", fontsize=14)
    plt.imshow(reconstructed, interpolation="nearest", cmap=plt.cm.gray)

    plt.show()

    """

    fig = plt.figure()
    fig.suptitle("Imagen original", fontsize=14)
    plt.imshow(band_mat_master_resam, interpolation="nearest", cmap=plt.cm.gray)



    fig = plt.figure()
    fig.suptitle("Imagen pan", fontsize=14)
    plt.imshow(band_mat_slave, interpolation="nearest", cmap=plt.cm.gray)



    plt.show()




    # Now reconstruct and plot the original image
    reconstructed = pywt.idwt2(coeffs2_fusion, 'bior1.3')
    fig = plt.figure()
    fig.suptitle("Imagen fusionada", fontsize=14)
    plt.imshow(reconstructed, interpolation="nearest", cmap=plt.cm.gray)
    plt.show()
"""

    #idwt