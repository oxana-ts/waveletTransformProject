import numpy
import scipy
from scipy import ndimage, signal
import sys



# Scale factor
factor = 2

# Input image
a = numpy.arange(16).reshape((4,4))


# Empty image enlarged by scale factor
b = numpy.zeros((a.shape[0]*factor, a.shape[0]*factor))

print a
print b



# Fill the new array with the original values
b[::factor,::factor] = a

print a
print b



# Define the convolution kernel
kernel_1d = scipy.signal.boxcar(factor)
kernel_2d = numpy.outer(kernel_1d, kernel_1d)

# Apply the kernel by convolution, seperately in each axis
c = scipy.signal.convolve(b, kernel_2d, mode="valid")

print c