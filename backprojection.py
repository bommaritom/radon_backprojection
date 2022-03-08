import matplotlib.pyplot as plt
import numpy as np
from numpy import arange, matrix
import io
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import skimage as sk
from skimage.data import shepp_logan_phantom, checkerboard, camera
from skimage.transform import radon, rescale

# select grayscale image

image = shepp_logan_phantom()

# mask the image so that it is a circle (code from skimage)

l_x, l_y = image.shape[0], image.shape[1]
X, Y = np.ogrid[:l_x, :l_y]
outer_disk_mask = (X - l_x / 2)**2 + (Y - l_y / 2)**2 > (l_x / 2)**2 - 1
image[outer_disk_mask] = 0

# rescale image

image = rescale(image, scale=.4, mode='reflect')

# create radon transform

NUM_ANGLES = 100
angles = np.linspace(0., 180., NUM_ANGLES, endpoint=False)
sinogram = radon(image, theta=angles)

dx, dy = 0.5 * 180.0 / NUM_ANGLES, 0.5 / sinogram.shape[0]

plt.title("Radon transform\n(Sinogram)")
plt.xlabel("Projection angle (deg)")
plt.ylabel("Projection position (pixels)")
plt.imshow(sinogram, cmap=plt.cm.Greys_r,
           extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
           aspect='auto')
plt.show()

# initialize constants

N = np.shape(sinogram)[0]

# generate radon projections

P = np.transpose(sinogram)

# begin backprojection algorithm

S = np.fft.fft(P)

# filter based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4341983/

x_vals = np.linspace(-N//2,N//2,N)
ramp = .5 * (np.sinc(x_vals)) - .25 * np.square(np.sinc(x_vals/2))
RAMP = np.fft.fft(np.fft.fftshift(ramp))

S_filtered = S * RAMP

# inverse fourier transform of each slice

Q = np.fft.ifft(S_filtered)

# backproject each slice

f = np.zeros((N,N))

# this could be rewritten without the loop, but it would be hard!!!
for k in range(len(Q)):

	theta = k/NUM_ANGLES * np.pi

	# get parameterization

	row_ind = np.arange(N)[::-1,None]
	col_ind = np.arange(N)

	t = (col_ind - (N/2))*np.cos(theta) + (row_ind - N/2)*np.sin(theta) + N/2

	# perform linear interpolation

	fl = np.minimum(np.floor(t).astype(int), len(Q[k])-1)
	ce = np.minimum(np.ceil(t).astype(int), len(Q[k])-1)
	up = np.take(Q[k],ce).real
	lo = np.take(Q[k],fl).real

	interp = lo + (t-fl)*(up-lo)

	# add backprojection to the reconstructed image

	in_bounds = np.logical_and(t >= 0, t < len(Q[k]))

	f += np.where(in_bounds, interp, np.zeros(np.shape(f)))

	print("  " + str(np.round(100*(k+1)/NUM_ANGLES, 1))+ "% complete", end='\r')

f *= np.pi/NUM_ANGLES

# display everything

error = f - image
print(f'FBP normalized rms reconstruction error: {np.sqrt(np.mean(error**2))/np.sqrt(np.mean(image**2)):.3g}')

imkwargs = dict(vmin=-0.2, vmax=0.2)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4.5),
                               sharex=True, sharey=True)
                               
min = 0
max = 1
                     
ax1.set_title("Original")
im1 = ax1.imshow(image, cmap=plt.cm.Greys_r, vmin = min, vmax = max)

ax2.set_title("Reconstruction\nFiltered backprojection\n"+str(NUM_ANGLES)+" angles")
im2 = ax2.imshow(f, cmap=plt.cm.Greys_r, vmin = min, vmax = max)

ax3.set_title("Absolute\nreconstruction error\nFiltered back projection")
im3 = ax3.imshow(np.abs(f-image), cmap = plt.cm.Greys_r, vmin = min, vmax = max)

fig.subplots_adjust(right=0.85)

cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
fig.colorbar(im1, cax = cbar_ax)
plt.show()
