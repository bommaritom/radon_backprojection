import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
import io
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import skimage as sk
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale
from numpy import matrix




# create shepp logan phantom image

image = shepp_logan_phantom()
image = rescale(image, scale=0.4, mode='reflect')

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

# initialize some constants

N = np.shape(sinogram)[0]

# generate radon projections

P = np.transpose(sinogram)

# begin backprojection algorithm

S = np.fft.fft(P)

# filter the slices

S_filtered = np.copy(S)
S_filtered[:,0] = S[:,0] * 0/N
S_filtered[:,N//2] = S[:,N//2] * 1/2 # == (N//2)/N
for m in range(1, N//2):
	S_filtered[:,m] = S[:,m] * m/N
	S_filtered[:,N-m] = S[:,N-m] * m/N

# inverse fourier transform of each slice

Q = np.fft.ifft(S_filtered)

# backproject each slice

f = np.zeros((N,N))

for k in range(len(Q)):
	theta = k/NUM_ANGLES * np.pi
	for i in range(N):
		for j in range(N):
			val = (j-(N/2))*np.cos(theta)+(i-(N/2))*np.sin(theta) + N/2 + 1
			if (val >= 0 and val < len(Q[k])):
				f[N-1-i][j] += Q[k][int(val)]
	
	print("  " + str(np.round(100*(k+1)/NUM_ANGLES, 1))+ "% complete", end='\r')
	
f *= np.pi/NUM_ANGLES

# display everything

#fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4.5))
#
#ax1.set_title("Original")
#ax1.imshow(image, cmap=plt.cm.Greys_r)
#
#dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / sinogram.shape[0]
#
#ax2.set_title("Radon transform\n(Sinogram)")
#ax2.set_xlabel("Projection angle (deg)")
#ax2.set_ylabel("Projection position (pixels)")
#ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
#           extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
#           aspect='auto')
#
#ax3.set_title("Reconstructed from " + str(NUM_ANGLES) + " angles.")
#ax3.imshow(f, cmap=plt.cm.Greys_r)
#
#fig.tight_layout()
#
#plt.show()
#


error = f - image
print(f'FBP rms reconstruction error: {np.sqrt(np.mean(error**2)):.3g}')

imkwargs = dict(vmin=-0.2, vmax=0.2)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4.5),
                               sharex=True, sharey=True)
                               
min = 0
max = 1
                     
                     
ax1.set_title("Original")
ax1.imshow(image, cmap=plt.cm.Greys_r, vmin = min, vmax = max)
#ax1.imshow(image, cmap=plt.cm.Greys_r)


ax2.set_title("Reconstruction\nFiltered backprojection\n"+str(NUM_ANGLES)+" angles")
im2 = ax2.imshow(f, cmap=plt.cm.Greys_r, vmin = min, vmax = max)
#im2 = ax2.imshow(f, cmap=plt.cm.Greys_r)


ax3.set_title("1 minus absolute\nreconstruction error\nFiltered back projection")
#im3 = ax3.imshow(f - image, cmap=plt.cm.Greys_r, **imkwargs)
ax3.imshow(max-np.abs(f-image), cmap = plt.cm.Greys_r, vmin = min, vmax = max)

fig.subplots_adjust(right=0.85)

cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
fig.colorbar(im2, cax = cbar_ax)
plt.show()
