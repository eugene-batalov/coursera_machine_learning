import numpy as np
from skimage.io import imread
from skimage import img_as_float
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from time import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


image = imread('parrots.jpg')
fimage = img_as_float(image)

w, h, d = original_shape = tuple(fimage.shape)
assert d == 3
image_array = np.reshape(fimage, (w * h, d))

image_array_sample = shuffle(image_array, random_state=0)[:1000]

def count_mse(clusters):
    print("clusters: %d" % clusters)
    t0 = time()
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(image_array_sample)
    labels = kmeans.predict(image_array)
    r_image = recreate_image(kmeans.cluster_centers_, labels, w, h)
    r_image_array = np.reshape(r_image, (w * h, d))
    print("done in %0.3fs." % (time() - t0))
    return mean_squared_error(image_array,r_image_array)#((image_array - r_image_array)**2).mean(axis=None)

mse = list(map(count_mse, range(1,21)))
psnr = 20.0 * np.log10(1.0) - 10.0 * np.log10(mse)

print('psnr = ', psnr, '\n min clusters for psnr > 10: ', 1 + np.argmax(psnr > 20.0))

# Display all results, alongside original image
"""
plt.figure(1)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Original image (96,615 colors)')
plt.imshow(image)

plt.figure(2)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image ('+str(n_colors)+' colors, K-Means)')
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))
"""