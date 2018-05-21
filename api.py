from sklearn.cluster import KMeans
from skimage.io import imsave
from PIL import Image
from io import BytesIO
import numpy as np

def posterize_numpy(image_data, num_colors = 16):
    image_shape = image_data.shape
    estimator = KMeans(n_clusters=num_colors, init='k-means++', n_init=1)

    pixel_labels = estimator.fit_predict(image_data.reshape(-1,3))
    pixel_data = [list(map(int,estimator.cluster_centers_[c])) for c in pixel_labels]
    pixel_data = np.array(pixel_data).reshape(*image_shape)
    return numpy_to_img(pixel_data)

def numpy_to_img(image_data):
    strIO = BytesIO()
    imsave(strIO, image_data, plugin='pil', format_str='png')
    strIO.seek(0)
    return strIO
