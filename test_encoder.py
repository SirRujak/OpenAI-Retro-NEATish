import tensorflow as tf
import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


decoder_model = tf.keras.models.load_model('models/decoder_model.h5')
input_img = np.asarray(Image.open('frames/' + '00f81c9e1f1e3d7c' + '.bmp')).astype(np.float32) / 255.0
decoded_img = decoder_model.predict(np.expand_dims(input_img, axis=0))[0]

img_height = 224
img_width = 320
double_height = 2 * img_height
encoder_images = np.zeros((double_height, img_width, 3))
fig,ax = plt.subplots(1,1)
encoder_images[:img_height,:,:] = input_img
encoder_images[img_height:,:,:] = decoded_img
image = ax.imshow(encoder_images[:,:,:])
plt.show()
