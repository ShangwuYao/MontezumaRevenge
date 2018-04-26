from PIL import Image
import cv2
import numpy as np
from scipy.misc import imsave

im = Image.open("images/img4.png")
rgb_im = np.array(im)
#rgb_im = im.convert('RGB')
rgb_im = cv2.resize(rgb_im, (1024, 1024), interpolation = cv2.INTER_CUBIC)
#rgb_im.save('images/img4.jpg')

imsave('images/img4.jpg', rgb_im)
