import skimage
import skimage.io
import skimage.transform
import os
import numpy as np
import utils
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # DO NOT CHANGE
    impath = os.path.join("images", "noisy_moon.png")
    im = utils.read_im(impath)

    # START YOUR CODE HERE ### (You can change anything inside this block)

    im_filtered = im
    im_filtered = np.fft.fft2(im_filtered)
    im_filtered = np.fft.fftshift(im_filtered)
    im_filtered = np.abs(im_filtered)
    
    print(np.argmax(im_filtered))
    plt.imshow(im_filtered)
    ### END YOUR CODE HERE ###
    utils.save_im("moon_filtered.png", utils.normalize(im_filtered))
