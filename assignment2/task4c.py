import skimage
import skimage.io
import skimage.transform
import os
import numpy as np
import utils
import matplotlib.pyplot as plt


#x1 = 203.017 , y=269.068
#x2=232.019, y=268.907
#x3=261.021 y=268.907

if __name__ == "__main__":
    # DO NOT CHANGE
    impath = os.path.join("images", "noisy_moon.png")
    im = utils.read_im(impath)

    # START YOUR CODE HERE ### (You can change anything inside this block)
    im_centerx = im.shape[0]//2
    im_centery = im.shape[1]//2
    im_filtered = im
    im_filtered = np.fft.fft2(im_filtered)


    kernel = np.ones(im.shape)
    spike_height = 4  # Lines are barely visible at < 4
    image_center_x = im.shape[1] // 2
    image_center_y = im.shape[0] // 2
    center_width = 28  # Lines are visible for > 28

    kernel[image_center_y - spike_height //
           2:image_center_y + spike_height // 2] = 0
    kernel[image_center_y - spike_height // 2:image_center_y + spike_height //
           2, image_center_x - center_width:image_center_x + center_width] = 1
    kernel[0:5,0:5] = 0
    print(kernel)
    plt.imshow(kernel, cmap="gray")
    plt.show()
    kernel = np.fft.fftshift(kernel)

    fft_im = np.fft.fft2(im)

    # Note that this is not matrix multiplication, only point-vise multiplication
    # For matrix multiplication, use np.matmul(..)
    fft_im_filtered = fft_im * kernel
    filt_im = np.fft.ifft2(fft_im_filtered).real
    plt.imshow(filt_im, cmap="gray")
    plt.show()
    # rows, cols = im_filtered.shape
    # fft_im = im_filtered.copy()
    # fft_im[268:269,240:250]=0
    # fft_im = np.fft.fftshift(fft_im)
    # plt.imshow(np.log(np.abs(fft_im)+1))
    # plt.show()
    # im_filtered = np.fft.ifft2(fft_im).real
    # plt.imshow(im_filtered)
    # plt.show()
    #im_filtered = np.fft.fftshift(im_filtered)
    #im_filtered = np.abs(im_filtered)
    #print(im_filtered.shape)
    
    #plt.imshow(im_filtered) #used to find the spikes.
    #plt.show()
    ### END YOUR CODE HERE ###
    #utils.save_im("moon_filtered1.png", utils.normalize(im_filtered))
