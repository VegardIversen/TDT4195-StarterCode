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
#will be from around 0-30

if __name__ == "__main__":
    # DO NOT CHANGE
    impath = os.path.join("images", "noisy_moon.png")
    im = utils.read_im(impath)

    # START YOUR CODE HERE ### (You can change anything inside this block)

    #im_centerx, im_centery = im.shape//2
    start_rm = 4
    end_rm = 28
    start = 207
    end = 255
    im_centerx = im.shape[1]//2
    im_centery = im.shape[0]//2
    kernel = np.ones(im.shape)

    #setting the zero value for the kernel

#     kernel[im_centery-start_rm //2: im_centery+start_rm //2] = 0
#     kernel[im_centery-start_rm//2: im_centery+start_rm //2, im_centerx-end_rm: im_centerx+end_rm] = 1

    kernel[im_centery-1:im_centery + 1] = 0
    #kernel[start: im_centery+start, im_centerx-end: im_centerx+end] = 1


    plt.imshow(np.log(np.abs(kernel)+1), cmap='gray')
    plt.savefig(utils.image_output_dir.joinpath("4c_test2.png"))
    plt.clf()
    #shifting the kernel

    #kernel = np.fft.fftshift(kernel)

    im_fft = np.fft.fft2(im)

    im_conv = im_fft*kernel

    im_filtered = np.fft.ifft2(im_conv).real


    plt.imshow(im_filtered, cmap="gray")
    plt.show()
    plt.savefig(utils.image_output_dir.joinpath("4c_test.png"))

#     im_centerx = im.shape[0]//2
#     im_centery = im.shape[1]//2
#     im_filtered = im
#     im_filtered = np.fft.fft2(im_filtered)


#     kernel = np.ones(im.shape)
#     spike_height = 4  # Lines are barely visible at < 4
#     image_center_x = im.shape[1] // 2
#     image_center_y = im.shape[0] // 2
#     center_width = 28  # Lines are visible for > 28

#     kernel[image_center_y - spike_height //
#            2:image_center_y + spike_height // 2] = 0
#     kernel[image_center_y - spike_height // 2:image_center_y + spike_height //
#            2, image_center_x - center_width:image_center_x + center_width] = 1
#     kernel[0:5,0:5] = 0
#     print(kernel)
#     plt.imshow(kernel, cmap="gray")
#     plt.show()
#     kernel = np.fft.fftshift(kernel)

#     fft_im = np.fft.fft2(im)

    # Note that this is not matrix multiplication, only point-vise multiplication
    # For matrix multiplication, use np.matmul(..)
#     fft_im_filtered = fft_im * kernel
#     filt_im = np.fft.ifft2(fft_im_filtered).real
#     plt.imshow(filt_im, cmap="gray")
#     plt.show()
    # rows, cols = im_filtered.shape
    
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
