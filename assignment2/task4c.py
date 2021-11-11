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
   
    kernel = np.ones(im.shape)
    measured_y = 269
    measured_x_center = 232
    measured_x_max = 261
    #the values on the end are changed to test the best solution we could find
    diff_x = measured_x_max-measured_x_center -4 
    spike = 5 
    #setting the zero value for the kernel

    kernel[measured_y-spike:measured_y + spike] = 0
    kernel[measured_y-spike:measured_y+spike,measured_x_center-diff_x:measured_x_center+diff_x] = 1
    #kernel[start: im_centery+start, im_centerx-end: im_centerx+end] = 1


    plt.imshow(np.log(np.abs(kernel)+1), cmap='gray')
    plt.savefig(utils.image_output_dir.joinpath("4c_test2.png"))
    plt.clf()
    #shifting the kernel

    im_fft = np.fft.fft2(im)
    kernel_shift = np.fft.fftshift(kernel)
    im_conv = im_fft*kernel_shift

    im_filtered = np.fft.ifft2(im_conv).real


    plt.imshow(im_filtered, cmap="gray")
    plt.show()
    plt.savefig(utils.image_output_dir.joinpath("4c_test.png"))

#     
    ### END YOUR CODE HERE ###
    utils.save_im("moon_filtered1.png", utils.normalize(im_filtered))
