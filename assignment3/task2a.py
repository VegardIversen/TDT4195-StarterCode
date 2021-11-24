import numpy as np
import skimage
import utils
import pathlib


def otsu_thresholding(im: np.ndarray) -> int:
    """
        Otsu's thresholding algorithm that segments an image into 1 or 0 (True or False)
        The function takes in a grayscale image and outputs a boolean image

        args:
            im: np.ndarray of shape (H, W) in the range [0, 255] (dtype=np.uint8)
        return:
            (int) the computed thresholding value
    """
    assert im.dtype == np.uint8
    # START YOUR CODE HERE ### (You can change anything inside this block)
    # You can also define other helper functions
    # Compute normalized histogram
    int_range = im.max()-im.min()+1
    hist, bins = np.histogram(im, bins=int_range)
    #normalizing histogram
    num_pixel = im.shape[0]*im.shape[1] #size of the image
    hist_norm = hist/num_pixel

    cum_sums= np.cumsum(hist_norm)
    #calculating the means
    cum_means = np.zeros(int_range)

    for i in range(1,len(hist_norm)):
        cum_means[i] = cum_means[i-1] + i*hist_norm[i]  #this will make both m_G and m(k)

    m_G = cum_means[-1] #global mean

    theta_b_sq = (m_G*cum_sums - cum_means)**2 / (cum_sums*(1-cum_sums))
    #finding the maximum values between classes
    max_val = np.max(theta_b_sq)
    ks = []
    for k, val  in enumerate(theta_b_sq):
        if(val ==max_val):
            ks.append(k)
    threshold = np.sum(ks)/len(ks)

    #need to add the im.min()
    threshold += im.min()
    #threshold = 128 #a bit unsure if i should have used this as k earlier 
    return threshold
    ### END YOUR CODE HERE ###


if __name__ == "__main__":
    # DO NOT CHANGE
    impaths_to_segment = [
        pathlib.Path("thumbprint.png"),
        pathlib.Path("polymercell.png")
    ]
    for impath in impaths_to_segment:
        im = utils.read_image(impath)
        threshold = otsu_thresholding(im)
        print("Found optimal threshold:", threshold)

        # Segment the image by threshold
        segmented_image = (im >= threshold)
        assert im.shape == segmented_image.shape, "Expected image shape ({}) to be same as thresholded image shape ({})".format(
            im.shape, segmented_image.shape)
        assert segmented_image.dtype == np.bool, "Expected thresholded image dtype to be np.bool. Was: {}".format(
            segmented_image.dtype)

        segmented_image = utils.to_uint8(segmented_image)

        save_path = "{}-segmented.png".format(impath.stem)
        utils.save_im(save_path, segmented_image)
