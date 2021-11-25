import utils
import numpy as np

#https://en.wikipedia.org/wiki/Moore_neighborhood



        

def region_growing(im: np.ndarray, seed_points: list, T: int) -> np.ndarray:
    """
        A region growing algorithm that segments an image into 1 or 0 (True or False).
        Finds candidate pixels with a Moore-neighborhood (8-connectedness). 
        Uses pixel intensity thresholding with the threshold T as the homogeneity criteria.
        The function takes in a grayscale image and outputs a boolean image

        args:
            im: np.ndarray of shape (H, W) in the range [0, 255] (dtype=np.uint8)
            seed_points: list of list containing seed points (row, col). Ex:
                [[row1, col1], [row2, col2], ...]
            T: integer value defining the threshold to used for the homogeneity criteria.
        return:
            (np.ndarray) of shape (H, W). dtype=np.bool

        Task:
            8 connectedness
             _ _ _
            |NW|N|NE|
            |W |C| E|
            |SW|S|SE|
    """
    # START YOUR CODE HERE ### (You can change anything inside this block)
    # You can also define other helper functions
    segmented = np.zeros_like(im).astype(bool)
    im = im.astype(float)
    

    for row, col in seed_points:
        #did it with recursion to since it seems to be a good recursion task, but got error when i changed to threshold to 90 because of to many recursion calls, could prob change some settings for this
        seed_point = im[row,col] #storing the seed_point value
        #list for pixels that are connected and above the threshold. the first one might not be over but will be pop and not the segmented will then not be true
        connected_pixel = [[row,col]]

        while  (len(connected_pixel) > 0): #as long as its not empty
            r, c = connected_pixel.pop()
            #going through the the 8 positions
            for i in range(r-1,r+2):
                for j in range(c-1,c+2):
                    #boundries
                    if(i == r and j == c): #no point of checking the middle value. Unsure if having a check is more worth than just running every pixel
                        continue
                    if (i >= 0 and j >= 0 and i < im.shape[0] and j < im.shape[1]):
                        if (not segmented[i, j] and abs(im[i, j]-seed_point) < T):
                            connected_pixel.append([i,j])
                            segmented[i,j] = True


    return segmented
    ### END YOUR CODE HERE ###


if __name__ == "__main__":
    # DO NOT CHANGE
    im = utils.read_image("defective-weld.png")

    seed_points = [  # (row, column)
        [254, 138],  # Seed point 1
        [253, 296],  # Seed point 2
        [233, 436],  # Seed point 3
        [232, 417],  # Seed point 4
    ]
    intensity_threshold = 50
    segmented_image = region_growing(im, seed_points, intensity_threshold)

    assert im.shape == segmented_image.shape, "Expected image shape ({}) to be same as thresholded image shape ({})".format(
        im.shape, segmented_image.shape)
    assert segmented_image.dtype == np.bool, "Expected thresholded image dtype to be np.bool. Was: {}".format(
        segmented_image.dtype)

    segmented_image = utils.to_uint8(segmented_image)
    utils.save_im("defective-weld-segmented4.png", segmented_image)
