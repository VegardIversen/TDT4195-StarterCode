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
    def neighboorhood(seed_point,row,col): #having the function here so i can use the image and T without having it as parameters 
        #iterate over rows and cols
        #N = 3 #NxN matrise, 
        #starting from -1 so that 0 is the center
        for i in range(-1,2):
            for j in range(-1,2):
                #skipping center point
                if (i == 0 and j == 0):
                    continue
                #handling for boundries 
                if(row + i < 0 or row + i >= im.shape[0] or col + j < 0 or col + j >= im.shape[1]):
                    continue

                #if the cell we are visiting are not visited[False] and the value at this point is larger than the threshold
                # will we set that cell to True and visit that cell
                #problem here since the lowest i and j value is 0 so we cant check the previous NW etc

                if(not segmented[row + i, col +j] and abs(im[row + i, col + j]-seed_point) < T):
                    segmented[row + i, col + j] = True
                    neighboorhood(seed_point,row+i,col+j)
    


    for row, col in seed_points:
        #segmented[row, col] = True
        #neighboorhood(im[row,col],row,col)
        seed_point = im[row,col]
        connected_pixel = [[row,col]]
        while  (len(connected_pixel) > 0): #as long as its not empty
            cell = connected_pixel.pop()
            for i in range(cell[0]-1,cell[0]+2):
                for j in range(cell[1]-1,cell[1]+2):
                    #boundries
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
    intensity_threshold = 90
    segmented_image = region_growing(im, seed_points, intensity_threshold)

    assert im.shape == segmented_image.shape, "Expected image shape ({}) to be same as thresholded image shape ({})".format(
        im.shape, segmented_image.shape)
    assert segmented_image.dtype == np.bool, "Expected thresholded image dtype to be np.bool. Was: {}".format(
        segmented_image.dtype)

    segmented_image = utils.to_uint8(segmented_image)
    utils.save_im("defective-weld-segmented90.png", segmented_image)
