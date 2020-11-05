# Utility to resize the images output by the visualize_cnn_filters functions
# Adam Santos (addrick)
# Modified: 10/31/20

import cv2

for i in range(12):
    im = cv2.imread('stitched_filter_'+str(i)+'.png')
    if im is not None:
        res = cv2.resize(im, dsize=(im.shape[1] * 2, im.shape[0] * 2))
        cv2.imwrite('resized_stitched_filter_' + str(i) + '.png', res)
    if im is None:
        print("Filter " + str(i) + " not found.")


