import cv2
import mss
import numpy as np
import time

def roi(img, vertices):
    #blank mask:
    mask = np.zeros_like(img)

    # fill the mask
    cv2.fillPoly(mask, vertices, 255)

    # now only show the area that is the mask
    masked = cv2.bitwise_and(img, mask)

    return masked

with mss.mss() as sct:
    monitor = {'top': 40, 'left': 0, 'width': 800, 'height': 640}
    img_matrix = []

    last_time = time.time()

    while True:

        # Get raw pixels from screen and save to numpy array
        img = np.array(sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.Canny(img, threshold1 = 200, threshold2=300)
        vertices = np.array([[10,500],[10,300],[300,200],[500,200],[800,300],[800,500]], np.int32)
        img = roi(img, [vertices])

        # Save img data as matrix
        img_matrix.append(img)

        # Display Image
        cv2.imshow('screencapture', img)

        # Frame rate display
        print(('{0:.0f} fps').format(1/(time.time()-last_time)))
        last_time = time.time()

        # Press q to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

