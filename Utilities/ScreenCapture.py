'''
import numpy as np
from PIL import ImageGrab
import cv2
import time

def screen_record():
    last_time = time.time()
    #while(True):
    # 800x600 windowed mode
    printscreen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
    print('loop took {} seconds'.format(time.time()-last_time))
    last_time = time.time()
    cv2.imshow('window',cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
            #break
    print(printscreen)

screen_record()
'''

import numpy as np
from PIL import ImageGrab
import cv2
import time
import json


def process_img(image):
    original_image = image
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection
    processed_img =  cv2.Canny(processed_img, threshold1 = 200, threshold2=200)


    #vertices = np.array([[5,720],[5,720],[640,150],[1280,150],[1915,720],[1915,720]], np.int32)

    #processed_img = roi(processed_img, [vertices])

    #lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, 20, 15)
    #draw_lines(processed_img, lines)

    return processed_img

def roi(img, vertices):
    #blank mask:
    mask = np.zeros_like(img)
    # fill the mask
    cv2.fillPoly(mask, vertices, 255)
    # now only show the area that is the mask
    masked = cv2.bitwise_and(img, mask)
    return masked

'''
def draw_lines(img, lines):
    for line in lines:
        coords = line[0]
        cv2.line(img, (coords[0]. coords[1]), (coords[2], coords[3]), [255,255,255], 3)
'''
def clearfile(filename):
    open(filename, "w").close()


def main():
    starting_value = 1
    file_name = "traningdata-1"
    training_data = []
    test = False
    last_time = time.time()
    while test == False:
        clearfile("C:\\Users\\denni\\Documents\\Assetto Corsa\\logs\\py_log.txt")

        #for 1920x1080
        #screen =  np.array(ImageGrab.grab(bbox=(0, 40, 1920, 1080)))

        screen = np.array(ImageGrab.grab(bbox=(0,40,800,640)))

        #print('Frame took {} seconds'.format(time.time()-last_time))
        last_time = time.time()

        new_screen = process_img(screen)
        cv2.imshow('window', new_screen)

        new_screen = cv2.resize(new_screen, (80,60))

        read = open("C:\\Users\\denni\\Documents\\Assetto Corsa\\logs\\py_log.txt", "r")
        lines = read.readlines()
        lines = [s.strip('\x00') for s in lines]
        lines = [s.strip('\n') for s in lines]
        for x in range(len(lines)):
            lines[x] = float(lines[x])
        lines[0] = (lines[0] + 450) / 900
        lines = np.array(lines)
        read.close()
        clearfile("C:\\Users\\denni\\Documents\\Assetto Corsa\\logs\\py_log.txt")
        new_screen = np.array(new_screen, dtype='uint8')
        training_data.append([new_screen, lines])
        np.save("test", training_data)

        if len(training_data) % 100 == 0:
            print(len(training_data))
            if len(training_data) == 4000:

                np.save(file_name, training_data)
                print("SAVED")
                training_data = []
                starting_value += 1
                file_name = "trainingdata-{}".format(starting_value)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        test = False

main()