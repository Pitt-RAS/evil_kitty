import cv2
import sys
import numpy as np

WIDTH = 640
HEIGHT = 480
STEPSIZE = 20
R = 10

# if no command line arguments, print usage and exit
if len(sys.argv) < 2:
    print("Usage: python webcam.py <path-to-cascade-file>")
    exit()

# path to the cascade file, taken from command line
cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

#initialize webcam capture as device 0
video_capture = cv2.VideoCapture(0)

# some predefined eye shapes
happy_eye = [(0,9), (0,8), (0,7), (0,6),
             (1,5), (1,4), (2,3), (2,2),
             (3,1), (4,0), (5,0), (6,0),
             (7,1), (8,2), (8,3), (9,4),
             (9,5), (10,6), (10,7), (10,8),
             (10,9)]

angry_eye_l = [(0,0), (1,0), (1,1), (2,1),
               (2,2), (3,2), (4,2),
               (2,3), (4,3), (5,3), (6,3),
               (1,4), (6,4), (7,4), (8,4),
               (1,5), (8,5), (9,5), (10,5),
               (1,6), (4,6), (5,6), (6,6), (9,6),
               (2,7), (4,7), (5,7), (6,7), (9,7),
               (3,8), (4,8), (5,8), (6,8), (7,8), (8,8)]

angry_eye_r = [(10,0), (9,0), (9,1), (8,1),
               (8,2), (7,2), (6,2),
               (8,3), (6,3), (5,3), (4,3),
               (9,4), (4,4), (3,4), (2,4),
               (9,5), (2,5), (1,5), (0,5),
               (9,6), (6,6), (5,6), (4,6), (1,6),
               (8,7), (6,7), (5,7), (4,7), (1,7),
               (7,8), (6,8), (5,8), (4,8), (3,8), (2,8)]

# draw a single "point" - circle on discretized image
def draw_point(x, y, color, frame):
    cv2.circle(frame, (x * STEPSIZE + R, y * STEPSIZE + R), R, color, -1)

# draw the shape given by the list of (x,y) tuples
def draw_pixel_shape(shape, color, frame, offset):
    for point in shape:
        draw_point(point[0] + offset[0], point[1] + offset[1], color, frame)

# draw two happy eyes
def draw_happy_eyes(frame):
    # eye 1
    draw_pixel_shape(happy_eye, (160, 160, 255), frame, (4,7))
 
    # eye 2
    draw_pixel_shape(happy_eye, (160, 160, 255), frame, (16,7))

# draw two angry eyes
def draw_angry_eyes(frame):
    draw_pixel_shape(angry_eye_l, (0,0,255), frame, (4,7))
    draw_pixel_shape(angry_eye_r, (0,0,255), frame, (16,7))

# capture and process camera data to find faces
# and decide which eyes to show
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    eyes = np.zeros((480,640,3), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    if len(faces) > 0:
        draw_happy_eyes(eyes)
    else:
        draw_angry_eyes(eyes)

    # Display the resulting frame
    cv2.imshow('Video', eyes)

    # if the user presses 'q', quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
