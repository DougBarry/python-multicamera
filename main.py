# import the necessary packages
from __future__ import print_function
# from imutils.video import WebcamVideoStream
# from imutils.video import FPS
import argparse
from Xlib.display import Display
import imutils
import cv2
from threading import Thread
from time import sleep
# import the necessary packages
import datetime


class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()


class WebcamVideoStream:
    def __init__(self, src=0, scale_percent: float=100):
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.scale_pc = scale_percent
        self.update(True)


    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self, once: bool=False):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
            width = int(self.frame.shape[1] * (self.scale_pc / 100))
            height = int(self.frame.shape[0] * (self.scale_pc / 100))
            self.resized = resized = cv2.resize(self.frame, (width, height), interpolation=cv2.INTER_AREA)

            if once:
                return

    def read_frame(self):
        # return the frame most recently read
        return self.frame

    def read_resized(self):
        # return resized frame
        return self.resized

    def set_scale(self, scale_percent):
        self.scale_pc = scale_percent

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


def clearCapture(capture):
    capture.release()


#    cv2.destroyAllWindows()

def isCameraWorking(index):
    try:
        cap = cv2.VideoCapture(index)
        ret, frame = cap.read()
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clearCapture(cap)
        return True
    except:
        clearCapture(cap)
    return False


def countCameras():
    n = 0
    for i in range(10):
        if isCameraWorking(i):
            n += 1
    return n


def get_screen_info() -> (int, int):
    screen = Display(':0').screen()
    return screen.width_in_pixels, screen.height_in_pixels


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=10000,
                help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=1,
                help="Whether or not frames should be displayed")
ap.add_argument("-a", "--all-cameras", type=int, default=1,
                help="Whether or not to capture from all attached cameras")
args = vars(ap.parse_args())

# created a *threaded* video stream, allow the camera sensor to warmup,
# and start the FPS counter
print("[INFO] sampling THREADED frames from webcam...")

scale_percent = 75  # percent of original size

max_x, max_y = get_screen_info()
window_pos = [
    (0, 0),
    (max_x / 3, 0),
    ((max_x / 3) * 2, 0),
    (0, max_y / 2),
    (max_x / 3, (max_y / 3)*2),
    ((max_x / 3) * 2, (max_y / 3)*2),
]

vs = []
n = 0
for camera_id in range(0, 10):
    if isCameraWorking(camera_id):
        window_name = f"Camera{camera_id}"
        cv2.namedWindow(window_name)
        x, y = window_pos[n]
        cv2.moveWindow(window_name, int(x), int(y))
        camera = WebcamVideoStream(src=camera_id, scale_percent=scale_percent).start()
        frame = camera.read_resized()
        cv2.imshow(window_name, frame)
        vs.append( (camera, window_name) )
        n += 1
    if args['all_cameras'] < 0:
        if vs.count() == 1:
            break

fps = FPS().start()

if len(vs) > 0:

    # loop over some frames...this time using the threaded stream
    while fps._numFrames < args["num_frames"]:
        frameset = []
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        for (camera, window_name) in vs:
            frame = camera.read_resized()
            if frame is not None:
                # frame = imutils.resize(frame, width=400)
                frameset.append((frame, window_name))
            # check to see if the frame should be displayed to our screen

        if args["display"] > 0:
            for (frame, window_name) in frameset:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF

        # update the FPS counter
        fps.update()
    # stop the timer and display FPS information
    fps.stop()

    for (camera, window_name) in vs:
        camera.stop()

    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    cv2.destroyAllWindows()

else:
    print("No cameras available")
