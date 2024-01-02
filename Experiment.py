import cv2 as cv
import numpy as np
from moviepy.video.fx.all import crop
from scipy.spatial.distance import cdist
from moviepy.editor import VideoFileClip


class Experiment():
    def __init__(self, video_path):
        self.video_path = video_path
        self.frames = None
        self.__debugging = False

    def enable_debugging(self):
        self.__debugging = True

    def disable_debugging(self):
        self.__debugging = False

    def __load_raw_video(self, ):
        video = cv.VideoCapture(self.video_path)
        frames = []

        while success:
            success, image = video.read()

            if success:
                frames.append(image)

        self.frames = np.array(frames)

    def __load_preprocess_video(self):
        video = cv.VideoCapture(self.video_path)
        number_of_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        video.set(cv.CAP_PROP_POS_FRAMES, number_of_frames//2)
        success, middle_frame = video.read()
        assert success

        if self.__debugging:
            cv.imshow('Middle Frame', middle_frame)
            cv.waitKey(0)
            cv.destroyAllWindows()

        original_img = cv.cvtColor(middle_frame,cv.COLOR_BGR2GRAY)
        cimg = cv.cvtColor(original_img,cv.COLOR_GRAY2BGR)

        circles = cv.HoughCircles(original_img,cv.HOUGH_GRADIENT,1,20, param1=50,param2=30,minRadius=10,maxRadius=20)
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

        circles = circles[0,:,0:2]

        assert circles.shape[0] == 16

        if self.__debugging:
            cv.imshow('All circles', cimg)
            cv.waitKey(0)
            cv.destroyAllWindows()

        distance_matrix = cdist(circles, circles)

        index = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)

        circle_one = circles[index[0]]
        circle_two = circles[index[1]]

        midpoint_between_circles = np.array((int(circle_one[0]+circle_two[0])/2, int(circle_one[1]+circle_two[1])/2), dtype=np.int16)

        if self.__debugging:
            cv.circle(cimg,circle_one,2,(0,0,255),3)
            cv.circle(cimg,circle_two,2,(0,0,255),3)
            cv.circle(cimg,midpoint_between_circles,2,(255,0,0),3)
            cv.imshow('Two circles of reference', cimg)
            cv.waitKey(0)
            cv.destroyAllWindows()

        circles = cv.HoughCircles(original_img,cv.HOUGH_GRADIENT,1,20, param1=50,param2=30,minRadius=200,maxRadius=900)
        circles = np.uint16(np.around(circles))

        distance_matrix = cdist(np.expand_dims(midpoint_between_circles, 0), circles[0,:,0:2])
        index = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
        big_circle = circles[0,index[1],:]

        if self.__debugging:
            cv.circle(cimg,(big_circle[0],big_circle[1]),big_circle[2],(0,255,0),2)
            cv.circle(cimg,(big_circle[0],big_circle[1]),2,(0,0,255),3)

            cv.imshow('Example - Show image in window', cimg)
            cv.waitKey(0)
            cv.destroyAllWindows()

        total_area = cimg.shape[0] * cimg.shape[1]
        new_area = (big_circle[-1]*2) * (big_circle[-1]*2)
        reduced = 1-(new_area/total_area)
        
        if self.__debugging:
            print(reduced*100)
        
        x_center = big_circle[0]
        y_center = big_circle[1]

        x_1, x_2 = x_center - big_circle[-1], x_center + big_circle[-1]
        y_1, y_2 = y_center - big_circle[-1], y_center + big_circle[-1]

        video.set(cv.CAP_PROP_POS_FRAMES, number_of_frames//2)

        frames = []

        while success:
            success, image = video.read()
            if success:
                frames.append(image[y_1:y_2,x_1:x_2,:])

        self.frames = np.array(frames)

        if self.__debugging:
            clip = VideoFileClip(self.video_path)
            cropped_clip = crop(clip, width=big_circle[-1]*2, height=big_circle[-1]*2, x_center=big_circle[0], y_center=big_circle[1])
            cropped_clip.write_videofile(self.video_path+'load_video_result_.mp4',codec="libx264")       

    def load_video(self, raw=False):
        """
        Loads the video as a numpy array.

        Args:
            raw (bool, optional): Setting raw to True saves the video as a numpy array
            without preprocessing, which might result in inefficient processing due to
            many unimportant pixels in the video. If set to False (default), the video
            undergoes preprocessing before being saved as a numpy array.
        """
        if raw:
            self.__load_raw_video()
        else:
            self.__load_preprocess_video()
