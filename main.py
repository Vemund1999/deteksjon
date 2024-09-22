
import cv2
import numpy as np
from trackers import Tracker
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import read_video, save_video, combine_videos
import time
import os
import subprocess
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.editor import VideoFileClip, clips_array


def main():
    video_frames = None
    
    #if not os.path.exists('pkl/video_frames.pkl'):
    #    video_frames = read_video('assets/traffic.mp4')
    #    with open('pkl/video_frames.pkl', 'wb') as f:
    #        pickle.dump(video_frames, f)
    
    # print("loading frames...")
    # with open('pkl/video_frames.pkl', 'rb') as f:
    #     video_frames = pickle.load(f)
    # print("done loading frames")

    tracker = Tracker('models/best_car_bus.pt')


    frames_dir = 'frames'
    os.makedirs(frames_dir, exist_ok=True)

    
    region_points = [(110, 242), (80, 260), (632, 260), (620, 242)]
    coordinate_points = [(10, 300), (290,65), (420,65), (680, 300)]



    #save_video(video_frames, 'output/output_video.avi')
    #save_video(coordinate_plot, 'output/coordinate_video.avi')
    #save_video(combined, 'output/combined_video.avi')
    # put_videos_side_by_side('output/output_video.avi', 'output/coordinate_video.avi', 'output/combined.avi')
    

    coordinate_frames = tracker.count_objects('assets/ny.mp4',region_points, coordinate_points)
    save_video(coordinate_frames, "output/coordinate_video.avi")
    combine_videos("output/object_counting_output.avi", "output/coordinate_video.avi", "output/final_output.avi")



if __name__ == "__main__":
    main()











