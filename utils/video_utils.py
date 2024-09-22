import cv2
import numpy as np
from moviepy.editor import VideoFileClip, clips_array



def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(np.array(frame))
    return np.array(frames)


def save_video(ouput_video_frames,output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()

def combine_videos(clip1_path, clip2_path, save_path):
    output_folder = "output/"
    # Load the two videos
    clip1 = VideoFileClip(clip1_path)
    clip2 = VideoFileClip(clip2_path)

    # Ensure both clips are of the same duration
    min_duration = min(clip1.duration, clip2.duration)
    clip1 = clip1.subclip(0, min_duration)
    clip2 = clip2.subclip(0, min_duration)

    # Stack them side by side
    final_clip = clips_array([[clip1, clip2]])

    # Write the output to a file
    final_clip.write_videofile(save_path, codec="libx264")
