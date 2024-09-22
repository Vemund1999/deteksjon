
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
from collections import Counter
from ultralytics.utils.plotting import Annotator
import threading
import time
from PIL import Image
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import easyocr

from PIL import Image
import webcolors
import pytesseract

from ultralytics import YOLO, solutions


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()







    def show_side_by_side(self, img_one, header_one, img_two, header_two):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(img_one, cmap='gray')
        plt.title(header_one)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(img_two, cmap='gray')
        plt.title(header_two)
        plt.axis('off')

        plt.show()

    def show_image(self, np_img):
        res = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

        # Display the image using plt
        plt.imshow(res)
        plt.title('Image with Lines')
        plt.axis('off')  # Hide axes
        plt.show()

    def determine_color(self, frame, xyxy):
        top_left = (int(xyxy[0][0]), int(xyxy[0][1]))
        bottom_right = (int(xyxy[0][2]), int(xyxy[0][3]))

        cutoff_percent = 0.1
        h_p = int(xyxy[0][1]) - int(xyxy[0][3])
        w_p = int(xyxy[0][2]) - int(xyxy[0][0])
        h_p = int(h_p * cutoff_percent)
        w_p = int(w_p * cutoff_percent)

        cropped_image_array = frame[top_left[1] - h_p:bottom_right[1] + h_p, top_left[0] + w_p:bottom_right[0] - w_p]
        cropped_image = Image.fromarray(cropped_image_array)

        k = 2
        pixels = cropped_image_array.reshape((-1, 3))
        kmeans = KMeans(n_clusters=k, n_init='auto')
        kmeans.fit(pixels)
        cluster_centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        clustered_array = labels.reshape(cropped_image_array.shape[0], cropped_image_array.shape[1])
        label_counts = np.bincount(labels)
        dominant_cluster = np.argmax(label_counts)
        dominant_color = cluster_centers[dominant_cluster] # rgb
        return tuple(dominant_color)

    def determine_license_plate(self, frame, xyxy):
        top_left = (int(xyxy[0][0]), int(xyxy[0][1]))
        bottom_right = (int(xyxy[0][2]), int(xyxy[0][3]))

        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        licence_plate_image = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        # blackhat = cv2.morphologyEx(licence_plate_image, cv2.MORPH_BLACKHAT, rectKern)




        grayscale_plate = cv2.cvtColor(licence_plate_image, cv2.COLOR_RGB2GRAY)
        gray = cv2.bilateralFilter(grayscale_plate, 13, 15, 15)
        text = pytesseract.image_to_string(gray, config='--psm 11')

        # self.show_side_by_side(gray, "gray", blackhat, "blackhat")
        return text




    def is_within_box(self, point, top_left, top_right, bottom_left, bottom_right):
        x, y = point
        tlx, tly = top_left
        trx, try_ = top_right
        blx, bly = bottom_left
        brx, bry = bottom_right
        
        # Assuming the box is aligned to the axes, meaning the edges are parallel to the axes
        # Check if point is within the horizontal bounds
        within_horizontal = min(tlx, blx) <= x <= max(trx, brx)
        # Check if point is within the vertical bounds
        within_vertical = min(bly, bry) <= y <= max(tly, try_)
        
        return within_horizontal and within_vertical




    def create_coordinate_system(self, frame, coordinate_points, object_positions_in_frame, object_colors_in_frame, object_ids_in_frame):
        src_points = np.array([coordinate_points[1], coordinate_points[2], coordinate_points[3], coordinate_points[0]], dtype=np.float32)
        width = int(np.linalg.norm(np.array(coordinate_points[3]) - np.array(coordinate_points[0])))
        height = int(np.linalg.norm(np.array(coordinate_points[0]) - np.array(coordinate_points[1])))


        positions_bottom = []
        for key in object_positions_in_frame:
            xyxy = object_positions_in_frame[key]
            position = [(xyxy[0][0] + (xyxy[0][2] - xyxy[0][0]) / 2), xyxy[0][3]]
            positions_bottom.append(position)

        positions_bottom = np.array(positions_bottom, dtype=np.float32).reshape(-1, 1, 2)


        dst_points = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped_positions = cv2.perspectiveTransform(positions_bottom, matrix)

        transformed_frame = cv2.warpPerspective(frame, matrix, (width, height))
        transformed_frame = np.ones(transformed_frame.shape, np.uint8) * 255

        if (len(positions_bottom) == 0):
            return transformed_frame

        warped_positions = warped_positions.reshape(-1, 2)


        for position, color_key, id in zip(warped_positions, object_colors_in_frame, object_ids_in_frame.values()):
            color = object_colors_in_frame.get(color_key) # fargene ender opp som svarte, fordi fargene faktisk er nesten kun svarte
            cv2.circle(transformed_frame, (int(position[0]), int(position[1])), 5, color, -1)
            text_position = (int(position[0]), int(position[1]) - 10)
            cv2.putText(transformed_frame, str(id), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return transformed_frame








    def get_colour_name(self, requested_color):
        min_colours = {}
        for name in webcolors.names("css3"):
            r_c, g_c, b_c = webcolors.name_to_rgb(name)
            rd = (r_c - requested_color[0]) ** 2
            gd = (g_c - requested_color[1]) ** 2
            bd = (b_c - requested_color[2]) ** 2
            min_colours[(rd + gd + bd)] = name
        return min_colours[min(min_colours.keys())]




    def count_objects(self, video_path, region_points, coordinate_points):
        # self.tracker
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        
        video_writer = cv2.VideoWriter("output/object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Init Object Counter
        
        counter = solutions.ObjectCounter(
            view_img=True,
            reg_pts=region_points,
            classes_names=self.model.names,
            draw_tracks=False,
            line_thickness=0,
            view_out_counts=False
        )
        
    
        # for storing track id
        objects_detected = {}

        coordinate_frames = []
        i = 0

        while cap.isOpened():
            i += 0
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            tracks = self.model.track(im0, persist=True, show=False)
            pred = self.model(im0)[0]

            dets = sv.Detections.from_ultralytics(pred)
            dets = self.tracker.update_with_detections(dets)
            
            im0 = counter.start_counting(im0, tracks)

            # for coordinate system
            object_positions_in_frame = {}
            object_colors_in_frame = {}
            object_ids_in_frame = {}

            # for counting objects over whole fram in this instance
            amount_of_objects_in_frame = {self.model.names[idx]: 0 for idx in range(len(self.model.names))}


            for det_i in range(len(dets) - 1):
                det = dets[det_i]

                tracker_id = det.tracker_id[0]

                

                top_left = (int(det.xyxy[0][0]), int(det.xyxy[0][1]))
                bottom_right = (int(det.xyxy[0][2]), int(det.xyxy[0][3]))
                bottom_left = (int(det.xyxy[0][0]), int(det.xyxy[0][3]))
                class_name = self.model.names[det.class_id[0]]



                if class_name == "license-plate":
                    if tracker_id not in objects_detected:
                        plate_text = self.determine_license_plate(im0, det.xyxy)
                        objects_detected[tracker_id] = plate_text
                    text = objects_detected[tracker_id]
                    if text == "":
                        text = "?"
                    # cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                    text_position = (bottom_left[0], bottom_left[1])
                    im0 = cv2.putText(im0, f'{text}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 123, 255), 2)

                # determine color of viechle, which will be used for the object box
                elif class_name == "car":
                    color = self.determine_color(im0, det.xyxy)
                    if tracker_id not in objects_detected: # if tracker_id hasn't been spotted before, or its a fram divisible by 500, set the color of the car
                        objects_detected[tracker_id] = color

                    object_positions_in_frame[det_i] = det.xyxy
                    object_colors_in_frame[det_i] = color
                    object_ids_in_frame[det_i] = tracker_id

                    color = objects_detected[tracker_id]
                    color = tuple(int(value) for value in color)
                    text = self.get_colour_name(color)
                    im0 = cv2.putText(im0, text, bottom_left, cv2.QT_FONT_NORMAL, 0.5, (0,255,0), 2)

                amount_of_objects_in_frame[class_name] += 1


            coordinate_frames.append( self.create_coordinate_system(im0, coordinate_points, object_positions_in_frame, object_colors_in_frame, object_ids_in_frame) )

            # drawing coordinate lines
            cv2.line(im0, coordinate_points[0], coordinate_points[1], (0,255,0), 1)
            cv2.line(im0, coordinate_points[1], coordinate_points[2], (0,255,0), 1)
            cv2.line(im0, coordinate_points[2], coordinate_points[3], (0,255,0), 1)
            cv2.line(im0, coordinate_points[3], coordinate_points[0], (0,255,0), 1)


            # drawing box for the amount of objects detected
            height, width, _ = im0.shape
            box_width, box_height = 400, 50
            top_left = (width - box_width - 10, height - box_height - 10)
            bottom_right = (width - 10, height - 10)
            im0 = cv2.rectangle(im0, top_left, bottom_right, (255, 255, 255), -1)
            text = ""
            for key, value in amount_of_objects_in_frame.items():
                if (key == "car"):
                    text += f"| amount of {key} in detection area: {value} |"
            # text = "| vehicle: 3 || license-plate: 4 |"
            text_position = (top_left[0] + 10, top_left[1] + 30)
            im0 = cv2.putText(im0, text, text_position, cv2.QT_FONT_NORMAL, 0.5, (0, 0, 0), 2)

            # max_height = max(im0.shape[0], coordinate_frames[i].shape[0])
            # img1_padded = np.pad(im0, ((0, max_height - im0.shape[0]), (0, 0), (0, 0)), mode='constant', constant_values=0)
            # img2_padded = np.pad(coordinate_frames[i], ((0, max_height - coordinate_frames[i].shape[0]), (0, 0), (0, 0)), mode='constant', constant_values=0)
            # combined_img = np.hstack((img1_padded, img2_padded))




            video_writer.write(im0)

        cap.release()
        video_writer.release()

        return coordinate_frames








