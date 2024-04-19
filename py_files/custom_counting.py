from collections import defaultdict
import math
import cv2
import numpy as np
import time
import csv
from ultralytics import YOLO

# Grab Currrent Time Before Running the Code
start = time.time()

#TODO set weight for detection & Classification below
# choose weight file (.pt)
weight_path = "best.pt"
model = YOLO(weight_path)

# Open the video file
#TODO Set input file name below
# fill in put path in any vdo file (.MOV or .avi or .mp4 file is recommended)
input_path = "INPUT_PATH"
cap = cv2.VideoCapture(input_path)
frame_width = int(cap.get(3))           # CV_CAP_PROP_FRAME_WIDTH
frame_height = int(cap.get(4))          # CV_CAP_PROP_FRAME_HEIGHT
fps= cap.get(5)                         # CV_CAP_PROP_FPS
# frame_current = int(cap.get(1))         # cv2.CAP_PROP_POS_FRAMES
frame_total = int(cap.get(7))           # cv2.CAP_PROP_FRAME_COUNT
frame_current = int(0)


#TODO Set percentage of output size  
# this number cannot be zero (1 is a minimize size and 100 is original size) 
scale_percent = 50 # scale down percentage of output size from original size // This affect to only size of collecting VDO, not counting method
resized_width = int(frame_width * scale_percent / 100)
resized_height = int(frame_height * scale_percent / 100)
out_dim = (resized_width, resized_height)

#TODO Set output file name below
output_path = "OUTPUT_PATH"
#set output path in avi file (.avi)
out = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc('M','J','P','G'), fps, (out_dim))

#TODO Set uncomment to store track history or object (inclding line 170-177 to show tracking line)
# # Store the track history
# track_history = defaultdict(lambda: []) # for tracking object

# Class names of this model
cls_name = ['full_trailer', 'heavy_bus', 'heavy_truck'\
    , 'light_bus', 'light_truck', 'medium_bus', 'medium_truck'\
    , 'motorcycle', 'passenger_car', 'semi_trailer', 'van-mpv'
    ]

# Color of each class in this model (can change if needed)
cls_color=[	(255,105,180), (64,224,208), (147,112,219)\
    , (139,69,19), (255,99,71), (119,136,153), (61, 193, 203)\
    , (32,178,170), (30,144,255), (255,140,0), (128,128,0)]

# Declare variables 
cls_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
cnt_left_line_cls_0= 0
cnt_left_line_cls_1 = 0
cnt_left_line_cls_2 = 0
cnt_left_line_cls_3 = 0
cnt_left_line_cls_4 = 0
cnt_left_line_cls_5 = 0
cnt_left_line_cls_6 = 0
cnt_left_line_cls_7 = 0
cnt_left_line_cls_8 = 0
cnt_left_line_cls_9 = 0
cnt_left_line_cls_10 = 0
sum_cnt_left_line = 0
cnt_cls_left_line = [cnt_left_line_cls_0, cnt_left_line_cls_1, cnt_left_line_cls_2, cnt_left_line_cls_3, cnt_left_line_cls_4, cnt_left_line_cls_5, cnt_left_line_cls_6, cnt_left_line_cls_7, cnt_left_line_cls_8, cnt_left_line_cls_9, cnt_left_line_cls_10]

counted_obj_id = []

xmin_list_left_line = []
ymin_list_left_line = []
xmax_list_left_line = []
ymax_list_left_line = []
id_list_left_line = []
cls_list_left_line = []
conf_list_left_line = []

#TODO set positions of counting lines below
# change percentage of start and end points of counting lines
offset_left_line=30
pos_y_left_line_1=int(frame_height*0.85)
# pos_y_left_line_2=int(frame_height*0.50)
pos_x_left_line_1=int(frame_width*0.42)
pos_x_left_line_2=int(frame_width*0.95)

################################################################################################################################
#Counting Method   
def count_obj(im, box, id, cls, conf):
    global cls_id, sum_cnt_left_line, cnt_left_line_cls_0, cnt_left_line_cls_1, cnt_left_line_cls_2, cnt_left_line_cls_3, \
        cnt_left_line_cls_4, cnt_mid_line_cls_4, cnt_right_line_cls_4, cnt_left_line_cls_5, cnt_left_line_cls_6, cnt_left_line_cls_7, cnt_left_line_cls_8, cnt_left_line_cls_9, \
        cnt_left_line_cls_10, cnt_cls_left_line, counted_obj_id, xmin_list_left_line, ymin_list_left_line, xmax_list_left_line, ymax_list_left_line, id_list_left_line, cls_list_left_line, conf_list_left_line
    center_coor = (int(box[0] + (box[2]-box[0])/2 ), int(box[1] + (box[3] - box[1])/2 ))
    if id not in counted_obj_id:
        # Counting for upstream (lefthand drive)   
        if center_coor[1] < (pos_y_left_line_1+offset_left_line) and center_coor[1] > (pos_y_left_line_1-offset_left_line):
            if center_coor[0] > (pos_x_left_line_1) and center_coor[0] < (pos_x_left_line_2):
                counted_obj_id.append(id)
                # cv2.rectangle(im, (pos_x_left_line_1, (pos_y_left_line_1-offset_left_line)), ((pos_x_left_line_1+pos_x_left_line_2-pos_x_left_line_1), pos_y_left_line_1+offset_left_line), (0, 200, 0), -1) 
                cv2.line(im, (pos_x_left_line_1, pos_y_left_line_1), (pos_x_left_line_2, pos_y_left_line_1), (0,255,255), thickness=5)
                for i in cls_id:
                    if cls == i:
                        cnt_cls_left_line[i] += 1
                        sum_cnt_left_line += 1
                    xmin_list_left_line.append(box[0])
                    ymin_list_left_line.append(box[1])
                    xmax_list_left_line.append(box[2])
                    ymax_list_left_line.append(box[3])
                    id_list_left_line.append(id)
                    cls_list_left_line.append(cls)
                    conf_list_left_line.append(conf)
                    
################################################################################################################################

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, annotated_frame = cap.read()
    frame_current += 1

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        ##TODO Select Tracker (Original YOLOv8 supports ByteTrack and BotSORT (Feb 2024))
        # results = model.track(annotated_frame, persist=True, tracker="bytetrack.yaml", verbose = False)
        results = model.track(annotated_frame, persist=True, tracker="botsort.yaml", verbose = False)
        
        for r in results:
            if results[0].boxes is None or results[0].boxes.id is None:
                continue
            # Get the boxes and track IDs
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            # Confidence
            confidences = results[0].boxes.conf.cpu()
            # Class Name
            class_ids = results[0].boxes.cls.int().cpu()
            # # Visualize the results on the frame
            # annotated_frame = results[0].plot(font_size=0.1)

            # Plot the tracks
            for bbox, track_id, conf, cls in zip(boxes, track_ids, confidences, class_ids):
                # # Draw the tracking lines using center coordinator
                center_coor = (int(bbox[0] + (bbox[2]-bbox[0])/2 ), int(bbox[1] + (bbox[3] - bbox[1])/2 ))
                #TODO Set uncomment to visualize tracking in result
                # track = track_history[track_id]
                # track.append(center_coor)  # x, y center point
                # if len(track) > 90:  # retain 90 tracks for 90 frames
                #     track.pop(0)
                # points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                # cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 230, 0), thickness=1)
                #TODO Set uncomment to visualize center point of object
                # cv2.circle(annotated_frame, center_coor, 1, (255,0,255), -1)
                # Visualize bounding boxes (bbox)
                cv2.rectangle(annotated_frame, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), cls_color[cls], thickness = 2) 
                # Visualize text of bbox class
                cls_colors = []
                cv2.putText(annotated_frame, cls_name[cls], (int(bbox[0]),int(bbox[1]-15)) \
                            , cv2.FONT_HERSHEY_COMPLEX , 1, cls_color[cls], 2, cv2.LINE_AA)
                # Run counting process
                count_obj(annotated_frame, bbox, track_id, cls, conf)
                
        #Visualize Left lane Counting
        line_color_left = (255, 0, 0)
        line_start_point_left = (pos_x_left_line_1, pos_y_left_line_1)
        line_end_point_left = (pos_x_left_line_2, pos_y_left_line_1)
        cv2.line(annotated_frame, line_start_point_left, line_end_point_left, line_color_left, thickness=2)
        font_color_left = (255, 0, 0)
        font_color_left_sum = (255, 0, 0)
        font_org_left = (int(frame_width*0.02), int(frame_height*0.03))     # origin point of result text of left line counting
        font_nl_left = 30                                                   # new line space
        font_style_left = cv2.FONT_HERSHEY_COMPLEX
        font_scale_left = 1                                                # 0.75 for FullHD or 1.5 for 4K
        font_scale_left_sum = 1                                             # 1 for FullHD or 2 for 4K
        font_thickness_left = 1                                             # 1 for FullHD or 2 for 4K
        font_thickness_left_sum = 2                                         # 2 for FullHD or 3 for 4K
        cv2.putText(annotated_frame, "motorcycle: "+str(cnt_cls_left_line[7]), font_org_left, font_style_left, font_scale_left, font_color_left, font_thickness_left, cv2.LINE_AA)
        cv2.putText(annotated_frame, "passenger_car: "+str(cnt_cls_left_line[8]), (font_org_left[0],font_org_left[1]+font_nl_left*1), font_style_left, font_scale_left, font_color_left, font_thickness_left, cv2.LINE_AA)
        cv2.putText(annotated_frame, "van: "+str(cnt_cls_left_line[10]), (font_org_left[0],font_org_left[1]+font_nl_left*2), font_style_left, font_scale_left, font_color_left, font_thickness_left, cv2.LINE_AA)
        cv2.putText(annotated_frame, "light_truck: "+str(cnt_cls_left_line[4]), (font_org_left[0],font_org_left[1]+font_nl_left*3), font_style_left, font_scale_left, font_color_left, font_thickness_left, cv2.LINE_AA)
        cv2.putText(annotated_frame, "medium_truck: "+str(cnt_cls_left_line[6]), (font_org_left[0],font_org_left[1]+font_nl_left*4), font_style_left, font_scale_left, font_color_left, font_thickness_left, cv2.LINE_AA)
        cv2.putText(annotated_frame, "heavy_truck: "+str(cnt_cls_left_line[2]), (font_org_left[0],font_org_left[1]+font_nl_left*5), font_style_left, font_scale_left, font_color_left, font_thickness_left, cv2.LINE_AA)
        cv2.putText(annotated_frame, "light_bus: "+str(cnt_cls_left_line[3]), (font_org_left[0],font_org_left[1]+font_nl_left*6), font_style_left, font_scale_left, font_color_left, font_thickness_left, cv2.LINE_AA)
        cv2.putText(annotated_frame, "medium_bus: "+str(cnt_cls_left_line[5]), (font_org_left[0],font_org_left[1]+font_nl_left*7), font_style_left, font_scale_left, font_color_left, font_thickness_left, cv2.LINE_AA)
        cv2.putText(annotated_frame, "heavy_bus: "+str(cnt_cls_left_line[1]), (font_org_left[0],font_org_left[1]+font_nl_left*8), font_style_left, font_scale_left, font_color_left, font_thickness_left, cv2.LINE_AA)
        cv2.putText(annotated_frame, "semi_trailer: "+str(cnt_cls_left_line[9]), (font_org_left[0],font_org_left[1]+font_nl_left*9), font_style_left, font_scale_left, font_color_left, font_thickness_left, cv2.LINE_AA)
        cv2.putText(annotated_frame, "full_trailer: "+str(cnt_cls_left_line[0]), (font_org_left[0],font_org_left[1]+font_nl_left*10), font_style_left, font_scale_left, font_color_left, font_thickness_left, cv2.LINE_AA)
        cv2.putText(annotated_frame,("sum: ")+str(sum_cnt_left_line),(font_org_left[0]+10,(font_org_left[1]+font_nl_left*11+5)), font_style_left , font_scale_left_sum , font_color_left_sum , font_thickness_left_sum , cv2.LINE_AA)

        print("Frames in Process : "+ str(frame_current)+ " / " + str(frame_total) + " frames" + "\t" +"("+str(round(frame_current/frame_total*100,2))+ "/ 100.0 %)"\
            + f" ------ Data has been written to {output_path}")
                
        # resize 
        resized = cv2.resize(annotated_frame, out_dim)
        # Write the frame into the file 'output.avi'
        out.write(resized)
        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", resized)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
out.release()
# Closes all the frames
cv2.destroyAllWindows()
