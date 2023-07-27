from datetime import datetime
import os
import cv2
import ultralytics
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.tools.detections import Detections, BoxAnnotator
from counter import LineCounter, LineCounterAnnotator
import sys
HOME = os.getcwd()
sys.path.append(f"{HOME}/ByteTrack")
from yolox.tracker.byte_tracker import BYTETracker, STrack 
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from typing import List
import numpy as np
from config import *
from ultralytics import YOLO
# from alarm import alarm_init

ultralytics.checks()
@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.7
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


#converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections, 
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)
    
    tracker_ids = [None] * len(detections)
    
    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids
    
if __name__ == "__main__":
    """## Load model""" 
    model = YOLO(MODEL)
    model.fuse()

    number_0f_frames=FRAMES
    names = model.model.names
    print(names)

    """## Predict and annotate single frame"""

    # dict maping class_id to class_name
    CLASS_NAMES_DICT = model.model.names
    CLASS_ID = [0, 24]

    """## Predict and annotate whole video """

    from tqdm.notebook import tqdm
    from supervision.draw.color import Color

    # for file_name in os.listdir(r"./"):
    byte_tracker = BYTETracker(BYTETrackerArgs())
    
    video_info=VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    width, height = video_info.width, video_info.height

    LINE_START_F = Point(x=0, y=0)
    LINE_END_F= Point(x=0, y=height)
    LINE_START_L= Point(x=width, y=0)
    LINE_END_L= Point(x=width, y=height)
    

    id_count={}
    id_in = set()
    current_frame = 0
    counter_flag=0

    # create BYTETracker instance
    # create VideoInfo instance
    # create frame generator
    # create LineCounter instance
    f_line_counter = LineCounter(start=LINE_START_F, end=LINE_END_F)
    l_line_counter = LineCounter(start=LINE_START_L, end=LINE_END_L)
    # create instance of BoxAnnotator and LineCounterAnnotator
    box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=1)
    line_annotator_F = LineCounterAnnotator(thickness=4, text_thickness=4, text_scale=2,color=Color.red())
    line_annotator_L = LineCounterAnnotator(thickness=4, text_thickness=4, text_scale=2,color=Color.green())
    initial_frames=[]

    # open target video file
    # os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "hw_decoders_any;cuda"
    # cap=cv2.VideoCapture(SOURCE_VIDEO_PATH,cv2.CAP_FFMPEG,(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY ))
    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
    os.makedirs(TARGET_VIDEO_PATH, exist_ok=True)
    # videoWriter=None
    if not cap.isOpened():
        print("Error opening video stream or file")

    video_file_name=os.path.join(TARGET_VIDEO_PATH, TARGET_VIDEO_NAME+".mp4")
    videoWriter=cv2.VideoWriter(video_file_name,cv2.VideoWriter_fourcc(*"mp4v"), 10, (int(width),int(height)))

    # Capture frames from the RTSP stream on loop
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        if not ret:
            continue
        
        # model prediction on single frame and conversion to supervision Detections
        # Resize the frame to a smaller size to speed up the object detection process
        results = model(frame,imgsz=640,conf=0.5)
        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int)
        )
        # filtering out detections with unwanted classes
        mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)
        # tracking detections
        tracks = byte_tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=frame.shape,
            img_size=frame.shape
        )
        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_id)
        # filtering out detections without trackers
        mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)
        # format custom labels
        labels = [
            f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]
        # print(detections.tracker_id)
        # updating line counter
        f_line_counter.update(detections=detections)
        l_line_counter.update(detections=detections)
        detected_ids=list(detections.tracker_id)
        print(detected_ids)
        
        f_tracker=f_line_counter.tracker_state
   
        
        l_tracker=l_line_counter.tracker_state

        id_in.update(detected_ids)
        # print(id_in)

        for key,val in id_count.items():
            id_count[key]+=1
        
        for id in detected_ids:
            id_count[id]=0
        print(id_count)
        
        for key, val in id_count.items():
            if val == 0 and key in detected_ids:
                counter_flag+=1

        if counter_flag>number_0f_frames:
            #Put alert text on frame
            cv2.putText(frame, "ALERT", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #make directory for alert images if doesn't exist
            if not os.path.exists("ALERT_FOLDER"):
                os.mkdir("ALERT_FOLDER")
            #save alert image if not saved
            if not os.path.exists(f"ALERT_FOLDER/{TARGET_VIDEO_NAME}.jpg"):
                cv2.imwrite(f"ALERT_FOLDER/{TARGET_VIDEO_NAME}.jpg",frame)


        if len(detected_ids)==0:
            counter_flag=0
        
        total_person=len([id for id in detections.class_id if id==0])

        frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
        cv2.putText(frame, f"Total Person Count: {total_person}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        line_annotator_F.annotate(frame=frame, line_counter=f_line_counter)
        line_annotator_L.annotate(frame=frame, line_counter=l_line_counter)
        videoWriter.write(frame)
        cv2.imshow("frame",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    videoWriter.release()
    cap.release()
    