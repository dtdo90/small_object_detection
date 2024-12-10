import cv2
import time
import torch
import numpy as np
from ultralytics import RTDETR
from utils import VisTrack


class InferenceBaseline:
    def __init__(self):
        self.detection_model=None
        self.names={
            "0": "pedestrian",
            "1": "people",
            "2": "bicycle",
            "3": "car",
            "4": "van",
            "5": "truck",
            "6": "tricycle",
            "7": "awning-tricycle",
            "8": "bus",
            "9": "motor"
        }

    def inference_image(self,weights,source):
        # load model
        if self.detection_model is None:
            self.detection_model=RTDETR(weights)
            self.detection_model.to("mps")
            self.detection_model.fuse()  # Fuse Conv2d + BatchNorm2d layers
            print(f"Model loaded! Device {self.detection_model.device}")
        # read image
        image=cv2.imread(source)
        with torch.no_grad():  # Disable gradient calculation
            results = self.detection_model(image,conf=0.35)

        results=results[0] # take out batch dimension
        bboxes=results.boxes.xyxy.cpu().numpy()
        ids=results.boxes.cls.cpu().numpy().astype(int)
        scores=results.boxes.conf.cpu().numpy() 

        image=VisTrack().draw_bounding_boxes(image,bboxes,ids,self.names,scores)  
        cv2.imwrite(source[:-4]+"_rtdetr.png",image)

    def process_frame(self,frames):
        # process a batch of frames
        with torch.no_grad():  # Disable gradient calculation
            results = self.detection_model(frames,conf=0.3)

        boxes=[result.boxes.xyxy.cpu().numpy() for result in results]
        ids=[result.boxes.cls.cpu().numpy() for result in results]
        scores=[result.boxes.conf.cpu().numpy() for result in results]

        return boxes, ids, scores

    def inference_video(self, weights, source, buffer_size):
        if self.detection_model is None:
            self.detection_model=RTDETR(weights)
            self.detection_model.to("mps")
            self.detection_model.fuse()  # Fuse Conv2d + BatchNorm2d layers
            print(f"Model loaded! Device {self.detection_model.device}")

        cap = cv2.VideoCapture(source)
        assert cap.isOpened(), "Error reading video file"
        frame_count = 0
        frame_buffer = []   # store batch of frames for processing
        processing_times = []
        vis_track = VisTrack()

        if type(source) is not int: # video input

            width, height, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
            num_frames = int(cap.get(7))
            print(f"Processing {num_frames} frames | Resolution: {width}x{height}")

            out = cv2.VideoWriter(source[:-4] + "_baseline_rtdetr_half_fps.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps//2, (width, height))
            
            
            while frame_count < num_frames:
                start_time = time.time() # start time for the whole process

                success, frame = cap.read()
                if not success:
                    print(f"Error reading frame {frame_count}")
                    break

                # append frame to buffer
                frame_buffer.append(frame)

                # Process when buffer is full or on last frame
                if len(frame_buffer) >= buffer_size or frame_count == num_frames - 1:
                    buffer_start_time=time.time() # start time for the buffer
                    bboxes, ids, scores = self.process_frame(frame_buffer)
                    inference_time=time.time()-buffer_start_time

                    # Calculate FPS based on total inference time divided by number of frames
                    fps = len(frame_buffer) / inference_time
                    fps_text = f"FPS: {fps:.2f}"

                    # Draw and write frames
                    for frame, bbox, id_, score in zip(frame_buffer, bboxes, ids, scores):
                        id_=id_.astype(int) # for suitability to VisTrack
                        frame_processed = vis_track.draw_bounding_boxes(
                            frame, bbox, id_, self.names, score)
                        
                        # Use the same FPS value for all frames in this buffer
                        cv2.putText(frame_processed, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 255, 0), 2, cv2.LINE_AA)
                        out.write(frame_processed)

                    # Clear buffer for the next one
                    frame_buffer.clear()

                frame_count += 1
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
            
            # Print average processing time
            if processing_times:
                avg_fps = 1 / (sum(processing_times) / len(processing_times))
                print(f"\n\nProcessing complete!")
                print(f"Average FPS: {avg_fps:.2f}")
            out.release()
            cap.release()
            cv2.destroyAllWindows()
                

        else: # real-time camera
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Reduce frame buffer
            cv2.namedWindow("Webcam Feed")
            fps_text = "FPS: Calculating..." # place holder for fps
            while True:
                start_time = time.time()
                success, frame = cap.read()
                assert success, "Fail to read frame"

                # append frame to buffer
                frame_buffer.append(frame)

                # Process when buffer is full or on last frame
                if len(frame_buffer) >= buffer_size:
                    bboxes, ids, scores = self.process_frame(frame_buffer)
                    # Draw and write frames
                    for frame, bbox, id_, score in zip(frame_buffer, bboxes, ids, scores):
                        id_=id_.astype(int) # for suitability to VisTrack
                        frame_processed = vis_track.draw_bounding_boxes(
                            frame, bbox, id_, self.names, score)
                        # Overlay FPS text on the frame
                        cv2.putText(frame_processed, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.imshow("Webcam Feed", frame_processed)
                    # Clear buffers
                    frame_buffer.clear()

                frame_count += 1
                processing_time = time.time() - start_time
                processing_times.append(processing_time)

                # Calculate running FPS (only every 4 frames)
                if frame_count % 4 == 0:
                    avg_time = sum(processing_times[-4:]) / len(processing_times[-4:])
                    current_fps = 1 / avg_time if avg_time > 0 else 0
                    fps_text = f"FPS: {current_fps:.2f}"


                # Check for ESC key press
                if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for ESC
                    print("ESC pressed, exiting...")
                    break
     
                
            cap.release()
            cv2.destroyAllWindows()


if __name__=="__main__":
    inference=InferenceBaseline()
    # image inference
    # inference.inference_image(weights="models/rt_detr_l_413.pt",
    #                           source="images/capture_frame.jpg")
    # video inference
    # inference.inference_video(
    #     weights="models/rt_detr_l_413.pt", 
    #     source="images/raw.mp4",
    #     buffer_size=4)
    # webcam inference
    inference.inference_video(
        weights="models/rt_detr_l_413.pt", 
        source=1,
        buffer_size=2)
    






