import cv2
import time
import numpy as np

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from utils import VisTrack

names={0: "pedestrian", 
       1: "people", 
       2: "bicycle", 
       3: "car", 
       4: "van", 
       5: "truck", 
       6: "tricycle",
       7: "awning-tricycle",
       8: "bus",
       9: "motor"
       }

class SAHIInference:
    def __init__(self):
        self.detection_model=None
        self.names={"0": "pedestrian", 
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
    
    def load_model(self,weights): # weights= path to trained model
        
        # Initiate detection model
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8", model_path=weights, confidence_threshold=0.3, device="mps"
        )
        print(f"Model loaded! Device {self.detection_model.device}")
    
    def inference_image(self,
                  source):
        
        image=cv2.imread(source)
        height,width=image.shape[:2]
        
        # init annotator for plotting detection
        results=get_sliced_prediction(image,
                                        self.detection_model,
                                        slice_height=height//3,
                                        slice_width=width//3,
                                        overlap_height_ratio=0.2,
                                        overlap_width_ratio=0.2)

        bboxes=np.array([[det.bbox.minx,det.bbox.miny,det.bbox.maxx,det.bbox.maxy] for det in results.object_prediction_list])
        ids=np.array([det.category.id for det in results.object_prediction_list])
        names=self.names
        scores=np.array([det.score.value for det in results.object_prediction_list])
        
        image=VisTrack().draw_bounding_boxes(image,bboxes,ids,names,scores)  
        cv2.imwrite(source[:-4]+"_detect.png",image)

    def inference_frame(self,frame,width,height):
        # get prediction
        results=get_sliced_prediction(frame,
                                    self.detection_model,
                                    slice_height=height//2,
                                    slice_width=width//2,
                                    overlap_height_ratio=0.1,
                                    overlap_width_ratio=0.1)

        bboxes=np.array([[det.bbox.minx,det.bbox.miny,det.bbox.maxx,det.bbox.maxy] for det in results.object_prediction_list])
        ids=np.array([det.category.id for det in results.object_prediction_list])
        scores=np.array([det.score.value for det in results.object_prediction_list])
        return bboxes,ids,scores
    
    def inference_video(self,
                  weights,        # path to trained model 
                  source,         # video_path or webcame
                  view_img=False,
                  save_img=False, # if source is img
                  exist_ok=False, 
                  track=False):
        if self.detection_model is None:
            self.load_model(weights)

        cap=cv2.VideoCapture(source)
        assert cap.isOpened(), "Error reading video file"
        # Retrieve width and height
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_count=0
        vis_track = VisTrack()

        if type(source) is not int: # video input
        
            fps, num_frames=cap.get(5), int(cap.get(7))
            print(f"Processing {num_frames} frames | Resolution: {width}x{height} | fps: {fps}")
                    
            # video writer
            out = cv2.VideoWriter(source[:-4]+"_sahi_baseline.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

            start_time=time.time()
            while frame_count<num_frames:
                success,frame=cap.read()
                if not success:
                    print(f"Error reading frame {frame_count}. SKipping...")
                    frame_count+=1
                    continue
    
                # get prediction on frame
                bboxes,ids,scores= self.inference_frame(frame,width,height)                
                frame_processed=vis_track.draw_bounding_boxes(frame,bboxes,ids,self.names,scores)         
                out.write(frame_processed)

                frame_count+=1

            end_time=time.time()
            print(f"Average fps: {num_frames/(end_time-start_time):.2f}")

            out.release()
            cap.release()
            cv2.destroyAllWindows
        
        else: # realtime camera
            while True:
                start_time=time.time()
                success,frame=cap.read()
                assert success, "Fail to read frame"

                # get prediction
                bboxes,ids,scores= self.inference_frame(frame,width,height)                
                frame_processed=VisTrack().draw_bounding_boxes(frame,bboxes,ids,self.names,scores)         
                
                # Overlay FPS text on the frame
                current_fps = 1/(time.time() - start_time)
                cv2.putText(frame_processed, f"FPS: {current_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("Webcam Feed", frame_processed)

                frame_count+=1

                # Check for ESC key press
                if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for ESC
                    print("ESC pressed, exiting...")
                    break

            cap.release()
            cv2.destroyAllWindows()
     


if __name__=="__main__":        
    sahi=SAHIInference()
    sahi.inference_video(weights="models/best_11m.pt",
                         source=1)
    #sahi.inference_image(weights="models/best_11m.pt",source="images/Capture.PNG")

