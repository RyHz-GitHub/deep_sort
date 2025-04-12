from ultralytics import YOLO
from ultralytics.utils.ops import xywh2ltwh
import argparse
import torch
import os


# usage: python generate_yolo_detections.py --mot_dir=MOT16_smallsample/train --save_images=False

# TODO generic video_seq instead of MOT
# TODO play video with yolo detections (nice extra not neccessary)
  


def yolo_detections_MOT16(mot_dir = "MOT16_smallsample\\train", output_dir = "YOLO", save_images = False, img_out_dir = "YOLO\\bbox_images" ,visualize = False):
    
    # Load Model, currently pre-trained, faster versions available but yolo11x is the most precise
    model = YOLO("yolo11x.pt")


    for sequence in os.listdir(mot_dir):
        print(f"Processing {sequence}")

        output_dir_sequence = os.path.join(output_dir, mot_dir, sequence, "det")
        output_file = os.path.join(output_dir_sequence,"det.txt")
        video_dir = os.path.join(mot_dir, sequence, "img1")
        os.makedirs(output_dir_sequence, exist_ok=True)
        if save_images:
            os.makedirs(os.path.join(img_out_dir,sequence),exist_ok=True)
        

        # Perform object detection on all frames of the video sequence
        # class 0 : Person 
        video_sequence = model(video_dir,stream = True,
                                classes = [0], iou = 0.7, conf = 0.25,       
                               )  
        #max_frame_idx = len(os.listdir(video_dir))

        results = []
        for frame_idx,frame in enumerate(video_sequence,1):
            #print(f"Frame {frame_idx:05d}/{max_frame_idx:05d}") 
            boxes = frame.boxes
            confs  = torch.Tensor.numpy(boxes.conf)
            # ltwh is the MOT format, boxes doesn't support immediate translation
            bboxes_xywh = torch.Tensor.numpy(boxes.xywh)
            bboxes_ltwh = xywh2ltwh(bboxes_xywh)
            
            # save the bounding boxes and the confidence values in MOT16 format
            for bbox,conf in zip(bboxes_ltwh,confs):
                results.append([frame_idx, -1, bbox[0], bbox[1], bbox[2], bbox[3], conf, -1,-1,-1]) 

            if save_images:
                frame.save(filename = f"YOLO\\bbox_images\\{sequence}\\{frame_idx:05d}.jpg")
                

        with open(output_file, "w") as f:
            for row in results:
                str_row = ','.join(str(item) for item in row)
                print(str_row, file=f)


def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid True/False choice")
    else:
        return (input_string == "True")

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Create YOLOv11 Detections.")
    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge directory (train or test).",
        default = "MOT16\\train")
    parser.add_argument(
        "--output_dir", help="Output directory. Will be created if it does not exist.",
        default="YOLO"
    )
    parser.add_argument(
        "--save_images", help="Save the annotated images of the video sequence, with the detection boxes created by the YOLO model."
        "Attention: This can need a lot of disk space.",
        default=False, type=bool_string
    )
    parser.add_argument(
        "--img_out_dir", help="Path to the saved annotated images, if that option is used.",
        default="YOLO\\bbox_images"
    )
    parser.add_argument(
        "--visualize", help="Show annotated results realtime. WIP!",
        default=False, type=bool_string
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    yolo_detections_MOT16(args.mot_dir, args.output_dir, args.save_images, args.img_out_dir ,args.visualize)

