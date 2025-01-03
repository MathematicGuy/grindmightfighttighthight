import logging
import os
from typing import Dict, List, Tuple
import cv2 as cv
import numpy as np
import torch
import yaml
from fastapi import HTTPException
import supervision as sv
from utils.calculate_missed_corners import calculate_missing_with_homography

try:
    # Attempt import for FastAPI (relative import)
    from utils.NMS import non_max_suppression
    print("Imported NMS from utils.NMS")
except ImportError:
    try:
        # Attempt import for standard Python (direct import if NMS.py is in same folder)
        from NMS import non_max_suppression
        print("Imported NMS from NMS")
    except ImportError:
        # If both fail, raise an error or handle it as needed.
        print("Could not import NMS module from either utils.NMS or NMS")
        raise  # Or handle more gracefully


# Load configuration
try:
    config_path = os.path.join(os.path.dirname(__file__), '../corners-config.yaml')
    with open(config_path) as file:
        corners_config = yaml.safe_load(file)
except FileNotFoundError:
    print("config.yaml not found. Please make sure the config file is available")
    exit() # or handle the situation depending on your logic
except yaml.YAMLError as e:
    print(f"Error loading config.yaml: {e}")
    exit()


def get_center_point(box): #? calc each bbox center point.
    """Calculates the center point of a bounding box.
    Args:
         box (list) Bounding box coordinates [x1, y1, x2, y2].
    Returns:
        tuple: Center point (x, y).
    """

    xmin, ymin, xmax, ymax = box
    return (xmax + xmin) // 2, (ymax + ymin) // 2 # center point of each bboxes


def perspective_transformation(image, src_points, dst_points):
    """
        Apply perspective transformation to the input image. Basically transform input image to the destination points
        image: input image
        src_points: source points (4, 2) represent 4 corners of the bounding box
        dst_points: destination points (4, 2) - custom values set by user
    """
    try:
        output_width = int(dst_points[:, 0].max())
        output_height = int(dst_points[:, 1].max())

        matrix = cv.getPerspectiveTransform(src_points, dst_points)
        destination_coord = cv.warpPerspective(image, matrix, (output_width, output_height), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
        
        return destination_coord
    except Exception as e:
        logging.error(f"Error during perspective transform: {e}")
        raise


#? Save Annotation After or Before reconstruct missing corners Step 
def annotate_image_with_corners(image: np.ndarray, label_boxes: Dict[str, Tuple[int, int]], path: str, image_name, debug: bool = False) -> np.ndarray:
    """Annotates the image with detected or reconstructed corners."""
    image_copy = image.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Red, Green, Blue, Yellow
    labels = ['topleft', 'topright', 'botleft', 'botright']

    for i, label in enumerate(labels):
        if label in label_boxes:
            point = tuple(np.int32(label_boxes[label]))
            cv.circle(image_copy, point, 5, colors[i], -1)
            cv.putText(image_copy, label, (point[0] + 10, point[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    if debug:
        # Save the annotated image to the detect folder
        detect_folder = path
        os.makedirs(detect_folder, exist_ok=True)
        annotated_image_path = f'{detect_folder}/annotated_{image_name}.jpg'
        cv.imwrite(annotated_image_path, image_copy)

    return image_copy


def crop_image(preprocessed_image: np.ndarray, detections: List[List[float]], image_name: str, debug: bool = False) -> Tuple[np.ndarray, np.ndarray, Dict[str,Tuple[int, int]]]:
    """
        Crops the image using perspective transformation based on detected bounding boxes.
        Args:
            preprocessed_image (np.ndarray): The input image.
            detections (list): List of detections, where each detection is a list:
                            [class_id, confidence, x1, y1, x2, y2].

            debug: flag to visualize reconstruct corners

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Annotated image, cropped image, label_boxes.
    """
    valid = True #? flag fail to detect image

    image_height, image_width = preprocessed_image.shape[:2]

    # Filter predictions using NMS (assumes `non_max_suppression` function defined previously)
    labels = corners_config['names'] # ['botleft', 'botright', 'topleft', 'topright']

    nms_list = non_max_suppression(detections, iou_threshold=0.5, threshold=0.4)
    bboxes = sorted(nms_list, key=lambda x: int(x[0])) # Sort by class_id
    # print(f'sorted bboxes: {bboxes}\n')
    final_boxes = [box[2:] for box in bboxes] # get coordinate from 0 to 3

    # Use get_center_point only for missing corners calculation
    final_points = list(map(lambda box: get_center_point(np.array(box).astype(int)), final_boxes))
    label_boxes = dict(zip(labels, final_points))
    reconstruct_label_boxes = label_boxes.copy()
    
    #? Check if any corners are missing
    if len(label_boxes) == 3: 
        print('Less than 4 boxes detected, Reconstructing missing corners...')
        reconstruct_label_boxes, valid = calculate_missing_with_homography(reconstruct_label_boxes, image_width, image_height)
        debug = True

        logging.info(f"Reconstructed missing corners: {reconstruct_label_boxes}")

    #? Check if reconstruct corners fail
    if valid == False:
        # Save the preprocessed image to the fail_to_detect folder
        fail_dir = "validation/fail_to_detect"
        os.makedirs(fail_dir, exist_ok=True)
        fail_path = f'{fail_dir}/fail_{os.path.splitext(image_name)[0]}.jpg'
        cv.imwrite(fail_path, preprocessed_image)
        logging.error(f"Failed to detect and reconstruct 4 corners, saving preprocessed image to {fail_path}")
        
        raise HTTPException(status_code=400, detail="Failed to detect and reconstruct 4 corners")

    # Annotate image before reconstruction
    annotate_image_with_corners(preprocessed_image, label_boxes, "utils/detect/before_reconstruct/", image_name, debug=debug)

    # Annotate image after reconstruction
    image_annotation = annotate_image_with_corners(preprocessed_image, reconstruct_label_boxes,  "utils/detect/after_reconstruct/", image_name, debug=debug)
    
    #? note: check for image size
    # Use the original bounding box coordinates for perspective transformation
    source_points = np.float32([
        reconstruct_label_boxes['topleft'],
        reconstruct_label_boxes['topright'],
        reconstruct_label_boxes['botleft'],
        reconstruct_label_boxes['botright'],
    ])
    
    #? reconstruct_label_boxes: {'botleft': (34, 417), 'botright': (424, 413), 'topleft': (42, 182), 'topright': (425, 184)}
    print('Reconstructed label boxes:', reconstruct_label_boxes)
    
        
    dest_points = np.float32(
       [[0,0], [640,0], [0,640], [640,640]]
    )

    cropped_image = perspective_transformation(preprocessed_image, source_points, dest_points)
    
    # Return the reconstructed label boxes, not the center points
    return image_annotation, cropped_image, reconstruct_label_boxes


if __name__ == '__main__':
    # Prediction bounding boxes as (class ID, confidence, x1, y1, x2, y2)
    # detections = torch.tensor([
    #     [2, 0.720081090927124, 33.40931, 242.45248, 70.53368, 282.84384],   # Box 2 (topleft)
    #     [3, 0.6698232889175415, 439.35947, 230.78842, 477.95145, 283.24826], # Box 3 (topright)
    #     [1, 0.5960968136787415, 443.94025, 484.69257, 489.34003, 536.5535],  # Box 1 (bottomright)
    #     [0, 0.44452500343322754, 21.572742, 492.26105, 61.92349, 544.39557], # Box 0 (bottomleft)
    #     [0, 0.29470255970954895, 427.79895, 0.0, 468.2058, 33.488106]
    # ])
    
    detections = torch.tensor([
        [1, 0.7302447557449341, 520.62744, 548.1746, 571.3467, 594.3139],
        [2, 0.6837620735168457, 63.045853, 53.913837, 103.93451, 94.56938],
        [3, 0.39823731780052185, 47.359848, 552.9736, 88.17268, 594.63995]
    ])

    # Load the image
    image_path = 'validation/fail_to_detect/testc.jpg'
    image = cv.imread(image_path)
    crop_image(image, detections)


