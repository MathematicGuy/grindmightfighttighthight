import logging
import torch

try:
    # Attempt import for FastAPI (relative import)
    from utils.IoU import intersection_over_union
    logging.info("Imported NMS from utils.NMS")
except ImportError:
    try:
        # Attempt import for standard Python (direct import if NMS.py is in same folder)
        from IoU import intersection_over_union
        logging.info("Imported NMS from NMS")
    except ImportError:
        # If both fail, raise an error or handle it as needed.
        logging.error("Could not import NMS module from either utils.NMS or NMS")
        raise  # Or handle more gracefully

#! Take input of each class. Meaning all bboxes is in the same class
def non_max_suppression(bboxes, # box in bboxes: [box_id, confidence, x1, y1, x2, y2]
    iou_threshold, # IoU threshold
    threshold,
    box_format="corners"
):
    """
    Apply Non-Maximum Suppression (NMS) to bounding boxes
        bboxes: bounding boxes predictions (N, 5): [[box_id, confidence, x1, y1, x2, y2]]
        iou_threshold: intersection over union threshold to eliminate overlapping boxes of the same class
        threshold: confidence threshold to keep boxes of different class by a threshold
        box_format: coordinate format  

        While BoundingBoxes:
            Take out the highest confidence box (box1)
            For each class:
                Remove all other boxes with IoU(box1, box_i) < threshold
    
        note:
        + confidence say how sure the model prediction is
        + IoU say how close the prediction bounding box to the label bounding box is
    """
    
    # assert type(bboxes) == list 
    
    #? filter all bboxes by confidence threshold
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes.sort(key=lambda box:box[1], reverse=True) # sort from max to min by confidence score
    
    #! Main Motive: Extract the best bounding boxes of each class  
    #! Method: Remove all bouding boxes that are too close to the best bounding box using best_bboxes as a anchor value (i.e. bboxes work around best_bbox)
    #* In other word, remove all that is too close, retain all is different in class
    
    #? Create a list to store best_bboxes called: after_nms_list = [] 
    after_nms_list = []
    
    #! For each class in bboxes, extract (remove) best_bboxes and remove all bboxes close to best_bbox by a threshold -> Method: Update bboxes by Condition 
    #? Update bboxes: 
    while bboxes:
        #? Extract best_bboxes from bboxes (since we have sort max to min by confidence score)
        best_bbox = bboxes.pop(0)
        #? Update bboxes and only retain bboxes that are not "in the same class" or "close" to best_bbox
        bboxes = [ 
            #? if (box and best_bboxes not in the same class) or (IoU(box, best_bbox) < iou_threshold) then keep box  
            box for box in bboxes 
            if box[0] != best_bbox[0] or            
            intersection_over_union(
                torch.tensor(box[2:]).unsqueeze(0),
                torch.tensor(best_bbox[2:]).unsqueeze(0),
                box_format=box_format
            ) < iou_threshold # if 2 boxes of the same class are too close, remove the box with lower confidence score
        ]

        #? save best_bboxes to after_nms_list 
        after_nms_list.append(best_bbox)
    
    return after_nms_list

#? Like NMS but ignore class_id = 0
def non_max_suppression_roi(bboxes, # box in bboxes: [box_id, confidence, x1, y1, x2, y2]
    iou_threshold, # IoU threshold
    threshold,
    box_format="corners"
):
    """
    Apply Non-Maximum Suppression (NMS) to bounding boxes
        bboxes: bounding boxes predictions (N, 5): [[box_id, confidence, x1, y1, x2, y2]]
        iou_threshold: intersection over union threshold to eliminate overlapping boxes of the same class
        threshold: confidence threshold to keep boxes of different class by a threshold
        box_format: coordinate format  

        While BoundingBoxes:
            Take out the highest confidence box (box1)
            For each class:
                Remove all other boxes with IoU(box1, box_i) < threshold
    
        note:
        + confidence say how sure the model prediction is
        + IoU say how close the prediction bounding box to the label bounding box is
    """
    
    # assert type(bboxes) == list 
    
    #? filter all bboxes by confidence threshold
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes.sort(key=lambda box:box[1], reverse=True) # sort from max to min by confidence score
    
    #! Main Motive: Extract the best bounding boxes of each class  
    #! Method: Remove all bouding boxes that are too close to the best bounding box using best_bboxes as a anchor value (i.e. bboxes work around best_bbox)
    #* In other word, remove all that is too close, retain all is different in class
    
    #? Create a list to store best_bboxes called: after_nms_list = [] 
    after_nms_list = []
    
    # Keep track of already selected box_ids
    selected_box_ids = set()
    
    #! For each class in bboxes, extract (remove) best_bboxes and remove all bboxes close to best_bbox by a threshold -> Method: Update bboxes by Condition 
    #? Update bboxes: 
    while bboxes:
        #? Extract best_bboxes from bboxes (since we have sort max to min by confidence score)
        best_bbox = bboxes.pop(0)
        
        # Skip if box_id = 0
        if best_bbox[0] == 0:
            after_nms_list.append(best_bbox)
            continue
        
        # Skip if box_id has already been selected
        if best_bbox[0] in selected_box_ids:
            continue

        #? Update bboxes and only retain bboxes that are not "in the same class" or "close" to best_bbox
        bboxes = [
            #? if (box and best_bboxes not in the same class) or (IoU(box, best_bbox) < iou_threshold) then keep box
            box for box in bboxes
            if box[0] != best_bbox[0] or
            intersection_over_union(
                torch.tensor(box[2:]).unsqueeze(0),
                torch.tensor(best_bbox[2:]).unsqueeze(0),
                box_format=box_format
            ) < iou_threshold  # if 2 boxes of the same class are too close, remove the box with lower confidence score
        ]

        #? save best_bboxes to after_nms_list
        after_nms_list.append(best_bbox)
        # Add the selected box_id to the set
        selected_box_ids.add(best_bbox[0])
    
    return after_nms_list



if __name__ == '__main__':
    pred_boxes = torch.tensor([[2, 0.720081090927124, 33.40931, 242.45248, 70.53368, 282.84384],
                               [3, 0.6698232889175415, 439.35947, 230.78842, 477.95145, 283.24826],
                               [1, 0.5960968136787415, 443.94025, 484.69257, 489.34003, 536.5535],
                               [0, 0.44452500343322754, 21.572742, 492.26105, 61.92349, 544.39557],
                               [0, 0.29470255970954895, 427.79895, 0.0, 468.2058, 33.488106]])

    # Test the nms function
    nms_list = non_max_suppression(pred_boxes.tolist(), iou_threshold=0.6, threshold=0.4) 
    print("Remaining boxes after NMS:")
    for box in nms_list:
        print(box)    
     
'''output
[2.0, 0.720081090927124, 33.40930938720703, 242.45248413085938, 70.53368377685547, 282.8438415527344]
[3.0, 0.6698232889175415, 439.3594665527344, 230.78842163085938, 477.9514465332031, 283.2482604980469]
[1.0, 0.5960968136787415, 443.94024658203125, 484.69256591796875, 489.34002685546875, 536.5535278320312]
[0.0, 0.44452500343322754, 21.572742462158203, 492.26104736328125, 61.92348861694336, 544.3955688476562]
'''
    
'''iou output
# Class 0 boxes
box1 = torch.tensor([21.572742, 492.26105, 61.92349, 544.39557])
box2 = torch.tensor([427.79895, 0.0, 468.2058, 33.488106])

# Calculate IoU
iou = intersection_over_union(box1.unsqueeze(0), box2.unsqueeze(0))
print(f"IoU between the two class 0 boxes: {iou.item()}")

# output: IoU between the two class 0 boxes: 0.043826162815093994
'''