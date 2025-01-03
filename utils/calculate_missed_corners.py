import numpy as np


def get_center_point(box):
    """Calculates the center point of a bounding box."""
    xmin, ymin, xmax, ymax = box
    return (xmax + xmin) // 2, (ymax + ymin) // 2


def find_miss_corner(coordinate_dict):
    labels = ['topleft', 'topright', 'botleft', 'botright']
    for i, label in enumerate(labels):
        if label not in coordinate_dict:
            return i
    return False


# def calculate_missing_with_homography(coordinate_dict,  image_width, image_height):
#     """
#         Calculates the coordinates of a missing corner based on the positions of the other three corners.
#         Adds input validation and output clamping.
        
#         Args:
#             coordinate_dict (dict): A dictionary containing the coordinates of the detected corners.
#                                     The keys should be 'topleft', 'topright', 'botleft', and 'botright'.
#                                     The values should be tuples representing (x, y) coordinates.

#         Returns:
#             dict: A dictionary containing the coordinates of all four corners, including the estimated one.
#     """
#     thresh = 0
#     valid = True #? flag fail to detect image
#     index = find_miss_corner(coordinate_dict)

#     # Input Validation: Check for basic consistency in input corners
#     if len(coordinate_dict) == 3:
#         coords = list(coordinate_dict.values())
        
#         # Assuming typical ID card orientation: topleft.x < topright.x, botleft.x < botright.x, topleft.y < botleft.y, topright.y < botright.y
#         if 'topleft' in coordinate_dict and 'topright' in coordinate_dict and coordinate_dict['topleft'][0] > coordinate_dict['topright'][0]:
#             print("Warning: Potential issue with topleft and topright x-coordinates.")
#             valid = False

#         if 'botleft' in coordinate_dict and 'botright' in coordinate_dict and coordinate_dict['botleft'][0] > coordinate_dict['botright'][0]:
#             print("Warning: Potential issue with botleft and botright x-coordinates.")
#             valid = False
            
#         if 'topleft' in coordinate_dict and 'botleft' in coordinate_dict and coordinate_dict['topleft'][1] > coordinate_dict['botleft'][1]:
#             print("Warning: Potential issue with topleft and botleft y-coordinates.")
#             valid = False
            
#         if 'topright' in coordinate_dict and 'botright' in coordinate_dict and coordinate_dict['topright'][1] > coordinate_dict['botright'][1]:
#             print("Warning: Potential issue with topright and botright y-coordinates.")
#             valid = False

    
#     # calculate missed corner coordinate
#     if index == 0:  # "topleft"
#         midpoint = np.array(coordinate_dict['topright']) + np.array(coordinate_dict['botleft'])
#         estimated_x = 2 * (midpoint[0] / 2) - coordinate_dict['botright'][0] - thresh
#         estimated_y = 2 * (midpoint[1] / 2) - coordinate_dict['botright'][1] - thresh
#         coordinate_dict['topleft'] = (estimated_x, estimated_y)
        
#     elif index == 1:  # "topright"
#         midpoint = np.array(coordinate_dict['topleft']) + np.array(coordinate_dict['botright'])
#         estimated_x = 2 * (midpoint[0] / 2) - coordinate_dict['botleft'][0] - thresh
#         estimated_y = 2 * (midpoint[1] / 2) - coordinate_dict['botleft'][1] - thresh
#         coordinate_dict['topright'] = (estimated_x, estimated_y)
        
#     elif index == 2:  # "botleft"
#         midpoint = np.array(coordinate_dict['topleft']) + np.array(coordinate_dict['botright'])
#         estimated_x = 2 * (midpoint[0] / 2) - coordinate_dict['topright'][0] - thresh
#         estimated_y = 2 * (midpoint[1] / 2) - coordinate_dict['topright'][1] - thresh
#         coordinate_dict['botleft'] = (estimated_x, estimated_y)
        
#     elif index == 3:  # "botright"
#         midpoint = np.array(coordinate_dict['botleft']) + np.array(coordinate_dict['topright'])
#         estimated_x = 2 * (midpoint[0] / 2) - coordinate_dict['topleft'][0] - thresh
#         estimated_y = 2 * (midpoint[1] / 2) - coordinate_dict['topleft'][1] - thresh

#         coordinate_dict['botright'] = (estimated_x, estimated_y)

#     # Output Clamping: Ensure calculated coordinates are within image boundaries
#     for key, (x, y) in coordinate_dict.items():
#         coordinate_dict[key] = (
#             int(np.clip(x, 0, image_width)),
#             int(np.clip(y, 0, image_height)),
#         )

#     return coordinate_dict, valid

    
    
def calculate_missing_with_homography(coordinate_dict, image_width, image_height):
    """
    Estimates the missing corner of a bounding box using the positions of the other three corners.
    
    Args:
        coordinate_dict (dict): Dictionary with detected corners ('topleft', 'topright', 'botleft', 'botright').
                                Each key maps to a tuple of (x, y) coordinates.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
    
    Returns:
        dict: Complete dictionary with all four corners, including the estimated one.
        bool: Flag indicating whether the calculation is valid.
    """
    # Check for valid input
    if len(coordinate_dict) != 3:
        raise ValueError("Exactly three corners must be provided.")
    
    # Detect missing corner
    index = find_miss_corner(coordinate_dict)
    if index is None:
        return coordinate_dict, True  # No missing corner
    
    labels = ['topleft', 'topright', 'botleft', 'botright']
    missing_label = labels[index]
    
    # Input validation
    valid = True
    if 'topleft' in coordinate_dict and 'topright' in coordinate_dict:
        valid &= coordinate_dict['topleft'][0] < coordinate_dict['topright'][0]
    if 'botleft' in coordinate_dict and 'botright' in coordinate_dict:
        valid &= coordinate_dict['botleft'][0] < coordinate_dict['botright'][0]
    if 'topleft' in coordinate_dict and 'botleft' in coordinate_dict:
        valid &= coordinate_dict['topleft'][1] < coordinate_dict['botleft'][1]
    if 'topright' in coordinate_dict and 'botright' in coordinate_dict:
        valid &= coordinate_dict['topright'][1] < coordinate_dict['botright'][1]
    
    if not valid:
        print("Warning: Inconsistent corner coordinates detected.")
    
    # Estimate missing corner
    if missing_label == 'topleft':
        midpoint = np.array(coordinate_dict['topright']) + np.array(coordinate_dict['botleft'])
        estimated_x = 2 * (midpoint[0] / 2) - coordinate_dict['botright'][0]
        estimated_y = 2 * (midpoint[1] / 2) - coordinate_dict['botright'][1]
        coordinate_dict['topleft'] = (estimated_x, estimated_y)
    elif missing_label == 'topright':
        midpoint = np.array(coordinate_dict['topleft']) + np.array(coordinate_dict['botright'])
        estimated_x = 2 * (midpoint[0] / 2) - coordinate_dict['botleft'][0]
        estimated_y = 2 * (midpoint[1] / 2) - coordinate_dict['botleft'][1]
        coordinate_dict['topright'] = (estimated_x, estimated_y)
    elif missing_label == 'botleft':
        midpoint = np.array(coordinate_dict['topleft']) + np.array(coordinate_dict['botright'])
        estimated_x = 2 * (midpoint[0] / 2) - coordinate_dict['topright'][0]
        estimated_y = 2 * (midpoint[1] / 2) - coordinate_dict['topright'][1]
        coordinate_dict['botleft'] = (estimated_x, estimated_y)
    elif missing_label == 'botright':
        midpoint = np.array(coordinate_dict['botleft']) + np.array(coordinate_dict['topright'])
        estimated_x = 2 * (midpoint[0] / 2) - coordinate_dict['topleft'][0]
        estimated_y = 2 * (midpoint[1] / 2) - coordinate_dict['topleft'][1]
        coordinate_dict['botright'] = (estimated_x, estimated_y)

    # Clamp output to image boundaries
    for key, (x, y) in coordinate_dict.items():
        coordinate_dict[key] = (
            int(np.clip(x, 0, image_width)),
            int(np.clip(y, 0, image_height)),
        )

    return coordinate_dict, valid