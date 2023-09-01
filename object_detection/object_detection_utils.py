from PIL import Image, ImageDraw, ImageFont
import copy
import torch


def detect_objects(image, detector_model, threshold, categories):
    detector_model.eval()
    with torch.no_grad():
        predictions = detector_model([image])[0]
    
    boxes = predictions["boxes"]
    scores = predictions["scores"]
    labels = predictions["labels"]
    
    # Get all the boxes above the threshold
    mask = scores >= threshold
    boxes_filtered = boxes[mask]
    
    # Get the names of the categories above the threshold
    labels_filtered = labels[mask]
    categories_filtered = {categories[label] for label in labels_filtered}
    
    # Get only the scores above the threshold
    scores_filtered = scores[mask]
    
    return boxes_filtered, scores_filtered, labels_filtered, categories_filtered


def draw_bounding_boxes(image, boxes, classes, labels, scores, colors, normalized_coordinates, add_text):
    font = ImageFont.truetype("arial.ttf", 15)
    image_with_bb = copy.deepcopy(image)
    painter = ImageDraw.Draw(image_with_bb)
    
    # Enumerate all the pairs (bounding_box, bounding box's label) in the image
    for i, (box, label) in enumerate(zip(boxes, labels)):        
        # get the color associated with the class/label of the bounding box
        #color = tuple(colors[label].astype(np.int32))
        color = colors[label]
        x_min, y_min, x_max, y_max = box
        
        # if the coordinates are given as values in [0,1]
        if normalized_coordinates:
            width, height = image.size
            x_min *= width
            y_min *= height
            x_max *= width
            y_max *= height

        coord_bb = [x_min, y_min, x_max, y_max]
        painter.rectangle(coord_bb, outline=color, width=4)
        
        # The label will be written in the bottom part of the bounding box
        # So we write inside the bounding box, in the bottom part
        if add_text:            
            #class_obj = classes[i]
            class_obj = classes[label]
            score = scores[i]
            # Visualize both the name of the clas/label of the object, and the probabiliy that the label is correct
            text_in_box = f'{class_obj}-{score:.2f}'
            # The text bottom will coincide withe the bottom of the bounding box
            text_bottom = y_max
            text_width, text_height = font.getsize(text_in_box)
            margin = np.ceil(0.05 * text_height)
            # Draw the rectangle that will cointain the text/label
            painter.rectangle([(x_min, text_bottom - text_height - 2 * margin), 
                               (x_min + text_width, text_bottom)], fill=color)
            
            # Write the label inside the rectangle
            painter.text((x_min + margin, text_bottom - text_height - margin), 
                         text_in_box, fill='black', font=font)

    return image_with_bb
