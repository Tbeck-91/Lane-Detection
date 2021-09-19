import os
import pathlib
import time

import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
from numpy.core.numeric import zeros_like
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageColor
import base64
import cv2

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

#class ID constants
ROAD_ID = 1
DRIVEWAY_ID = 2
PARKING_ID = 3

# load a base64 encoded image into a numpy array
def load_image_into_numpy_array(img_data):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: the file path to the image

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  image = Image.open(BytesIO(base64.b64decode(img_data)))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# Load the COCO Label Map
category_index = {
    ROAD_ID: {'id': ROAD_ID, 'name': 'Road'},
    DRIVEWAY_ID: {'id': DRIVEWAY_ID, 'name': 'Driveway'},
    PARKING_ID: {'id': PARKING_ID, 'name': 'Parking'},
}

configs = config_util.get_configs_from_pipeline_file(r"./mask_rcnn.config")


def line_overlap (line_arr, line, margin):
  x1,y1,x2,y2=line
  for check_line in line_arr:
    check_x1,check_y1,check_x2,check_y2=check_line
    x_overlap = x1 >= (check_x1 - margin) and x2 <= (check_x2 + margin)
    y_overlap = y1 >= (check_y1 - margin) and y2 <= (check_y2 + margin)
    if (x_overlap and y_overlap):
      return True
  return False
############## LOAD MODEL #####################

start_time = time.time()
tf.keras.backend.clear_session()
model = tf.saved_model.load(r"./saved_model")
end_time = time.time()
elapsed_time = end_time - start_time

def run_detection(img_data):

  image_np = load_image_into_numpy_array(image_data)
  output_dict = {"Roads": []}

  input_tensor = tf.convert_to_tensor(image_np)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis, ...]
  # input_tensor = np.expand_dims(image_np, 0)
  detections = model(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.

  num_detections = int(detections.pop('num_detections'))
  #we need to make sure we're only getting detections for certain keys
  need_detection_key = ['detection_classes','detection_boxes','detection_masks','detection_scores']
  detections = {key: detections[key][0, :num_detections].numpy()
                for key in need_detection_key}
  detections['num_detections'] = num_detections
  # detection_classes should be ints.
  detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

  image_np_with_detections = image_np.copy()
  image_np_with_edge_detection = image_np.copy()
  img_height, img_width = image_np.shape[:2]

  for i in range(num_detections):
    class_id = detections['detection_classes'][i]
    if detections['detection_scores'][i] >= 0.99 and class_id == ROAD_ID:      
      ymin, xmin, ymax, xmax = detections['detection_boxes'][i]
      mask = detections['detection_masks'][i]  

      #scale normalized coordinates to image
      ymin *= img_height
      ymax *= img_height
      xmin *= img_width
      xmax *= img_width      
      
      #get width and height of box
      box_width = xmax - xmin
      box_height = ymax -  ymin
      mask = cv2.resize(mask, (int(box_width), int(box_height)))
      mask = (mask >= 0.99)
      contours, hierarchy = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      contour = contours[0]
      contour_len = cv2.arcLength(contour, True)
      contour = cv2.approxPolyDP(contour, 0.02*contour_len, 5, True)
      contour.reshape(2,2)
      contour = contour + (xmin, ymin)

      roi = image_np_with_detections[int(ymin):int(ymax), int(xmin):int(xmax)]      
      canny = cv2.Canny(gray,100,150)

      #if rows are off after resizing, pad mask
      if(mask.shape[1] != canny.shape[1]):
        difference = canny.shape[1] - mask.shape[1]
        add_arr = np.full((mask.shape[0], difference), False)
        mask = np.append(mask, add_arr, axis=1)

      if mask.shape[0] != canny.shape[0]:
        difference = canny.shape[0] - mask.shape[0]
        print(f"difference {difference}")
        add_arr = np.full((difference, mask.shape[1]), False)
        print(f"add_arr.shape {add_arr.shape} mask.shape {mask.shape}")
        mask = np.append(mask, add_arr, axis=0)

      print(f"Mask Shape: {mask.shape} - Canny Shape: {canny.shape}")

      visMask = (mask * 255).astype("uint8") 
      masked_image=cv2.bitwise_and(canny,visMask) 
      roi_image_with_lines = roi.copy()
      lines = cv2.HoughLinesP(masked_image, 0.8, np.pi/180, 90, np.array([]), minLineLength=50, maxLineGap=200)
      original_image = image_np.copy()
      road_contour = zeros_like(image_np)
      original_image_with_road_contours = image_np.copy()
      cv2.drawContours(original_image_with_road_contours, contours, -1, (0, 255, 0), 3)

      test = np.zeros_like(roi_image_with_lines)
  
      cleaned_lines = []
      for line in lines:
        for segment in line:
          if not line_overlap(cleaned_lines, segment, 10):
            cleaned_lines.append(line[0])

      road_dict = {"Road": contour, "Lanes": cleaned_lines}
      output_dict["Roads"].append(road_dict)
    #end if
  #end for
  return output_dict
  