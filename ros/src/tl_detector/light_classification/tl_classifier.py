from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import cv2
import sys

# imports from the object detection module
sys.path.append('./models/research')
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class TLClassifier(object):
    def __init__(self):
        NUM_CLASSES = 4
        model_name = 'ssd_mobilenet' #ssd_inception, ssd_mobilenet
        model_path = '/home/workspace/CarND-Capstone/ros/src/tl_detector/light_classification/frozen_%s/frozen_inference_graph.pb'%model_name
        PATH_TO_LABELS = '/home/workspace/CarND-Capstone/ros/src/tl_detector/light_classification/label_map.pbtxt'
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, 
                                                                    max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        detection_graph = tf.Graph()

        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(model_path, 'rb') as fid:        
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
               
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as self.sess:
                # Definite input and output Tensors for detection_graph
                self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

                # Each box represents a part of the image where a particular object was detected.
                self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    def get_classification(self, image):
           """Determines the color of the traffic light in the image

           Args:
               image (cv::Mat): image containing the traffic light

           Returns:
               int: ID of traffic light color (specified in styx_msgs/TrafficLight)

           """
           image_np_expanded = np.expand_dims(image, axis=0)

           (boxes,scores,classes,num_detections) = self.sess.run(
                 [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                 feed_dict={self.image_tensor: image_np_expanded})

           boxes = np.squeeze(boxes)
           scores = np.squeeze(scores)
           classes = np.squeeze(classes).astype(np.int32)

           plot = True
           min_score_thresh = .30

           vis_util.visualize_boxes_and_labels_on_image_array(
                image, boxes, classes, scores,
                self.category_index,
                min_score_thresh=min_score_thresh,
                use_normalized_coordinates=True,
                line_thickness=3)
            
           prediction = classes[0]#int(np.around(np.median(classes[:3])))
           '''
           i=0
           h,w,_=image.shape
           green = 0
           red = 0
           yellow = 0
           other = 0
           for box in boxes:
               #print(box)
                    
               if scores[i]>0.05:
                
                   x1,y1,x2,y2 = box
                   x1 = int(h*x1)
                   y1 = int(w*y1)
                   x2 = int(h*x2)
                   y2 = int(w*y2)
                   light_height = y2-y1
                   light_width = x2-x1
                   print("I: ",i)
                   print("class: ",classes[i])
                   print("Score: ",scores[i])
                   print("light height: ",light_height)
                   print("light width: ",light_width)

                   font = cv2.FONT_HERSHEY_SIMPLEX
                   cv2.putText(image,"score: " + str(scores[i]), (y1,x1), font, 0.5, (255,255,255), 2)
                   cv2.putText(image,"bbox height: " + str(light_height), (y1,x1-15), font, 0.5, (255,255,255), 2)
                   cv2.putText(image,"bbox width: " + str(light_width), (y1,x1-30), font, 0.5, (255,255,255), 2)

                   if classes[i] == 1:
                       green += scores[i]
                       image = cv2.rectangle(image,(y1,x1),(y2,x2),(0,255,0))
                   elif classes[i] == 2:
                       red += scores[i]
                       image = cv2.rectangle(image,(y1,x1),(y2,x2),(0,0,255))
                   elif classes[i] == 1:
                       yellow += scores[i]
                       image = cv2.rectangle(image,(y1,x1),(y2,x2),(0,255,255))
                   elif classes[i] == 1:
                       other += scores[i]
                       image = cv2.rectangle(image,(y1,x1),(y2,x2),(255,0,0))
               i+=1

           print(green)
           print(red)
           print(yellow)
           '''
           if prediction == 1:
               print("GREEN")
               return 2, image
           elif prediction == 2:
               print("RED")
               return 0, image
           elif prediction == 4:
               print("YELLOW")
               return 1, image
           else:
               print("UNKOWN")
               return 4, image
