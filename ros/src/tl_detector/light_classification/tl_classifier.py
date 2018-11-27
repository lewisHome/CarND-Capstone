from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import cv2
import sys

class TLClassifier(object):
    def __init__(self):
        
        model_name = 'ssd_mobilenet' #ssd_inception, ssd_mobilenet
        model_path = '/home/workspace/CarND-Capstone/ros/src/tl_detector/light_classification/frozen_%s/frozen_inference_graph.pb'%model_name

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
           image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
           image_np_expanded = np.expand_dims(image, axis=0)

           (_,scores,classes,_) = self.sess.run(
                 [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                 feed_dict={self.image_tensor: image_np_expanded})

           scores = np.squeeze(scores)
           classes = np.squeeze(classes).astype(np.int32)

           if scores[0]>0.1:
               prediction = classes[0]
           
           #remap classes to match simulator ground truth
           if prediction == 1:
               return 2
           elif prediction == 2:
               return 0
           elif prediction == 3:
               return 1
           else:
               return 4
