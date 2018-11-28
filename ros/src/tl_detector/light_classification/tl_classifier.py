from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import cv2
import time



class TLClassifier(object):
    def __init__(self):
        self.init = False
        #Define model location
        model_name = 'ssd_mobilenet' #ssd_inception, ssd_mobilenet
        model_path = '/home/workspace/CarND-Capstone/ros/src/tl_detector/light_classification/frozen_%s/frozen_inference_graph.pb'%model_name

        #create object detector
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(model_path, 'rb') as fid:        
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
               
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as self.sess:
                # Define input tensor for detection_graph
                self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

                # Define output tensor for detection graph
                # score returns confidence in detection results
                self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                # Class returns type of detection
                self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                
                self.init = True
                
    def model_loaded(self):
        return self.init
    
    def get_classification(self, image):
           """Determines the color of the traffic light in the image

           Args:
               image (cv::Mat): image containing the traffic light

           Returns:
               int: ID of traffic light color (specified in styx_msgs/TrafficLight)

           """
           tnow = int(time.time()) % 10;
           if (tnow > 5):
                return TrafficLight.GREEN
           else:
                return TrafficLight.RED
            
           #image comes from cv_bridge as BGR image but networks have been trained with RGB images
           image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
           #expand iamge to 4D array to pass to NN
           image_np_expanded = np.expand_dims(image, axis=0)

           #run image through NN
           (scores,classes) = self.sess.run(
                 [self.detection_scores, self.detection_classes],
                 feed_dict={self.image_tensor: image_np_expanded})

           #remove redundant dimensions
           scores = np.squeeze(scores)
           classes = np.squeeze(classes).astype(np.int32)

           
           #if a light is detected with over 30% confidence use prediction as output
           #otherwise return trafficLight.UNKNOWN
           if scores[0]>0.1:
               prediction = classes[0]
           else:
               prediction = 4
            
           #remap classes to match simulator ground truth
           if prediction == 1:
               return TrafficLight.GREEN
           elif prediction == 2:
               return TrafficLight.RED
           elif prediction == 3:
               return TrafficLight.YELLOW
           else:
               return TrafficLight.UNKNOWN 
