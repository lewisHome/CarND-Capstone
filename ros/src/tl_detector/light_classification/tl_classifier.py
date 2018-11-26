from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import cv2

class TLClassifier(object):
    def __init__(self):

        model_name = 'ssd_mobilenet' #ssd_inception, ssd_mobilenet
        model_path = '/home/workspace/CarND-Capstone/ros/src/tl_detector/light_classification/frozen_%s/frozen_inference_graph.pb'%model_name
        #PATH_TO_LABELS = 'label_map.pbtxt'

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
           classes = np.squeeze(classes)

           prediction = int(np.around(np.median(classes[:3])))

           i=0
           h,w,_=image.shape
           for box in boxes:
               #print(box)
               x1,y1,x2,y2 = box
               x1 = int(h*x1)
               y1 = int(w*y1)
               x2 = int(h*x2)
               y2 = int(w*y2)
               if classes[i] == 1:
                   image = cv2.rectangle(image,(y1,x1),(y2,x2),(0,255,0))
               elif classes[i] == 2:
                   image = cv2.rectangle(image,(y1,x1),(y2,x2),(0,0,255))
               elif classes[i] == 1:
                   image = cv2.rectangle(image,(y1,x1),(y2,x2),(0,255,255))
               elif classes[i] == 1:
                   image = cv2.rectangle(image,(y1,x1),(y2,x2),(255,0,0))
               i+=1
               if i ==3:
                    break
                    
           if prediction == 1:
               print("GREEN")
               return 2, image
           elif prediction == 2:
               print("RED")
               return 0, image
           elif prediction == 3:
               print("YELLOW")
               return 1, image
           else:
               print("UNKOWN")
               return 4, image
