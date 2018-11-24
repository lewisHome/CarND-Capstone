#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math
from scipy.spatial import KDTree
import numpy as np


import tf

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.got_waypoints = False
        
        self.camera_image = None
        self.lights = []
        self.lights_map = []
        self.got_lights_map=False
        self.look_ahead_distance = 100.0
        self.last_image_time = rospy.get_time()
        self.processing_image = False
        self.waypoint_tree = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoints_x = []
        self.waypoints_y = []


       
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb, queue_size=1, buff_size = 2*52428800)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.updateRate = 8
        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg


    def waypoints_cb(self, waypoints):
        # load base waypoints
        print("new waypoints")
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints = waypoints.waypoints
            self.got_waypoints = True
            # convert waypoints to (x,y) list
            self.waypoints_2d = [
                [
                    waypoint.pose.pose.position.x,
                    waypoint.pose.pose.position.y
                ] for waypoint in waypoints.waypoints
            ]
            maxdist = 0
            prev_x = -1
            prev_y = -1
            for waypoint in waypoints.waypoints:
                self.waypoints_x.append(waypoint.pose.pose.position.x);
                self.waypoints_y.append(waypoint.pose.pose.position.y);
                if prev_x >= 0:
                    x = waypoint.pose.pose.position.x - prev_x
                    y = waypoint.pose.pose.position.y - prev_y

                    dist = math.sqrt((x*x) + (y*y))
                    if dist > maxdist:
                        maxdist = dist
                prev_x = waypoint.pose.pose.position.x
                prev_y = waypoint.pose.pose.position.y
            # build KDTree
            self.waypoint_tree = KDTree(self.waypoints_2d)
            # for Highway map, maxdist = 2.6486
            print("Waypoints max distance between points = ",maxdist)
            
    def traffic_cb(self, msg):
        self.lights = msg.lights
            
    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        now_time = rospy.get_time()
        # limit image update to max 2 fps
        if (self.processing_image == False) and (now_time > (self.last_image_time + 0.5)):
            self.last_image_time = now_time

            self.has_image = True
            self.camera_image = msg
            self.processing_image = True
            light_wp, state = self.process_traffic_lights()
            self.processing_image = False
            '''
            Publish upcoming red lights at camera frequency.
            Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
            of times till we start using it. Otherwise the previous stable state is
            used.
            '''
            if self.state != state:
                self.state_count = 0
                self.state = state
            elif self.state_count >= STATE_COUNT_THRESHOLD:
                self.last_state = self.state
                light_wp = light_wp if state == TrafficLight.RED else -1
                self.last_wp = light_wp
                print("LIGHT WAY POINT: ",light_wp)
                self.upcoming_red_light_pub.publish(Int32(light_wp))
            else:
                self.upcoming_red_light_pub.publish(Int32(self.last_wp))
            self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        diff_x = 1e6
        diff_y = 1e6
        
        if self.waypoints != None:
            
            for point in range(len(self.waypoints)):
                
                x = abs (pose.position.x - self.waypoints[point].pose.pose.position.x)
                y = abs (pose.position.y - self.waypoints[point].pose.pose.position.y)
                
                if  x < diff_x and y < diff_y:
                    diff_x = x
                    diff_y = y
                    ind = point        
                    
            return ind
        else:
            return

    def get_closest_waypoint_idx(self, pose):
        x = pose.position.x
        y = pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        # check if closest is ahead or behind vehilcle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        # equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx


    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False
        if self.light_classifier == None:
            return

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
       
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
       
        if self.waypoints == None:
            return
        
        stop_x = 1e6
        stop_y = 1e6

        if stop_line_positions == None:
                print("stop_line_positions is None ")


        
        for stop_line_position in stop_line_positions:
            light_stop_pose = Pose()
            light_stop_pose.position.x = stop_line_position[0]
            light_stop_pose.position.y = stop_line_position[1]
            light_stop_wp = self.get_closest_waypoint_idx(light_stop_pose)

        if(self.pose):
            position_index = self.get_closest_waypoint_idx(self.pose.pose)
            car_position = (self.waypoints[position_index].pose.pose.position.x,
                            self.waypoints[position_index].pose.pose.position.y)
        #    print("car position",car_position)
        #TODO find the closest visible traffic light (if one exists)
        
        #look 100m forward
        q = [self.pose.pose.orientation.x, self.pose.pose.orientation.y, self.pose.pose.orientation.z, self.pose.pose.orientation.w]
        
        theta = tf.transformations.euler_from_quaternion(q)[2]

        x = self.pose.pose.position.x
        x_in_front = x  + 1 * math.cos(theta)
        y = self.pose.pose.position.y
        y_in_front = y + 1 * math.sin(theta)

        traffic_light_found = False
        light_state = TrafficLight.UNKNOWN        

        if self.lights == None:
                print("self.lights is None ")

        for light in self.lights:
            light_x = light.pose.pose.position.x
            light_y = light.pose.pose.position.y
            
            dist_to_light = (((light_x - x)**2 + (light_y - y)**2)**0.5)
            light_orient = (light_x - x + light_y - y)
            car_orient = x_in_front - x + y_in_front - y

            # prevent "referenced before assignment" error
            pred_state = -1

            if  dist_to_light < self.look_ahead_distance and \
                car_orient * light_orient > 1:

                state = light.state
                print("GROUND TRUTH STATE: ",state)
                traffic_light_found = True
                pred_state = self.get_light_state(light)

                light_stop_pose = Pose()
                light_stop_pose.position.x = light_x
                light_stop_pose.position.y = light_y
                light_stop_wp = self.get_closest_waypoint_idx(light_stop_pose)

            
                if pred_state == 0:
                    light_state = TrafficLight.RED
                elif pred_state == 1:
                    light_state = TrafficLight.YELLOW
                elif pred_state == 2:
                    light_state = TrafficLight.GREEN
                else:
                    light_state = TrafficLight.UNKNOWN
                
                print("PREDICTION STATE: ",light_state)                
            
        if not traffic_light_found:           
            return -1, light_state
        else:
            return light_stop_wp, light_state


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
