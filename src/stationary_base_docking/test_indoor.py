#!usr/bin/env python

import roslib
roslib.load_manifest('stationary_base_docking')
import rospy
import mavros
import cv2
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from geometry_msgs.msg import PoseStamped
import numpy as np


class tracking_test:

	def __init__(self):
		self.state_sub = rospy.Subscriber('mavros/state', State, self.state_cb)
		self.local_pos_pub = rospy.Publisher('mavros/setpoint_position/local', PoseStamped, queue_size=1)

		self.cv_pose = np.array([np.nan, np.nan, np.nan])
		self.cv_pose_avg = np.array([np.nan, np.nan, np.nan])

		# cv pose publisher (motion capture)
		self.cv_feed_pos_pub = rospy.Publisher('/mavros/mocap/pose', PoseStamped, queue_size=1)

		self.set_mode_client = rospy.ServiceProxy('mavros/set_mode', SetMode)

	def state_cb(self,data):
		global current_state
		current_state = data

	def position_control(self):
		pose = PoseStamped()
		pose.pose.position.x = 0.0
		pose.pose.position.y = 0.0
		pose.pose.position.z = 2.0

		# Update timestamp and publish pose 
		pose.header.stamp = rospy.Time.now()
		self.local_pos_pub.publish(pose)

	def detect_led(self):
		global pose_avg_store, cv_avg_poses

		# capture frame
		ret, img = vid.read()

		# blur image edges
		img = cv2.medianBlur(img, 3)
	
		# convert from bgr to hsv
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		
		# threshold the HSV image, keep only the red pixels
		lower_red_hue_range = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
		upper_red_hue_range = cv2.inRange(hsv, (160, 100, 100), (179, 255, 255))
		
		# Combine the above two images
		red_hue_image = cv2.addWeighted(lower_red_hue_range, 0, upper_red_hue_range, 1.0, 0.0)
		red_hue_image = cv2.GaussianBlur(red_hue_image, (9, 9), 2, 2)
	
		# detect contours
		_, contours, _ = cv2.findContours(red_hue_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

		if contours:
			# find contour with largest area
			led_cnt = sorted(contours, key = cv2.contourArea, reverse=True)

			if len(led_cnt) < 2:
				self.cv_pose = np.array([np.nan, np.nan, np.nan])
			elif len(led_cnt) > 1:
				# extract the outer and inner edges of the LED
				outer_cnt = led_cnt[0]
				inner_cnt = led_cnt[1]

				# compute the equivalent diameters of the inner/outer edges
				outer_area = cv2.contourArea(outer_cnt)
				inner_area = cv2.contourArea(inner_cnt)
				outer_dia = np.sqrt(4.0*outer_area/np.pi)
				inner_dia = np.sqrt(4.0*inner_area/np.pi)

				# find the centers of the inner/outer edges
				outer_M = cv2.moments(outer_cnt)
				inner_M = cv2.moments(inner_cnt)
				outer_cx = outer_M['m10']/outer_M['m00']
				outer_cy = outer_M['m01']/outer_M['m00']
				inner_cx = inner_M['m10']/inner_M['m00']
				inner_cy = inner_M['m01']/inner_M['m00']

				# take the averages of the values calculated above
				led_x = (outer_cx + inner_cx)/2
				led_y = (outer_cy + inner_cy)/2
				led_dia = (outer_dia + inner_dia)/2

				# extract information about the cameras view
				img_height, img_width, _ = img.shape
				img_center_x = img_width/2
				img_center_y = img_height/2
				diff_x = img_center_x - led_x
				diff_y = led_y - img_center_y
				fov = np.radians(74)
				foc_len = (img_width/2)/(np.tan(fov/2))
		
				# compute position of MAV from above values and known LED diameter
				led_dia_irl = 0.2286
				unit_dist = led_dia_irl/led_dia
				self.cv_pose[0] = diff_y*unit_dist
				self.cv_pose[1] = diff_x*unit_dist
				self.cv_pose[2] = foc_len*unit_dist

				# simple averaging
				if pose_avg_store >= 0:
					pose_avg_store -= 1
					cv_avg_poses[pose_avg_store,:] = self.cv_pose
				elif pose_avg_store == -1:
					pose_avg_store = 5
					self.cv_pose_avg = np.mean(cv_avg_poses, axis=0)
		
				# display LED detection on image with a cricle and center point
				img = cv2.circle(img, (np.uint16(led_x),np.uint16(led_y)), np.uint16(led_dia)/2, (0,255,0), 2)
				img = cv2.circle(img, (np.uint16(led_x),np.uint16(led_y)), 2, (0,255,0), 3)

		# display images
		cv2.imshow("Image Stream", img)
		cv2.waitKey(3)

	def mocap_feedback(self):
		global cam_alt

		# check that cv position averages exist
		if np.isfinite(self.cv_pose_avg).any():
			feed_pose = PoseStamped()
			feed_pose.pose.position.x = self.cv_pose_avg[0]
			feed_pose.pose.position.y = self.cv_pose_avg[1]
			feed_pose.pose.position.z = self.cv_pose_avg[2] + cam_alt

			# publish pose 	
			feed_pose.header.stamp = rospy.Time.now()
			self.cv_feed_pos_pub.publish(feed_pose)


offb_set_mode = SetMode
current_state = State()

# capture video
vid = cv2.VideoCapture(1)
vid.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
vid.set(cv2.CAP_PROP_EXPOSURE, 0.005)
vid.set(cv2.CAP_PROP_CONTRAST, 0.0005)
vid.set(cv2.CAP_PROP_SATURATION, 3.0)

pose_avg_store = 5
cv_avg_poses = np.empty((pose_avg_store, 3))

cam_alt = 0.19685

def main():
	tt = tracking_test()
	rospy.init_node('tracking_test')

	global current_state
	prev_state = current_state

	rate = rospy.Rate(40.0) # MUST be more then 2Hz

	# send a few setpoints before starting
	for i in range(100):
		tt.position_control()
		rate.sleep()

	# wait for FCU connection
	while not current_state.connected:
		rate.sleep()

	last_request = rospy.get_rostime()
	while not rospy.is_shutdown():
		now = rospy.get_rostime()
		# if current_state.mode != "OFFBOARD" and (now - last_request > rospy.Duration(5.)):
		# 	tt.set_mode_client(base_mode=0, custom_mode="OFFBOARD")
		# 	last_request = now

		# older versions of PX4 always return success==True, so better to check Status instead
		if prev_state.armed != current_state.armed:
			rospy.loginfo("Vehicle armed: %r" % current_state.armed)
		if prev_state.mode != current_state.mode: 
			rospy.loginfo("Current mode: %s" % current_state.mode)
		prev_state = current_state

		tt.detect_led()
		tt.mocap_feedback()
		tt.position_control()
		rate.sleep()


if __name__ == '__main__':
	main()