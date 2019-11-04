#!usr/bin/env python2

import roslib
roslib.load_manifest('stationary_base_docking')
import rospy
import cv2
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State
from threading import Thread, Event, Timer
from mavros_msgs.srv import CommandBool, SetMode
import numpy as np
from queue import Queue
import time

class Common(object):

	def __init__(self):
		self.pos = PoseStamped()
		self.vel = TwistStamped()
		self.state = State()
		self.vision_pos = PoseStamped()

		self.ros_rate = 40
		self.alt = 5.0                # m
		self.yaw_offset = 0.0
		self.dead_dia_irl = 0.1778
		self.dead_switch = 0
		self.shift_time = 0.0		  # s
		self.last_lost = 0.0 		  # s
		self.takeoff = True
		self.led_dia_irl = 0.2413
		self.cam_alt = 1              # m
		self.camera_matrix = np.array([[1.1327077728604816e+03, 0.0, 640.0],
			[0.0, 1.1327077728604816e+03, 360.0], 
			[0.0, 0.0, 1.0]])
		self.distort_coeffs = np.array([-4.8479906637422165e-01, 2.0146579393624808e-01, 
			0.0, 0.0, 3.3405561891249991e-01])

		self.cv_shift = np.array([np.nan, np.nan, np.nan])
		self.current_pose = np.array([np.nan, np.nan, np.nan, 
			np.nan, np.nan, np.nan, np.nan])
		self.takeoff_ori = np.array([np.nan, np.nan, np.nan, np.nan])
		self.cv_pose_raw = np.array([np.nan, np.nan, np.nan])
		self.cv_pose = np.array([np.nan, np.nan, np.nan])
		self.cv_pose_notf = np.array([np.nan, np.nan, np.nan])
		self.cv_vel_raw	=np.zeros(3)
		self.cv_vel = np.zeros(3)
		self.carrier_pose = np.zeros(3)

		self.rolling_init = 0
		self.rolling_cv = np.zeros([2, 3])
		self.rolling_time = np.zeros(2)
		self.reject_time = 0
		self.collect_ind = 0
		self.collect_size = 1200
		self.raw_txt = np.empty([self.collect_size, 3])
		self.filter_txt = np.empty([self.collect_size, 3])
		self.current_txt = np.empty([self.collect_size, 3])
		self.transform_txt = np.empty([self.collect_size, 3])

		self.local_pos_pub_docker = rospy.Publisher(
			'/mavros/setpoint_position/local', PoseStamped, queue_size=1)
		self.vel_setpoint_pub = rospy.Publisher(
			'/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=1)
		self.vision_pos_pub = rospy.Publisher(
			'/mavros/vision_pose/pose', PoseStamped, queue_size=1)
		
		self.local_pos_sub = rospy.Subscriber(
			'/mavros/local_position/pose', PoseStamped, self.pose_cb)
		self.state_docker_sub = rospy.Subscriber(
			'/mavros/state', State, self.state_docker_cb)

		self.arming_client_docker = rospy.ServiceProxy(
			'/mavros/cmd/arming', CommandBool)
		self.set_mode_client_docker = rospy.ServiceProxy(
			'/mavros/set_mode', SetMode)

		self.pos_desired_q = Queue() 

		self.pos_pub_thread = Thread(target=self.position_pub)
		self.pos_reached_thread = Thread(target=self.position_reached)
		self.detect_led_thread = Thread(target=self.detect_led)
		self.filter_thread = Thread(target=self.call_filter)
		self.collect_thread = Thread(target=self.collect_led_data)
		self.vel_pub_thread = Thread(target=self.velocity_pub)
		self.vel_dock_thread = Thread(target=self.dock_velocity)
		self.vision_thread = Thread(target=self.vision_feedback)

		self.led_event = Event()
		self.filter_event = Event()
		self.collect_event = Event()
		self.reached_event = Event()
		self.docking_event = Event()
		self.final_event = Event()
		self.lost_event = Event()
		self.stop_pos_event = Event()
		self.vision_event = Event()

		self.vid = cv2.VideoCapture(1)
		# outdoor (sunny)
		self.vid.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
		self.vid.set(cv2.CAP_PROP_EXPOSURE, 0.0005)
		self.vid.set(cv2.CAP_PROP_CONTRAST, 0.0005)
		self.vid.set(cv2.CAP_PROP_SATURATION, 0.5)
		# # outdoor (cloudy)
		# self.vid.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
		# self.vid.set(cv2.CAP_PROP_EXPOSURE, 0.0001)
		# self.vid.set(cv2.CAP_PROP_CONTRAST, 0.0005)
		# self.vid.set(cv2.CAP_PROP_SATURATION, 0.7)
		# # indoor
		# self.vid.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
		# self.vid.set(cv2.CAP_PROP_EXPOSURE, 0.001)
		# self.vid.set(cv2.CAP_PROP_CONTRAST, 0.0005)
		# self.vid.set(cv2.CAP_PROP_SATURATION, 3.0)

		fourcc = cv2.VideoWriter_fourcc(*'MJPG')
		self.vid.set(cv2.CAP_PROP_FOURCC, fourcc)
		self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
		self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
		self.vid.set(cv2.CAP_PROP_FPS, 60)

		# video writer
		direc = "/home/ryan/catkin_ws/src/stationary_base_docking/vids/"
		fps = int(self.vid.get(5))
		vid_width = int(self.vid.get(3))
		vid_height = int(self.vid.get(4))
		self.out = cv2.VideoWriter(direc + 
			'vid_{}.avi'.format(time.strftime("%d-%b-%H:%M")), 
			cv2.VideoWriter_fourcc('M','J','P','G'), fps, (vid_width, vid_height))

	
	def state_docker_cb(self, data):
		self.state = data


	def pose_cb(self, data):
		self.current_pose[0] = data.pose.position.x
		self.current_pose[1] = data.pose.position.y
		self.current_pose[2] = data.pose.position.z
		self.current_pose[3] = data.pose.orientation.x
		self.current_pose[4] = data.pose.orientation.y
		self.current_pose[5] = data.pose.orientation.z
		self.current_pose[6] = data.pose.orientation.w 


	def set_desired_pose(self, desired_pose):
		"""
		Function takes in pose data from an array and populates a PoseStamped message
		"""
		self.pos.pose.position.x = desired_pose[0]
		self.pos.pose.position.y = desired_pose[1]
		self.pos.pose.position.z = desired_pose[2]
		self.pos.pose.orientation.x = desired_pose[3]
		self.pos.pose.orientation.y = desired_pose[4]
		self.pos.pose.orientation.z = desired_pose[5]
		self.pos.pose.orientation.w = desired_pose[6]


	def position_reached(self, offset=1.0, gps_offset=2.0):
		"""
		Checks if a position setpoint is reached

		Runs in the pos_reached_thread and is called whenever the position queue
		is given a new position setpoint
		"""
		desired_pose = np.zeros(6)
		rate = rospy.Rate(self.ros_rate)
		while not rospy.is_shutdown() and not self.stop_pos_event.is_set():
			# check if mav is at desired position
			off_check = np.linalg.norm(desired_pose[:3] - self.current_pose[:3]) < offset
			# check it position has been reached
			reached = self.reached_event.is_set()
			# mav will always start in a takeoff state
			if self.takeoff:
				# extract the position setpoint from the queue
				desired_pose = self.pos_desired_q.get()
				# set the desired pose
				self.set_desired_pose(desired_pose)
				self.takeoff = False
			elif off_check and not reached:
				time.sleep(10)
				self.reached_event.set()
				rospy.loginfo("Position setpoint reached")
				# if there is another desired position, take it out of the queue
				if not self.pos_desired_q.empty():
					desired_pose = self.pos_desired_q.get()
					# set the desired pose
					self.set_desired_pose(desired_pose)

			elif not off_check and not self.docking_event.is_set():
				self.reached_event.clear()
			elif reached and not self.pos_desired_q.empty():
				desired_pose = self.pos_desired_q.get()
				# set the desired pose
				self.set_desired_pose(desired_pose)
				self.reached_event.clear()

			rate.sleep()


	def position_setpoint(self, x, y, z, ori_x=0, ori_y=0, ori_z=0, ori_w=1):
		self.pos_desired_q.put(np.array([x, y, z, ori_x, ori_y, ori_z, ori_w]))


	def position_pub(self):
		rate = rospy.Rate(self.ros_rate)
		while not rospy.is_shutdown() and not self.stop_pos_event.is_set():
			self.pos.header.stamp = rospy.Time.now()
			self.local_pos_pub_docker.publish(self.pos)

			rate.sleep()


	def set_arm(self, arm):
		if arm and not self.state.armed:
			self.arming_client_docker(True)
			rospy.loginfo("Docker armed: %r" % arm)

		elif not arm:
			self.arming_client_docker(False)
			rospy.loginfo("Docker armed: %r" % arm)

			
	def set_offboard(self):
		if self.state.mode != "OFFBOARD":
			self.set_mode_client_docker(base_mode=0, custom_mode="OFFBOARD")
			rospy.loginfo("Docker mode: %s" % "OFFBOARD")	


	def rej_outlier(self, previous, current, xy_offset, z_offset):
		"""
		Compares the element wise absolute difference between two arrays and
		returns the 'previous' array if the difference to the 'current' array
		is greater than a desired 'offset' 
		"""
		offsets = abs(previous - current)
		if offsets[0] > xy_offset or offsets[1] > xy_offset or offsets[2] > z_offset:
			# reject outlier case
			return True
		else:
			return False


	def kalman_filter(self):
		"""
		Kalman filter used to filter the raw cv measurements

		Typical 1D (position only) kalman filtering method
		"""
		rate = rospy.Rate(self.ros_rate)
		switch = 0
		e_mea = 0.00001                           # measurement error
		e_pro = 1e-5 		   			          # process covariance
		e_est = np.array([0.2, 0.2, 0.4])         # initial estimation error (guess)
		cv_kalman = self.cv_pose_raw			  # initial cv pose 

		while not rospy.is_shutdown():
			cv_measured = self.cv_pose_raw
			cv_minus = cv_kalman
			e_est_minus = e_est + e_pro
			k_gain = e_est_minus/(e_est_minus + e_mea)
			cv_kalman = cv_minus + k_gain*(cv_measured - cv_minus)
			e_est = (1 - k_gain)*e_est_minus

			self.cv_pose_notf = cv_kalman

			# set the filter event if has not been set and the mav is not docked
			if not self.filter_event.is_set() and not self.final_event.is_set():
				self.filter_event.set()
			# clear the filter event if the mav docks, this stops cv pose updates
			elif self.final_event.is_set():
				self.filter_event.clear()

			rate.sleep()


	def call_filter(self):
		"""
		Calls the kalman filter shortly after led detection begins

		Runs in the filter_thread
		"""
		if self.led_event.wait():
			Timer(1.0, self.kalman_filter).start()


	def led_to_cv(self, img, led_data):
		# extract information about the cameras view
		img_height, img_width, _ = img.shape
		img_center_x = img_width/2
		img_center_y = img_height/2
		diff_x = img_center_x - led_data[0]
		diff_y = led_data[1] - img_center_y

		# compute position of mav from above values and known LED diameter
		unit_dist = self.led_dia_irl/led_data[2]
		cv = np.empty(3)
		cv[0] = diff_y*unit_dist
		cv[1] = diff_x*unit_dist
		cv[2] = self.camera_matrix[0, 0]*unit_dist

		return cv


	def cv_to_led(self, img, cv_data):
		# extract information about the cameras view
		img_height, img_width, _ = img.shape
		img_center_x = img_width/2
		img_center_y = img_height/2
		
		unit_dist = cv_data[2]/self.camera_matrix[0, 0]
		led = np.zeros(3)
		led[0] = img_center_x - cv_data[1]/unit_dist
		led[1] = cv_data[0]/unit_dist + img_center_y
		led[2] = self.led_dia_irl/unit_dist

		return led


	def detect_led(self):
		rate = rospy.Rate(self.ros_rate)
		while not rospy.is_shutdown():
			# capture frame
			ret, img = self.vid.read()

			# compute image size
			img_height, img_width, _ = img.shape

			# compute undistortion and rectification transformation map
			mapx, mapy = cv2.initUndistortRectifyMap(self.camera_matrix, 
				self.distort_coeffs, None, self.camera_matrix, (img_width, img_height), 5)

			# apply transformation map to image
			img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

			# blur image edges
			img = cv2.medianBlur(img, 3)

			# convert from bgr to hsv
			hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		
			# threshold the HSV image, keep only the red pixels
			lower_red_hue_range = cv2.inRange(hsv, (0, 135, 50), (30, 255, 255))
			upper_red_hue_range = cv2.inRange(hsv, (160, 135, 50), (179, 255, 255))
		
			# Combine the above two images
			red_hue_image = cv2.bitwise_or(lower_red_hue_range, upper_red_hue_range)
			red_hue_image = cv2.GaussianBlur(red_hue_image, (9, 9), 2, 2)

			# detect contours
			_, contours, _ = cv2.findContours(
				red_hue_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

			# check that two contours exist (outside and inside eges of LED)
			if len(contours) < 2:
				lost_time = time.time()
				# send nan values for led if the led has not been found for the first time yet
				if not self.led_event.is_set():
					led_raw = np.array([np.nan, np.nan, np.nan])
				elif self.led_event.is_set() and lost_time - self.last_lost > 0.5:
					# set the event signaling that the led is not detected
					self.lost_event.set()
		
			else:
				# signal that vision estimates are being collected
				self.led_event.set()
				self.last_lost = time.time()
				self.lost_event.clear()

				# sort contours by area
				led_cnt = sorted(contours, key=cv2.contourArea, reverse=True)

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
				led_raw = np.array([(outer_cx + inner_cx)/2, (outer_cy + inner_cy)/2, 
					(outer_dia + inner_dia)/2])

				cv_raw = self.led_to_cv(img, led_raw)

				cols = [0, 1, 2]
				if self.rolling_init < 2:
					self.rolling_cv[self.rolling_init, cols] = cv_raw
					self.rolling_time[self.rolling_init] = time.time()
					self.cv_pose_raw = cv_raw
					self.rolling_init += 1
				else:
					self.rolling_cv[0, cols] = self.rolling_cv[1, cols]
					self.rolling_time[0] = self.rolling_time[1]
					current_time = time.time()
					if not self.rej_outlier(self.rolling_cv[0, cols], cv_raw, 0.2, 0.4):
						self.rolling_cv[1, cols] = cv_raw
						self.rolling_time[1] = time.time()
						self.cv_vel_raw = np.subtract(self.rolling_cv[1, cols],
							self.rolling_cv[0, cols])/np.subtract(self.rolling_time[1],
							self.rolling_time[0])
						self.reject_time = time.time()
					else:
						if current_time - self.reject_time > 0.2:
							self.rolling_init = 0
					self.cv_pose_raw = self.rolling_cv[1, cols]		

				if self.filter_event.is_set():
					# transform the cv pose from the camera frame to the local frame
					self.cv_pose[0] = (np.cos(self.yaw_offset)*self.cv_pose_raw[0] -
						np.sin(self.yaw_offset)*self.cv_pose_raw[1])
					self.cv_pose[1] = (np.sin(self.yaw_offset)*self.cv_pose_raw[0] +
						np.cos(self.yaw_offset)*self.cv_pose_raw[1])
					self.cv_pose[2] = self.cv_pose_raw[2]

					# transform the cv velocity from the camera frame to the local frame
					self.cv_vel[0] = (np.cos(self.yaw_offset)*self.cv_vel_raw[0] -
						np.sin(self.yaw_offset)*self.cv_vel_raw[1])
					self.cv_vel[1] = (np.sin(self.yaw_offset)*self.cv_vel_raw[0] +
						np.cos(self.yaw_offset)*self.cv_vel_raw[1])
					self.cv_vel[2] = self.cv_vel_raw[2]

					# display LED detection on image with a cricle and center point
					led_filtered = self.cv_to_led(img, self.cv_pose_raw)
					img = cv2.circle(img, (np.uint16(led_filtered[0]),
						np.uint16(led_filtered[1])), 
						np.uint16(led_filtered[2])/2, (0,255,0), 2)
					img = cv2.circle(img, (np.uint16(led_filtered[0]),
						np.uint16(led_filtered[1])), 2, (0,255,0), 3)

					# calculate dead zone diameter to display
					dead_led_ratio = self.dead_dia_irl/self.led_dia_irl
					dead_dia = dead_led_ratio*led_filtered[2]

					# display dead zone
					img_height, img_width, _ = img.shape
					img_center_x = img_width/2
					img_center_y = img_height/2
					img = cv2.circle(img, (np.uint16(img_center_x),np.uint16(img_center_y)), 
						np.uint16(dead_dia)/2, (255,0,0), 2)

					# display image
					cv2.imshow("Image Stream", img)
					cv2.waitKey(3)

					if self.collect_ind < self.collect_size:
						# write img to video out
						self.out.write(img)

						# collect raw/filtered led data into a txt file
						rows = [self.collect_ind, self.collect_ind, self.collect_ind]
						self.raw_txt[rows, cols] = \
							np.array([self.cv_pose_raw[0], self.cv_pose_raw[1], 
							self.cv_pose_raw[2]])
						self.filter_txt[rows, cols] = \
							np.array([self.cv_pose_notf[0], self.cv_pose_notf[1], 
							self.cv_pose_notf[2]])
						self.current_txt[rows, cols] = \
							np.array([self.current_pose[0], 
							self.current_pose[1], self.current_pose[2]])
						self.transform_txt[rows, cols] = \
							np.array([self.cv_pose[0], self.cv_pose[1], 
							self.cv_pose[2]])
						self.collect_ind += 1

						# end collection
						if self.collect_ind == self.collect_size:
							self.collect_event.set()

			rate.sleep()


	def collect_led_data(self):
		if self.collect_event.wait():
			# release video out
			self.out.release()

			# write led data to csv files
			direc = "/home/ryan/catkin_ws/src/stationary_base_docking/plots/"
			np.savetxt(direc + 
				'irl_raw_{}.csv'.format(time.strftime("%d-%b-%H:%M")), 
				self.raw_txt, delimiter=',')
			np.savetxt(direc + 
				'irl_filtered_{}.csv'.format(time.strftime("%d-%b-%H:%M")), 
				self.filter_txt, delimiter=',')
			np.savetxt(direc + 
				'irl_current_{}.csv'.format(time.strftime("%d-%b-%H:%M")), 
				self.current_txt, delimiter=',')
			np.savetxt(direc + 
				'irl_transform_{}.csv'.format(time.strftime("%d-%b-%H:%M")), 
				self.transform_txt, delimiter=',')

			rospy.loginfo("LED data collected")


	def vision_feedback(self):
		if self.filter_event.wait() and self.reached_event.wait():
		# if self.filter_event.wait():
			time.sleep(2)
			rospy.loginfo("Vision feedback initialized")

			init_pose = self.current_pose[:3] - self.cv_pose

			self.vision_event.set()

			rate = rospy.Rate(self.ros_rate)
			while not rospy.is_shutdown():
				if not np.any(np.isnan(self.cv_pose)):
					self.vision_pos.pose.position.x = self.cv_pose[0] + init_pose[0]
					self.vision_pos.pose.position.y = self.cv_pose[1] + init_pose[1]
					self.vision_pos.pose.position.z = self.cv_pose[2] + self.cam_alt

					self.vision_pos.header.stamp = rospy.Time.now()
					self.vision_pos_pub.publish(self.vision_pos)

				rate.sleep()


	def velocity_pub(self):
		if self.stop_pos_event.wait():
			rate = rospy.Rate(self.ros_rate)
			while not rospy.is_shutdown():
				self.vel.header.stamp = rospy.Time.now()
				self.vel_setpoint_pub.publish(self.vel)

				rate.sleep()


	def dock_velocity(self):
		if self.reached_event.wait() and self.filter_event.wait():
		# if self.vision_event.wait():
			self.stop_pos_event.set()

			alt_thresh = 2.0
			ascent_rate = 0.4
			descent_rate = 0.2     
			descent_time = self.cam_alt/descent_rate 
			reached = False
			hit_center = False

			rate = rospy.Rate(self.ros_rate)
			while not rospy.is_shutdown() and not self.final_event.is_set():
				desired_vel_x = -7.2*self.cv_pose[0]*abs((7.2*self.cv_pose[0])**3)
				# desired_vel_x = -2*self.cv_pose[0]
				if desired_vel_x > 0.3:
					desired_vel_x = 0.3

				desired_vel_y = -7.2*self.cv_pose[1]*abs((7.2*self.cv_pose[1])**3)
				# desired_vel_y = -2*self.cv_pose[1]
				if desired_vel_y > 0.3:
					desired_vel_y = 0.3
				
				self.vel.twist.linear.x = desired_vel_x
				self.vel.twist.linear.y = desired_vel_y

				off_center = 2*np.sqrt(self.cv_pose[0]**2 +
					self.cv_pose[1]**2) > self.dead_dia_irl
				current_time = time.time()
				# if self.cv_pose[2] >= alt_thresh and self.lost_event.is_set():
				# 	self.vel.twist.linear.z = ascent_rate
				# 	time.sleep(5)
				# elif self.cv_pose[2] < alt_thresh and off_center:
				# 	self.vel.twist.linear.z = ascent_rate
				# 	time.sleep(5)
				# elif self.cv_pose[2] < alt_thresh and not off_center:
				# 	self.final_event.set()
				# 	time.sleep(descent_time)
				# 	self.set_arm(False)
				if self.cv_pose[2] <= alt_thresh and not reached:
					self.vel.twist.linear.z = 0.0
					reached = True
				elif self.cv_pose[2] > alt_thresh and not reached:
					self.vel.twist.linear.z = 0.0

					if not off_center:
						hit_center = True

					if hit_center:
						self.vel.twist.linear.x = desired_vel_x #- self.cv_vel[0]
						self.vel.twist.linear.y = desired_vel_y #- self.cv_vel[1]						

				rate.sleep()