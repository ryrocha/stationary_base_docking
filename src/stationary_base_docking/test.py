#!usr/bin/env python2

import roslib
roslib.load_manifest('stationary_base_docking')
from common import Common
from threading import Timer
import rospy
import time
from sensor_msgs.msg import Image
import tf

class stationary_docking(Common):

	def __init__(self):
		super(stationary_docking, self).__init__()


	def docking_procedure(self):
		# delay for pose subscriber to get initial position
		time.sleep(5)

		# set initial orientation
		self.takeoff_ori = self.current_pose[3:]
		self.zero_pose = self.current_pose[:3]
		self.yaw_offset = tf.transformations.euler_from_quaternion(self.takeoff_ori)[2]

		# send initial setpoint
		self.position_setpoint(self.zero_pose[0], self.zero_pose[1], self.alt, 
			self.takeoff_ori[0], self.takeoff_ori[1], 
			self.takeoff_ori[2], self.takeoff_ori[3])

		# start threadsg to publish positions and check that positions are being reached
		self.pos_reached_thread.start()
		self.pos_pub_thread.start()

		# delay switch to offboard mode to ensure sufficient initial setpoint stream
		Timer(5.0, self.set_offboard).start()

		# begin filtering vision data
		self.detect_led_thread.start()
		self.filter_thread.start()
		self.collect_thread.start()

		# # begin vision feedback
		# self.vision_thread.start()

		# start velocity controlled docking
		self.vel_dock_thread.start()
		self.vel_pub_thread.start()
		

def main():
	# initialize ros node 
	rospy.init_node('stationary_docking')

	# begin the docking procedure 
	stationary_docking().docking_procedure()

if __name__ == '__main__':
	main()