#!usr/bin/env python

import numpy as np
import cv2
import time

class img_capture(object):

	def __init__(self):	
		# capture video
		self.vid = cv2.VideoCapture(1)
		self.vid.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
		self.vid.set(cv2.CAP_PROP_EXPOSURE, 0.001)
		self.vid.set(cv2.CAP_PROP_CONTRAST, 0.01)
		self.vid.set(cv2.CAP_PROP_SATURATION, 0.7)
		fourcc = cv2.VideoWriter_fourcc(*'MJPG')
		self.vid.set(cv2.CAP_PROP_FOURCC, fourcc)
		self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
		self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
		self.vid.set(cv2.CAP_PROP_FPS, 60)

		# camera matrix
		self.camera_matrix = np.array([[1.1327077728604816e+03, 0.0, 640.0],
			[0.0, 1.1327077728604816e+03, 360.0], 
			[0.0, 0.0, 1.0]])

		# distortion coefficients
		self.distort_coeffs = np.array([-4.8479906637422165e-01, 2.0146579393624808e-01, 
			0.0, 0.0, 3.3405561891249991e-01])

		# video writer
		direc = "/home/ryan/catkin_ws/src/stationary_base_docking/vids/"
		fps = int(self.vid.get(5))
		vid_width = int(self.vid.get(3))
		vid_height = int(self.vid.get(4))
		self.out = cv2.VideoWriter(direc + 
			'vid_{}.avi'.format(time.strftime("%d-%b-%H:%M")), 
			cv2.VideoWriter_fourcc('M','J','P','G'), fps, (vid_width, vid_height))


	def img_stream(self):
		while(True):	
			# capture frame
			ret, img = self.vid.read()

			if ret == True:
				# extract image size
				img_height, img_width, _ = img.shape

				# compute undistortion and rectification maps
				mapx, mapy = cv2.initUndistortRectifyMap(self.camera_matrix, 
					self.distort_coeffs, None, 
					self.camera_matrix, (img_width, img_height), 5)

				# apply transformation map to image
				img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)	

				# display image and write to file
				self.out.write(img)
				cv2.imshow("Image Stream", img)
				
				# press 'Q' to end recording
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

			# break the loop
			else:
				break

		# release video and output file
		self.vid.release()
		self.out.release()

		# close all the frames
		cv2.destroyAllWindows()


def main():
	img_capture().img_stream()

if __name__ == '__main__':
	main()