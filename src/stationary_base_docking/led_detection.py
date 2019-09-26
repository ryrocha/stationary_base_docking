#!usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt

# capture video
vid = cv2.VideoCapture(1)
vid.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
vid.set(cv2.CAP_PROP_EXPOSURE, 0.000005)
vid.set(cv2.CAP_PROP_CONTRAST, 0.0005)
vid.set(cv2.CAP_PROP_SATURATION, 0.5)

while(True):	
	# capture frame
	ret, img = vid.read()

	# # blur image edges
	# img = cv2.medianBlur(img, 3)
	
	# convert from bgr to hsv
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		
	# threshold the HSV image, keep only the red pixels
	lower_red_hue_range = cv2.inRange(hsv, (0, 100, 50), (10, 255, 255))
	upper_red_hue_range = cv2.inRange(hsv, (160, 100, 50), (179, 255, 255))
		
	# Combine the above two images
	red_hue_image = cv2.bitwise_or(lower_red_hue_range, upper_red_hue_range)
	# red_hue_image = cv2.addWeighted(lower_red_hue_range, 0, upper_red_hue_range, 1.0, 0.0)
	red_hue_image = cv2.GaussianBlur(red_hue_image, (9, 9), 2, 2)
	
	# detect contours
	_, contours, _ = cv2.findContours(red_hue_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	if contours:
		# find contour with largest area
		led_cnt = sorted(contours, key = cv2.contourArea, reverse=True)

		if len(led_cnt) > 1:
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

			# display LED detection on image with a cricle and center point
			img = cv2.circle(img, (np.uint16(led_x),np.uint16(led_y)), np.uint16(led_dia)/2, (0,255,0), 2)
			img = cv2.circle(img, (np.uint16(led_x),np.uint16(led_y)), 2, (0,255,0), 3)

			# draw raw contours
			img = cv2.drawContours(img, contours, -1, (255, 0, 0), 3)

	# display images
	cv2.imshow("Image Stream", img)
	cv2.waitKey(3)