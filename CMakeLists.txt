cmake_minimum_required(VERSION 2.8.3)
project(stationary_base_docking)

add_definitions(-std=c++11)

find_package(catkin REQUIRED COMPONENTS mavros_msgs geometry_msgs rospy roscpp sensor_msgs)

catkin_package(
	CATKIN_DEPENDS
		mavros_msgs
		geometry_msgs
		rospy
		roscpp
		sensor_msgs
)

include_directories(include ${catkin_INCLUDE_DIRS})

foreach(dir src bin)
	install(DIRECTORY ${dir}/
		DESTINATION ${CATKIN_PACKAGE_SHARE_DESTIONATION}/${dir})
endforeach(dir)

catkin_python_setup()

catkin_install_python(PROGRAMS bin/test_indoor bin/test
	DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})