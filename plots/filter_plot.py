#!usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import sys

directory = "/home/ryan/catkin_ws/src/stationary_base_docking/plots/"

raw_df = pd.read_csv(directory + str(sys.argv[1]), header=None, names=['x', 'y', 'z'])
filtered_df = pd.read_csv(directory + str(sys.argv[2]), header=None, names=['x', 'y', 'z'])
# setpoint_df = pd.read_csv(directory + str(sys.argv[3]), header=None, names=['x', 'y', 'z'])

plt.figure()
plt.scatter(raw_df.index, raw_df['x'], s=1.0, c='k', label='raw')
plt.plot(filtered_df.index, filtered_df['x'], 'g', label='filtered')
# plt.plot(setpoint_df.index, setpoint_df['x'], 'b', lw=1, label='setpoint')
plt.xlabel('iteration')
plt.ylabel('y [m]')
plt.grid(True)
plt.legend()

plt.figure()
plt.scatter(raw_df.index, raw_df['y'], s=1.0, c='k', label='raw')
plt.plot(filtered_df.index, filtered_df['y'], 'g', label='filtered')
# plt.plot(setpoint_df.index, setpoint_df['y'], 'b', lw=1, label='setpoint')
plt.xlabel('iteration')
plt.ylabel('x [m]')
plt.grid(True)
plt.legend()

plt.figure()
plt.scatter(raw_df.index, raw_df['z'], s=1.0, c='k', label='raw')
plt.plot(filtered_df.index, filtered_df['z'], 'g', label='filtered')
# plt.plot(setpoint_df.index, setpoint_df['y'], 'b', lw=1, label='setpoint')
plt.xlabel('iteration')
plt.ylabel('z [m]')
plt.grid(True)
plt.legend()


plt.show()