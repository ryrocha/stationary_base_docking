#!usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import sys


"""
18-Oct-08:47 - 0.04 
17-Oct - 0.05
18-Oct-08:53 - 0.06
18-Oct-08:58 - 0.07
18-Oct-09:02 - 0.08

Seems to operate ok within region 0.05 - 0.07
"""

directory = "/home/ryan/catkin_ws/src/stationary_base_docking/plots/"

# raw_df = pd.read_csv(directory + str(sys.argv[1]), header=None, names=['x', 'y', 'z'])
filtered_df = pd.read_csv(directory + str(sys.argv[1]), header=None, names=['x', 'y', 'z'])
# current_df = pd.read_csv(directory + str(sys.argv[3]), header=None, names=['x', 'y', 'z'])

plt.figure()
# plt.plot(raw_df.index, raw_df['x'], 'k', label='raw cv')
plt.plot(filtered_df.index, filtered_df['x'], 'g', label='transform cv')
# plt.plot(current_df.index, current_df['x'], 'b', lw=1, label='px4')
plt.xlabel('iteration')
plt.ylabel('x [m]')
plt.grid(True)
plt.legend()

plt.figure()
# plt.plot(raw_df.index, raw_df['y'], 'k', label='raw cv')
plt.plot(filtered_df.index, filtered_df['y'], 'g', label='transform cv')
# plt.plot(current_df.index, current_df['y'], 'b', label='px4')
plt.xlabel('iteration')
plt.ylabel('y [m]')
plt.grid(True)
plt.legend()

# plt.figure()
# plt.plot(raw_df.index, raw_df['z'], 'k', label='raw cv')
# plt.plot(filtered_df.index, filtered_df['z'], 'g', label='transform cv')
# plt.plot(current_df.index, current_df['z'], 'b', label='px4')
# plt.xlabel('iteration')
# plt.ylabel('z [m]')
# plt.grid(True)
# plt.legend()


plt.show()