import numpy as np
import matplotlib.pyplot as plt

def parse_clf_file(file_path):
    odom_data = []
    laser_data = []
    true_positions = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 0 or parts[0].startswith('#'):
                continue
            if parts[0] == 'ODOM':
                x, y, theta = float(parts[1]), float(parts[2]), float(parts[3])
                odom_data.append((x, y, theta))
            elif parts[0] == 'FLASER' or parts[0] == 'RLASER':
                num_readings = int(parts[1])
                ranges = list(map(float, parts[2:2+num_readings]))
                lx, ly, ltheta = float(parts[2+num_readings]), float(parts[3+num_readings]), float(parts[4+num_readings])
                laser_data.append((num_readings, ranges, lx, ly, ltheta))
            elif parts[0] == 'TRUEPOS':
                true_x, true_y, true_theta = float(parts[1]), float(parts[2]), float(parts[3])
                true_positions.append((true_x, true_y, true_theta))

    return odom_data, laser_data, true_positions


def transform_laser_scans(laser_data, odom_data):
    map_points = []
    fov = np.pi  
    for (odom, laser) in zip(odom_data, laser_data):
        _, ranges, lx, ly, ltheta = laser
        ox, oy, otheta = odom
        laser_points = []
        angle_increment = fov / len(ranges)
        for i, range_reading in enumerate(ranges):
            angle = ltheta + i * angle_increment - fov / 2
            x_global = ox + lx + range_reading * np.cos(angle + otheta)
            y_global = oy + ly + range_reading * np.sin(angle + otheta)

            laser_points.append((x_global, y_global))

        map_points.extend(laser_points)

    return np.array(map_points)

def plot_map(map_points):
    plt.figure(figsize=(10, 8))
    plt.scatter(map_points[:, 0], map_points[:, 1], s=1, alpha=0.5)
    plt.title('Robot Laser Trajectory')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

odom_data, laser_data, true_positions = parse_clf_file('intel.clf')
map_points = transform_laser_scans(laser_data, odom_data)
plot_map(map_points)
