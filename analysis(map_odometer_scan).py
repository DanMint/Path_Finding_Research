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

def plot_odometry(odom_data):
    x_coords = [odom[0] for odom in odom_data]
    y_coords = [odom[1] for odom in odom_data]
    plt.figure(figsize=(10, 8))
    plt.plot(x_coords, y_coords, label='Odometry Trajectory', marker='o')
    plt.title('Robot Odometer Trajectory')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.legend()
    plt.grid(True)
    plt.axis('equal') 
    plt.show()

file_path = 'intel.clf'
odom_data, laser_data, true_positions = parse_clf_file(file_path)
plot_odometry(odom_data)
