import matplotlib.pyplot as plt
import numpy as np
import icp
import g2o

class PoseGraphOptimization(g2o.SparseOptimizer):
    def __init__(self):
        super().__init__()
        self._setup_optimizer()

    def _setup_optimizer(self):
        solver = g2o.BlockSolverSE2(g2o.LinearSolverDenseSE2())
        optimization_algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
        self.set_algorithm(optimization_algorithm)

    def optimize(self, max_iterations=20):
        self.initialize_optimization()
        super().optimize(max_iterations)

    def add_vertex(self, id, pose, fixed=False):
        vertex = g2o.VertexSE2()
        vertex.set_id(id)
        vertex.set_estimate(pose)
        vertex.set_fixed(fixed)
        super().add_vertex(vertex)

    def add_edge(self, vertices, measurement, information=np.identity(3), robust_kernel=None):
        edge = g2o.EdgeSE2()
        self._connect_vertices(edge, vertices)
        edge.set_measurement(measurement)
        edge.set_information(information)
        if robust_kernel:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def _connect_vertices(self, edge, vertices):
        for i, vertex_id in enumerate(vertices):
            vertex = self.vertex(vertex_id) if isinstance(vertex_id, int) else vertex_id
            edge.set_vertex(i, vertex)

    def get_pose(self, id):
        return self.vertex(id).estimate()



class LaserScanner:
    def __init__(self):
        self.lasers = []
        self.odoms = []

    def read_data(self, input_file):
        with open(input_file, 'r') as file:
            for line in file:
                tokens = line.split(' ')
                if tokens[0] == 'FLASER':
                    num_readings = int(tokens[1])
                    scans = np.array(tokens[2:2 + num_readings], dtype=np.cfloat)
                    angles = np.radians(np.arange(-90, 90 + 180 / num_readings, 180 / num_readings))
                    angles = np.delete(angles, num_readings // 2)
                    x_coords = np.cos(angles) * scans
                    y_coords = np.sin(angles) * scans
                    converted_scans = np.column_stack((x_coords, y_coords))
                    self.lasers.append(converted_scans)
                    x, y, theta = map(float, tokens[2 + num_readings:5 + num_readings])
                    self.odoms.append([x, y, theta])

    def process_data(self):
        self.odoms = np.array(self.odoms)
        self.lasers = np.array(self.lasers)

    def optimize_pose(self):
        optimizer = PoseGraphOptimization()
        current_pose = np.eye(3)
        optimizer.add_vertex(0, g2o.SE2(g2o.Isometry2d(current_pose)), True)

        vertex_idx = 1
        registered_scans = []

        for odom_idx, odom in enumerate(self.odoms):
            if odom_idx == 0:
                prev_odom = odom.copy()
                prev_idx = 0
                registered_scans.append(self.lasers[odom_idx])
                continue

            delta_odom = odom - prev_odom
            if np.linalg.norm(delta_odom[:2]) > 0.4 or abs(delta_odom[2]) > 0.2:
                prev_scan = self.lasers[prev_idx]
                current_scan = self.lasers[odom_idx]
                delta_x, delta_y, delta_yaw = delta_odom[0], delta_odom[1], delta_odom[2]
                init_transform = np.array([[np.cos(delta_yaw), -np.sin(delta_yaw), delta_x],
                                        [np.sin(delta_yaw), np.cos(delta_yaw), delta_y],
                                        [0, 0, 1]])

                try:
                    transformed_scan, _, _, cov = icp.icp(current_scan, prev_scan, init_transform,
                                                        max_iterations=80, tolerance=0.0001)
                except Exception as e:
                    continue

                current_pose = np.matmul(current_pose, transformed_scan)
                optimizer.add_vertex(vertex_idx, g2o.SE2(g2o.Isometry2d(current_pose)))
                information_matrix = np.linalg.inv(cov)
                optimizer.add_edge([vertex_idx - 1, vertex_idx], g2o.SE2(g2o.Isometry2d(transformed_scan)),
                                information_matrix)

                prev_odom = odom
                prev_idx = odom_idx
                registered_scans.append(current_scan)

                vertex_idx += 1

        return optimizer, registered_scans


def visualize_graph(optimizer, registered_lasers):
    plt.figure(figsize=(10, 10))
    num_lasers = len(registered_lasers)
    for idx in range(num_lasers):
        pose = optimizer.get_pose(idx)
        isometry = pose.to_isometry()
        r = isometry.R
        t = isometry.t
        point_cloud = (r @ registered_lasers[idx].T + t[:, np.newaxis]).T
        plt.plot(point_cloud[:, 0], point_cloud[:, 1], '.b', markersize=0.1)
        if idx < num_lasers - 1:
            next_pose = optimizer.get_pose(idx + 1).to_isometry()
            current_x, current_y = t[0], t[1]
            next_x, next_y = next_pose.t[0], next_pose.t[1]
            plt.plot([current_x, next_x], [current_y, next_y], '-g')

    plt.show()

input_file = 'intel.clf'

scanner = LaserScanner()
scanner.read_data(input_file)
scanner.process_data()
optimizer, registered_lasers = scanner.optimize_pose()
visualize_graph(optimizer, registered_lasers)
