import numpy as np
import colorsys
import matplotlib.pyplot as plt


class Plot3DPose:

    def __init__(self, ax, cameras, total_ids=20):
        self._ax = ax
        self._cameras = cameras
        self._total_ids = total_ids
        self._skeleton = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                          (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                          (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
        self._config_ax()

    def _id_to_rgb_color(self, id):
        # Calcula a matiz (cor) com base no ID usando o operador módulo
        hue = (id % self._total_ids) / self._total_ids
        # Define uma saturação e luminosidade fixas para cores vibrantes
        saturation = 0.8
        luminance = 0.6
        # Converte a cor de HSL para RGB
        r, g, b = [x for x in colorsys.hls_to_rgb(hue, luminance, saturation)]
        return r, g, b

    def _get_ax_limits(self):
        cam_positions = np.array([cam.position for cam in self._cameras])
        x = cam_positions[:, 0]
        y = cam_positions[:, 1]
        z = cam_positions[:, 2]
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        z_max = np.max(z)
        return x_min, x_max, y_min, y_max, z_max

    def _config_ax(self):
        x_min, x_max, y_min, y_max, z_max = self._get_ax_limits()

        self._ax.view_init(azim=28, elev=32)
        self._ax.set_xlim(x_min, x_max)
        self._ax.set_xticks(np.arange(x_min, x_max, 2))
        self._ax.set_ylim(y_min, y_max)
        self._ax.set_yticks(np.arange(y_min, y_max, 4))
        self._ax.set_zlim(0, z_max)
        self._ax.set_zticks(np.arange(0, z_max + 0.25, 0.5))
        self._ax.set_xlabel('X', labelpad=20)
        self._ax.set_ylabel('Y', labelpad=10)
        self._ax.set_zlabel('Z', labelpad=5)

    def _plot_cameras(self):
        for cam in self._cameras:
            x, y, z = cam.position
            self._ax.scatter3D(x, y, z, color='black', marker='o', s=10)

    def plot(self, person_3d_ids, pts_3d):
        self._ax.clear()
        self._plot_cameras()

        n_persons = len(person_3d_ids)
        for p in range(n_persons):
            person_id = person_3d_ids[p]
            color_id = self._id_to_rgb_color(person_id)

            x, y, z = pts_3d[p]
            # Plotando ID da pessoa
            self._ax.text(x[0], y[0], z[0]+0.1, '%s' % (str(person_id)), size=10, zorder=1, color='k')

            for i, (u, v) in enumerate(self._skeleton):
                x_pair = [x[u], x[v]]
                y_pair = [y[u], y[v]]
                z_pair = [z[u], z[v]]
                self._ax.plot(x_pair, y_pair, zs=z_pair, linewidth=3, color=color_id)
