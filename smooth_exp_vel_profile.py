import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d

class BoundedSmoothVelocityProfile(object):

    @staticmethod
    def g(x, T, v1, v2):
        x = np.asarray(x)

        def g_(x):
            y = x.copy()
            idx = np.where((y>0) & (y < 1))
            y[y >= 1] = v2
            y[y <= 0] = v1
            # y[idx] = v1 + (v2-v1)*np.exp(-1./(y[idx])) / \
            #     (np.exp(-1./(y[idx]))+np.exp(-1./(1-y[idx])))
            y[idx] = v1 + (v2-v1)/(1+np.exp((1-2*y[idx])/(y[idx] - np.square(y[idx]))))
            return y

        return g_(x/T)

    def __init__(self, start_vel=0.1, stop_vel=1., transition_time=1, resolution=1000, *args, **kwargs):

        self._max_time = float(transition_time)
        self._vel_map = lambda x: BoundedSmoothVelocityProfile.g(
            x, self._max_time, start_vel, stop_vel)

    def get_velocity_at(self, t):
        if isinstance(t, np.ndarray):
            t[t > self._max_time] = self._max_time
            t[t < 0.] = 0
        else:
            t = max(min(t, self._max_time), 0)
        return self._vel_map(t)

    def get_full_velocity_curve(self, resolution=1000):
        timeline = np.linspace(0., self._max_time, resolution)
        return self.get_velocity_at(timeline), timeline

    def get_total_distance_at(self, t, resolution=1000):
        return self.get_full_distance_curve(resolution=resolution)[0][int(t/self._max_time*resolution)-1]

    def get_full_distance_curve(self, resolution=1000):
        vel_curve, timeline = self.get_full_velocity_curve(
            resolution=resolution)
        spl = UnivariateSpline(timeline, vel_curve, s=0)
        ispl = spl.antiderivative()
        return ispl(timeline), timeline

    def get_full_acceleration_curve(self, resolution=1000):
        v, t = self.get_full_velocity_curve()
        return np.gradient(v, t), t

    def get_acceleration_at(self, t, resolution=1000):
        return self.get_full_acceleration_curve()[0][t]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from velprof_utils import get_distance_from_vel_curve
    a = BoundedSmoothVelocityProfile.g
    b = np.linspace(-0,1.5,100)
    y= a(b,1,1.2,0.5)

    plt.subplot(4,1,1)
    plt.plot(b,y)
    plt.title("Velocity Curve")
    plt.axvline(0, c="g", linestyle="--", label="transition start")
    plt.axvline(1,  c="r", linestyle="--", label="transition end")
    plt.legend()
    plt.xticks([], [])
    # plt.ylabel("Speed")
    plt.subplot(4, 1, 2)
    plt.plot(b, get_distance_from_vel_curve(y, b))
    plt.xticks([],[])
    plt.title("Distance Traj")
    plt.axvline(0, c="g", linestyle="--")
    plt.axvline(1,  c="r", linestyle="--")
    plt.subplot(4,1,3)
    acc = np.gradient(y,b)
    plt.plot(b,acc)
    plt.title("Acceleration Curve")
    plt.axvline(0, c="g", linestyle="--")
    plt.axvline(1,  c="r", linestyle="--")
    plt.xticks([],[])
    plt.subplot(4, 1, 4)
    jerk = np.gradient(acc, b)
    plt.plot(b, jerk)
    plt.title("Jerk Curve")
    plt.axvline(0, c="g", linestyle="--")
    plt.axvline(1,  c="r", linestyle="--")
    plt.xlabel("Time")
    plt.show()
