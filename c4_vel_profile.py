import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d


class C4SmoothVelocityProfile(object):
    """
    https://ieeexplore.ieee.org/abstract/document/8594339
    """

    @staticmethod
    def p(x, T, v1, v2, m0=0., m1=0.):
        x = np.asarray(x)
        x[x < 0.] = 0
        x[x > T] = T

        def p_(x):
            return -20*x**7 + 70*x**6 - 84*x**5 + 35*x**4
        
        y = p_(x/T)
        return y*v2 + (1-y)*v1

    def __init__(self, start_vel=0.1, stop_vel=1., transition_time=1, resolution=1000, *args, **kwargs):

        self._max_time = float(transition_time)
        self._vel_map = lambda x: C4SmoothVelocityProfile.p(
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
        return self.get_full_acceleration_curve(resolution=resolution)[0][int(t/self._max_time*resolution)-1]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    a = C4SplineVelocityProfile(0.1, 1.2, 1., 1000)
    vel, T = a.get_full_velocity_curve()
    acc, T = a.get_full_acceleration_curve()
    dist, T = a.get_full_distance_curve()
    plt.subplot(3, 1, 1)
    plt.title("vel")
    plt.plot(T, vel)
    plt.subplot(3, 1, 2)
    plt.title("acc")
    plt.plot(T, acc)
    plt.subplot(3, 1, 3)
    plt.title("dist")
    plt.plot(T, dist)
    plt.show()
