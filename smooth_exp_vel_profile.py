import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d

class BoundedSmoothVelocityProfile(object):

    @staticmethod
    def g(x, T, v1, v2):
        x = np.asarray(x)
        
        def f(x):
            y = x.copy()
            y[y<=0.] = 0.
            y[y>0.] = np.exp(-1./(y[y>0.]))
            return y

        def g_(x):
            y = x.copy()
            y[y >= 1.] = 1.
            y = f(y)/(f(y)+f(1.-y))
            return y*v2 + (1-y)*v1
            
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
    a = BoundedSmoothVelocityProfile.g
    b = np.linspace(-1.,1.5,100)
    y= a(b)
    plt.plot(b,y)
    plt.show()

    
