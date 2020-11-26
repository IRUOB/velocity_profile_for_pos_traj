import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d

class HyperbolicTangentVelocityProfile(object):

    def __init__(self, start_vel=0.1, stop_vel=1., transition_time=1, resolution=1000, range_val=3.5):

        self._range_map = range_val
        self._time_map = interp1d(np.linspace(0,transition_time,resolution),np.linspace(-self._range_map,self._range_map,resolution), kind='cubic')
        self._vel_map = interp1d(np.linspace(start_vel,stop_vel,resolution),np.linspace(-1.,1.,resolution), kind='cubic')

        self._inv_time_map = interp1d(np.linspace(-self._range_map,self._range_map,resolution), np.linspace(0,transition_time,resolution), kind='cubic')
        self._inv_vel_map = interp1d(np.linspace(-1.,1.,resolution),np.linspace(start_vel,stop_vel,resolution), kind='cubic')


    def get_velocity_at(self, t):
        if isinstance(t,np.ndarray):
            t[t>1.] = 1
            t[t<0.] = 0
        else:
            t = max(min(t,1.),0)
        return np.round(self._inv_vel_map(np.tanh(self._time_map(t))),7)

    def get_time_when_velocity_is(self, vel):
        return self._inv_time_map(np.arctanh(self._vel_map(vel)))

    def get_full_velocity_curve(self, resolution=1000):
        timeline = self._inv_time_map(np.linspace(-self._range_map,self._range_map,resolution))
        return self.get_velocity_at(timeline), timeline

    def get_total_distance_at(self, t, resolution=1000):
        return self.get_full_distance_curve(resolution=resolution)[0][int(t*resolution)-1]

    def get_full_distance_curve(self, resolution=1000):
        vel_curve, timeline = self.get_full_velocity_curve(resolution=resolution)
        spl = UnivariateSpline(timeline, vel_curve, s=0)
        ispl = spl.antiderivative()
        return ispl(timeline), timeline

    def get_full_acceleration_curve(self, resolution=1000):
        v,t = self.get_full_velocity_curve()
        return np.gradient(v,t), t

    def get_acceleration_at(self, t, resolution=1000):
        return self.get_full_acceleration_curve()[0][t]

if __name__ == "__main__":

    tanh_vp = HyperbolicTangentVelocityProfile(start_vel=0., stop_vel=1.)

    v, t = tanh_vp.get_full_velocity_curve()
    d, _ = tanh_vp.get_full_distance_curve()

    add_ = d[-1]+1/1000
    add_d = np.asarray([0 for _ in range(1000)])
    # print add_d
    d = np.append(add_d,d)
    t = np.append(np.linspace(0,1,add_d.size),1+t)
    # print t.shape, d.shape

    vel = np.gradient(d,t)
    acc = np.gradient(vel,t)

    plt.subplot(3,1,1)
    # plt.plot(t,v)
    plt.plot(t,vel)
    plt.ylabel("Vel")

    plt.subplot(3,1,2)
    plt.plot(t,d)
    plt.ylabel("Dist")
    
    plt.subplot(3,1,3)
    plt.plot(t,acc)
    plt.ylabel("Acc")
    
    plt.show()