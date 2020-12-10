import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import UnivariateSpline, interp1d
from velprof_utils import LogisticFuncUtils as SigFunc

class SigmoidVelocityProfile(object):

    def __init__(self, start_vel=0.1, stop_vel=1., transition_time=1, resolution = None, sigmoid_bound_percent=0.995):

        self._startval = start_vel
        self._L = stop_vel
        self._x0 = float(transition_time)/2
        self._k = self._find_k_with_check(sigmoid_bound_percent)

    def _find_k_with_check(self, sigmoid_bound_percent):
        y1 = self._startval + (1-sigmoid_bound_percent)*(self._L - self._startval)
        y2 = self._startval + (sigmoid_bound_percent)*(self._L - self._startval)

        k1 = SigFunc.solve_for_k(0,y1,self._x0,self._L,self._startval)
        k2 = SigFunc.solve_for_k(self._x0*2,y2,self._x0,self._L,self._startval)
        # print k1, k2, self._x0, self._L, self._startval,y1,y2
        assert np.isclose(k1,k2,rtol=1.e-6,atol=0.0), "Error computing logarithmic growth rate k for sigmoid! Diff: {}".format(np.abs(k1-k2))
        return k1

    def get_velocity_at(self, t):
        return SigFunc.logistic_function(t,self._x0,self._k,self._L,self._startval)

    def get_time_when_velocity_is(self, vel):
        return SigFunc.inverse_logistic_function(vel, self._x0, self._k, self._L, self._startval)

    def get_full_velocity_curve(self, resolution=1000):
        timeline = np.linspace(0,self._x0*2,resolution)
        return self.get_velocity_at(timeline), timeline

    def get_total_distance_at(self, t, resolution=1000):
        return self.get_full_distance_curve(resolution=resolution)[0][int(t/(self._x0*2)*resolution)-1]

    def get_full_distance_curve(self, resolution=1000):
        vel_curve, timeline = self.get_full_velocity_curve(resolution=resolution)
        spl = UnivariateSpline(timeline, vel_curve, s=0)
        ispl = spl.antiderivative()
        return ispl(timeline), timeline

    def get_full_acceleration_curve(self, resolution=1000):
        v,t = self.get_full_velocity_curve()
        return np.gradient(v,t), t

    def get_acceleration_at(self, t, resolution=1000):
        return self.get_full_acceleration_curve()[0][int(t/(self._x0*2)*resolution)-1]
        

if __name__ == "__main__":

    svp = SigmoidVelocityProfile(start_vel=0., stop_vel=0.5, transition_time=0.5,sigmoid_bound_percent=0.99)

    v, t = svp.get_full_velocity_curve()
    d, _ = svp.get_full_distance_curve()

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
