import numpy as np
from tanh_vel_profile import HyperbolicTangentVelocityProfile as TanhVP
from sigmoid_vel_profile import SigmoidVelocityProfile as SigVP
from scipy.interpolate import UnivariateSpline, interp1d
import matplotlib.pyplot as plt
from utils import get_distance_from_vel_curve

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def get_modified_traj(traj, timeline, tr_point, tr_vel):
    
    dist_array = np.cumsum(np.abs(np.diff(traj,prepend=traj[0])))
    # print dist_array
    # print dist_array.size
    vel = np.abs(np.gradient(traj,timeline).mean())
    # tr_vel = np.sign(vel)*tr_vel
    # print vel

    vp = TanhVP(vel,tr_vel,1.,int(timeline.size/(timeline[-1]-timeline[0])))
    # print vp.get_velocity_at(1.)
    curve_dist = np.round(vp.get_total_distance_at(1.),3)
    print curve_dist

    traj_mod_start_val = dist_array[find_nearest(traj,tr_point)[0]] - curve_dist
    print traj_mod_start_val

    start_idx, start_val = find_nearest(dist_array,traj_mod_start_val)
    end_idx, end_val = find_nearest(traj, tr_point)

    vel_curve, sigmoid_timeline = vp.get_full_velocity_curve()
    # print sigmoid_timeline
    # print timeline.size
    # print vel
    # print int(1.1*vel/tr_vel)
    sigmoid_timeline = np.append(sigmoid_timeline, np.asarray([sigmoid_timeline[-1]+(sigmoid_timeline[-1]-sigmoid_timeline[-2])*(n+1) for n in range(int((timeline.size - end_idx)*1.1*vel/tr_vel))]))

    vel_curve = np.append(vel_curve,np.asarray([tr_vel for _ in range(sigmoid_timeline.size - vel_curve.size)]))

    # print sigmoid_timeline

    # pos, sigmoid_timeline = vp.get_full_distance_curve()
    pos = get_distance_from_vel_curve(vel_curve, sigmoid_timeline)
    # print pos.size
    pos = pos[pos<=dist_array[-1]-dist_array[start_idx]]
    # print pos.size
    sigmoid_timeline = sigmoid_timeline[:pos.size]
    vel_curve = vel_curve[:pos.size]
    # print start_idx
    # print (traj[start_idx:].size)
    act_pos_reshaped = interp1d(np.arange(traj[start_idx:].size), traj[start_idx:],kind='cubic')(
        np.linspace(0, traj[start_idx:].size-1, pos.size))
    dist_to_pos_map = interp1d(np.linspace(pos[0],pos[-1],pos.size), act_pos_reshaped,kind='cubic')

    act_time_to_dist_map = interp1d(sigmoid_timeline,pos,kind='cubic')

    updated_pos_part = dist_to_pos_map(act_time_to_dist_map(sigmoid_timeline))

    sigmoid_timeline += timeline[start_idx]

    final_traj = np.append(traj[:start_idx],updated_pos_part)
    final_timeline = np.append(timeline[:start_idx],sigmoid_timeline)
    
    vel_curve = np.append(np.asarray([vel for _ in range(start_idx)]),vel_curve)

    return final_traj, final_timeline#, vel_curve



if __name__ == "__main__":
    
    traj = np.linspace(35,0,5000)
    timeline = np.linspace(0,5,5000)

    tr_point = 3
    tr_vel = 0.5

    new_traj, new_timeline = get_modified_traj(traj, timeline, tr_point, tr_vel)

    vel = np.gradient(new_traj,new_timeline)
    # vel_map = interp1d(new_timeline,vel)
    # print vel.size, new_timeline.size
    acc = np.gradient(vel,new_timeline)

    acc_map = UnivariateSpline(new_timeline,acc,s=50.0)
    acc = acc_map(new_timeline)
    
    plt.subplot(3,1,1)
    plt.ylabel("Vel Profile")
    plt.plot(new_timeline,vel,label='vel')
    # plt.
    # plt.plot(new_timeline[new_traj<=3],vel[new_traj<=3])
    plt.grid()
    # plt.xlim([4,5.5])

    # plt.xlim(0,1)
    plt.subplot(3,1,2)
    plt.plot(new_timeline,new_traj,label='pos')
    plt.plot(timeline,traj)
    # plt.plot(new_timeline[new_traj<=3],new_traj[new_traj<=3])
    # plt.plot(timeline,tot_pos,label='old_pos')
    plt.ylabel("Pos Traj")
    plt.grid()
    # plt.xlim([4,5.5])
    # plt.xlim(0,1)
    plt.subplot(3,1,3)
    plt.plot(new_timeline,acc,label='acc')
    # plt.plot(new_timeline[new_traj<=3],acc[new_traj<=3])
    plt.ylabel("Acc Profile")
    plt.grid()
    # plt.xlim([4,5.5])
    # plt.xlim(850,950)
    plt.xlabel('time')
    plt.legend()
    plt.show()
