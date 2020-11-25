import numpy as np
from tanh_vel_profile import HyperbolicTangentVelocityProfile as TanhVP
from sigmoid_vel_profile import SigmoidVelocityProfile as SigVP
from scipy.interpolate import UnivariateSpline, interp1d
import matplotlib.pyplot as plt

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def time_dilate_trajectory(pos_traj, times):
    # print pos_traj, times
    # raw_input()
    times = np.abs(times)
    x = np.asarray(range(pos_traj.shape[0]))*times
    f1 = interp1d(x, pos_traj)
    # f2 = interp1d(x, ori_traj, kind='nearest', axis=0)

    # plt.plot(f1(range(int(x[-1])+1)))
    # plt.show()
    # print 
    return f1(range(int(x[-1])+1))

def get_modified_traj(traj, timeline, tr_point, tr_vel):

    dist_array = np.cumsum(np.diff(traj,prepend=traj[0]))
    # print dist_array
    # print dist_array.size
    vel = np.gradient(traj,timeline).mean()
    # print vel

    vp = TanhVP(vel,tr_vel,1.,int(timeline.size/(timeline[-1]-timeline[0])))
    # print vp.get_velocity_at(1.)
    curve_dist = np.round(vp.get_total_distance_at(1.),3)
    print curve_dist

    traj_mod_start_val = tr_point - curve_dist
    print traj_mod_start_val

    start_idx, start_val = find_nearest(traj,traj_mod_start_val)
    end_idx, end_val = find_nearest(traj, tr_point)

    pos, sigmoid_timeline = vp.get_full_distance_curve()

    act_pos_reshaped = interp1d(np.arange(traj[start_idx:end_idx].size), traj[start_idx:end_idx],kind='cubic')(
        np.linspace(0, traj[start_idx:end_idx].size-1, pos.size))
    dist_to_pos_map = interp1d(np.linspace(pos[0],pos[-1],pos.size), act_pos_reshaped,kind='cubic')

    act_time_to_dist_map = interp1d(sigmoid_timeline,pos,kind='cubic')

    updated_pos_part = dist_to_pos_map(act_time_to_dist_map(sigmoid_timeline))

    sigmoid_timeline += timeline[start_idx]

    final_traj = (np.append(traj[:start_idx],updated_pos_part) )
    final_timeline = np.append(timeline,np.asarray([timeline[-1]+(n+1)*0.001 for n in range(final_traj.size - timeline.size)]))
    

    return final_traj, final_timeline



if __name__ == "__main__":
    
    traj = np.linspace(35,0,5000)
    timeline = np.linspace(0,5,5000)

    tr_point = 3
    tr_vel = -0.5

    new_traj, new_timeline = get_modified_traj(traj, timeline, tr_point, tr_vel)

    vel = np.gradient(new_traj,new_timeline)
    acc = np.gradient(vel,new_timeline)
    
    plt.subplot(3,1,1)
    plt.ylabel("Vel Profile")
    plt.plot(new_timeline,vel,label='vel')
    plt.grid()

    # plt.xlim(0,1)
    plt.subplot(3,1,2)
    plt.plot(new_timeline,new_traj,label='pos')
    # plt.plot(timeline,tot_pos,label='old_pos')
    plt.ylabel("Pos Traj")
    plt.grid()
    # plt.xlim(0,1)
    plt.subplot(3,1,3)
    plt.plot(new_timeline,acc,label='acc')
    plt.ylabel("Acc Profile")
    plt.grid()
    # plt.xlim(850,950)
    plt.xlabel('time')
    plt.legend()
    plt.show()
