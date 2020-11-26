import numpy as np
from tanh_vel_profile import HyperbolicTangentVelocityProfile as TanhVP
from sigmoid_vel_profile import SigmoidVelocityProfile as SigVP
from scipy.interpolate import UnivariateSpline, interp1d
import matplotlib.pyplot as plt
from utils import get_distance_from_vel_curve

def find_nearest(array, value):
    if len(array.shape) == 1:
        array = array.reshape([-1,1])
    # if not isinstance(value,float):
    else:
        value = np.asarray(value).reshape(1,array.shape[1])

    idx = np.linalg.norm(array - value,axis=1).argmin()
    return idx, array[idx]

def get_traj_representations(traj, timeline, one_d = True):
    """
    Return the cumulative distance trajectory, velocity trajectory for the provided trajectory

    :param traj: N x D array for the D-dimensional trajectory consistiing of N points in the D-dim space
    :type traj: np.ndarray
    :param timeline: N x 1 array of Time (normally in sec) corresponding to each point in the trajectory
    :type timeline: np.ndarray
    """
    dist_array = np.cumsum(np.abs(np.diff(traj,prepend=traj[0,:].reshape([1,-1]),axis=0)),axis=0)
    vel = np.gradient(traj,timeline,axis=0)

    if one_d:
        dist_array = np.linalg.norm(dist_array, axis=1)
        vel = np.linalg.norm(vel,axis=1)

    return dist_array, vel

def get_modified_traj(traj, timeline, tr_point, tr_vel, transition_duration=1.):
    """
    Get the modified position trajectory when the velocity has to be switched to 'tr_vel' smoothly
    by the time it enters 'tr_point'. For optimality, the velocity transition should be completed
    exactly as it reaches 'tr_point'. The tanH or sigmoid velocity profile is applied before this
    point. Returns the new position trajectory and associated timeline.

    :param traj: N x D array for the D-dimensional position trajectory consisting of N D-dimensional points
    :type traj: np.ndarray
    :param timeline: N x 1 array of corresponding timestamps (in sec)
    :type timeline: np.ndarray
    :param tr_point: the D-dimensional point at which the velocity should be made to be 'tr_vel'
    :type tr_point: np.ndarray
    :param tr_vel: the desired speed to be applied from 'tr_point' to end of trajectory
    :type tr_vel: float
    :param transition_duration: duration of transition from previous velocity to 'tr_vel', defaults to 1 sec.
    :type transition_duration: float, optional
    :return: modified (M x D) position trajectory and (M x 1) timeline
    :rtype: np.ndarray, np.ndarray
    """

    # get the cumulative distance for the trajectory, and it's 1d speed profile
    # The cumsum distance is used as a 1D representation of the trajectory
    dist_array, ori_vel_prof = get_traj_representations(traj, timeline, one_d=True)
    ori_vel_avg = ori_vel_prof.mean() # assuming this is the velocity at the region where the switching has to start

    timeline = timeline.flatten()

    # create a tanh velocity profile that transitions from the detected mean velocity to desired velocity in the specified transition time. The resolution is adjusted to match the same as that of the trajectory
    vp = TanhVP(ori_vel_avg, tr_vel, transition_duration, int(timeline.size*transition_duration/(timeline[-1]-timeline[0])))

    # find the index in the original trajectory where the tr_point is closest
    tanh_end_idx, tanh_end_val = find_nearest(traj,tr_point)

    # Find the total distance attained at the end of the transition_duration when following the transition velocity profile
    tanh_curve_dist = vp.get_total_distance_at(transition_duration)

    # On the true trajectory, the required starting point can be found by subtracting the tanh_curve_dist from the cumulative distance traversed to reach 'tr_point'
    tanh_start_dist = dist_array[tanh_end_idx] - tanh_curve_dist
    tanh_start_idx, tanh_start_val = find_nearest(dist_array, tanh_start_dist)

    # get the transition velocity curve and timeline, and add uniform velocity curve at its end to maintain the 'tr_vel'
    tanh_vel_curve, tanh_timeline = vp.get_full_velocity_curve()
    transition_timeline = np.append(tanh_timeline, np.asarray([tanh_timeline[-1]+(tanh_timeline[-1]-tanh_timeline[-2])*(n+1) for n in range(int((timeline.size - tanh_end_idx)*1.1*ori_vel_avg/tr_vel))])) # the tail length depends on the ratio between the new and old velocity + extra to be safe
    transition_vel_curve = np.append(tanh_vel_curve,np.asarray([tr_vel for _ in range(transition_timeline.size - tanh_vel_curve.size)]))

    # get total distance curve when using this new velocity profile
    pos = get_distance_from_vel_curve(transition_vel_curve, transition_timeline)
    pos = pos[pos<=dist_array[-1]-dist_array[tanh_start_idx]] # remove extra part from tail
    transition_timeline = transition_timeline[:pos.size]

    # create a distance to position map for the region in traj that starts with transition, after interpolating it to same size as pos
    act_pos_reshaped = interp1d(np.arange(dist_array[tanh_start_idx:].size), traj[tanh_start_idx:,:],axis=0, kind='cubic')(
        np.linspace(0, dist_array[tanh_start_idx:].size-1, pos.size))
    dist_to_pos_map = interp1d(np.linspace(pos[0],pos[-1],pos.size), act_pos_reshaped,kind='cubic', axis=0)

    # find the modified trajectory mapped by the updated timeline
    act_time_to_dist_map = interp1d(transition_timeline,pos,kind='cubic') # create a mapping from new timeline to new distance traj
    updated_pos_part = dist_to_pos_map(act_time_to_dist_map(transition_timeline)) # get the position traj for the new time line

    transition_timeline += timeline[tanh_start_idx] # update start of transition timeline to begin with the end of the prev part

    # create final position traj by appending the newly created position traj to the end of the original trajectory's first part
    final_traj = np.vstack([traj[:tanh_start_idx,:],updated_pos_part])
    final_timeline = np.append(timeline[:tanh_start_idx],transition_timeline)
    
    return final_traj, final_timeline

def get_modified_traj_old(traj, timeline, tr_point, tr_vel):
    
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
    
    traj = np.linspace(np.asarray([35.,-12,2]),np.asarray([-1,-1,-25]),5000)
    timeline = np.linspace(0,6,5000)
    # print traj.shape
    # get_traj_representations(traj, timeline)

    
    tr_point = traj[4000,:]#[-0.98559712 , -1.00440088 ,-24.98919784]
    tr_vel = 0.5
    # print traj
    # print find_nearest(traj,tr_point)

    new_traj, new_timeline = get_modified_traj(traj, timeline, tr_point, tr_vel)

    print new_traj.shape, new_timeline.shape


    # vel = np.gradient(new_traj,new_timeline)
    # # vel_map = interp1d(new_timeline,vel)
    # # print vel.size, new_timeline.size
    # acc = np.gradient(vel,new_timeline)

    # acc_map = UnivariateSpline(new_timeline,acc,s=50.0)
    # acc = acc_map(new_timeline)
    
    plt.subplot(3,1,1)
    plt.ylabel("Vel Profile")
    plt.plot(new_timeline,new_traj[:,0],label='vel')
    plt.grid()
    # plt.xlim([4,5.5])

    # plt.xlim(0,1)
    plt.subplot(3,1,2)
    plt.plot(new_timeline,new_traj[:,1],label='pos')
    # plt.plot(timeline,tot_pos,label='old_pos')
    plt.ylabel("Pos Traj")
    plt.grid()
    # plt.xlim([4,5.5])
    # plt.xlim(0,1)
    plt.subplot(3,1,3)
    plt.plot(new_timeline,new_traj[:,2],label='acc')
    plt.ylabel("Acc Profile")
    plt.grid()
    # plt.xlim([4,5.5])
    # plt.xlim(850,950)
    plt.xlabel('time')
    plt.legend()
    plt.show()
