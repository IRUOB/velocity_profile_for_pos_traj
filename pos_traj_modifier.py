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

def get_slowed_down_traj(traj, timeline, tr_point, tr_vel, transition_duration=1.):
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
