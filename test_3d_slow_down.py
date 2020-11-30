import numpy as np
import matplotlib.pyplot as plt
from pos_traj_modifier import get_slowed_down_traj
from scipy.interpolate import UnivariateSpline, interp1d

if __name__ == "__main__":
    
    traj = np.linspace(np.asarray([35.,-12,2]),np.asarray([35.,-12,-25]),10000)
    timeline = np.linspace(0,6,10000)
    # print traj.shape
    # get_traj_representations(traj, timeline)

    
    tr_point = traj[4000,:]#[-0.98559712 , -1.00440088 ,-24.98919784]
    tr_vel = 0.5
    # print traj
    # print find_nearest(traj,tr_point)

    new_traj, new_timeline = get_slowed_down_traj(traj, timeline, tr_point, tr_vel, smooth_start=True)
    pos_traj = new_traj[:,2]
    # print new_traj.shape, new_timeline.shape
    # new_traj_map = interp1d(new_timeline, new_traj, axis=0)
    # new_timeline = np.linspace(new_timeline[0],new_timeline[-1],10000)
    # pos_traj = new_traj_map(new_timeline)[:,2]
    # pos_traj = np.linalg.norm(new_traj, axis=1)
    # pos_traj = new_traj[:,2]
    # new_timeline = new_timeline[:-5]
    # pos_traj -= pos_traj[0]

    vel = np.gradient(pos_traj,new_timeline)
    # # vel_map = interp1d(new_timeline,vel)
    # # print vel.size, new_timeline.size
    acc = np.gradient(vel,new_timeline)
    # print acc
    acc[acc>100]=0
    # acc_map = UnivariateSpline(new_timeline,acc,s=50.0)
    # acc = acc_map(new_timeline)
    
    plt.subplot(3,1,1)
    plt.ylabel("Vel Profile")
    plt.plot(new_timeline,vel,label='vel')
    plt.grid()

    # plt.xlim(0,1)
    plt.subplot(3,1,2)
    plt.plot(new_timeline,pos_traj,label='pos')
    # plt.plot(timeline,tot_pos,label='old_pos')
    plt.ylabel("Pos Traj")
    # plt.xlim([3.5,5.1])
    plt.grid()
    # plt.xlim([4,5.5])
    # plt.xlim(0,1)
    plt.subplot(3,1,3)
    plt.plot(new_timeline,acc,label='acc')
    plt.ylabel("Acc Profile")
    plt.grid()
    # plt.xlim([4,5.5])
    # plt.xlim(850,950)
    plt.xlabel('time')
    plt.legend()
    plt.show()
