import numpy as np
import matplotlib.pyplot as plt
from pos_traj_modifier import get_slowed_down_traj

if __name__ == "__main__":
    
    traj = np.linspace(np.asarray([35.,-12,2]),np.asarray([-1,-1,-25]),5000)
    timeline = np.linspace(0,6,5000)
    # print traj.shape
    # get_traj_representations(traj, timeline)

    
    tr_point = traj[4000,:]#[-0.98559712 , -1.00440088 ,-24.98919784]
    tr_vel = 0.5
    # print traj
    # print find_nearest(traj,tr_point)

    new_traj, new_timeline = get_slowed_down_traj(traj, timeline, tr_point, tr_vel)

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
