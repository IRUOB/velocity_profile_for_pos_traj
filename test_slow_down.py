import numpy as np
import matplotlib.pyplot as plt
from pos_traj_modifier import get_slowed_down_traj, smooth_start_traj

if __name__ == "__main__":
    
    traj = np.linspace(35,0,5000).reshape([-1,1])
    timeline = np.linspace(0,5,5000)

    tr_point = 3
    tr_vel = 0.5

    new_traj, new_timeline = get_slowed_down_traj(traj, timeline, tr_point, tr_vel, smooth_start=False, transition_duration=0.5)
    # new_traj, new_timeline = smooth_start_traj(traj, timeline, smooth_start_duration=1.8)
    # print new_traj.shape, new_timeline.shape
    vel = np.gradient(new_traj,new_timeline, axis=0)
    # vel_map = interp1d(new_timeline,vel)
    # print vel.size, new_timeline.size
    acc = np.gradient(vel,new_timeline, axis=0)

    jerk = np.gradient(acc, new_timeline, axis=0)
    # acc_map = UnivariateSpline(new_timeline,acc,s=50.0)
    # acc = acc_map(new_timeline)
    
    plt.subplot(4,1,1)
    plt.ylabel("Vel Profile")
    plt.plot(new_timeline,vel,label='vel')
    # plt.
    # plt.plot(new_timeline[new_traj<=3],vel[new_traj<=3])
    plt.grid()
    # plt.xlim([4,5.5])

    # plt.xlim(0,1)
    plt.subplot(4,1,2)
    plt.plot(new_timeline,new_traj,label='pos')
    plt.plot(timeline,traj)
    # plt.plot(new_timeline[new_traj<=3],new_traj[new_traj<=3])
    # plt.plot(timeline,tot_pos,label='old_pos')
    plt.ylabel("Pos Traj")
    plt.grid()
    # plt.xlim([4,5.5])
    # plt.xlim(0,1)
    plt.subplot(4,1,3)
    plt.plot(new_timeline,acc,label='acc')
    # plt.plot(new_timeline[new_traj<=3],acc[new_traj<=3])
    plt.ylabel("Acc Profile")
    plt.grid()

    plt.subplot(4, 1, 4)
    plt.plot(new_timeline, jerk, label='acc')
    plt.ylabel("Jerk Profile")
    plt.grid()
    # plt.xlim([4,5.5])
    # plt.xlim(850,950)
    plt.xlabel('time')
    plt.legend()
    plt.show()
