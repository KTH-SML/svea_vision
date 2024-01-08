import matplotlib.pyplot as plt
import random
from svea_vision.kalman_filter import KF
from math import sin, cos, pi, atan2


# Linear
def linear_test(T):
    s = [[0, 0], 1, 0]
    m_traj = [[0, 0]]
    kf = KF(1, *s)

    truth_traj_x = [i/10 for i in range(0, T)]
    truth_traj_y = [i/10 + random.randrange(-10, 10)/10 for i in range(0, T)]

    for t in range(1, T):
        kf.predict()
        phi = t/5  # Derivative of truth_traj_y equation
        z = [truth_traj_x[t], truth_traj_y[t], 1, phi]
        kf.update(z)
        m_traj.append(kf.x[0:2])
    
    m_traj_x, m_traj_y = [], []
    for t in range(T):
        m_traj_x.append(m_traj[t][0])
        m_traj_y.append(m_traj[t][1])

    return (truth_traj_x, truth_traj_y), (m_traj_x, m_traj_y)

# Circle
def circle_test(radius, T):
    def PointsInCircum(r, n=100):
        return [(cos(2*pi/n*x)*r, sin(2*pi/n*x)*r) for x in range(0, n+1)]
    

    circle = PointsInCircum(radius, T)
    circle_traj_x, circle_traj_y = [], []
    for x, y in circle:
        circle_traj_x.append(x+random.randrange(-10, 10)/10)
        circle_traj_y.append(y+random.randrange(-10, 10)/10)
    s = [[radius, 0], 1, pi]
    kf = KF(1, *s)

    c_traj = [[radius, 0]]

    for t in range(1, T):
        kf.predict()
        diff_y = circle[t][1] - circle[t-1][1]
        diff_x = circle[t][0] - circle[t-1][0]
        # Derivative of circle_traj_y equation
        phi = atan2(diff_y, diff_x)
        z = [circle_traj_x[t], circle_traj_y[t], 1, phi]
        kf.update(z)
        c_traj.append(kf.x[0:2])

    c_traj_x, c_traj_y = [], []
    for t in range(T):
        c_traj_x.append(c_traj[t][0])
        c_traj_y.append(c_traj[t][1])

    return (circle_traj_x, circle_traj_y), (c_traj_x, c_traj_y)

if __name__ == "__main__":
    T = 100
    radius = 10

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.ion()
    fig.suptitle("Kalman filter estimates of different trajectories")
    ax1.set_xlim([-1, T/10+1]), ax1.set_ylim([-1, T/10+1])
    ax2.set_xlim([-radius*1.4, radius*1.4]
                 ), ax2.set_ylim([-radius*1.4, radius*1.4])
    
    # Axis 1 = Linear
    (truth_traj_x, truth_traj_y), (m_traj_x, m_traj_y) = linear_test(T)
    ax1.plot(truth_traj_x[0], truth_traj_y[0], color="green")
    ax1.plot(m_traj_x[0], m_traj_y[0], color="orange")
    ax1.grid()
    ax1.set_xlabel("x"), ax1.set_ylabel("y")
    ax1.legend(["Noisy meas.", "KF est."])
    ax1.set_title("Linear")

    # Axis 2 = Circle
    (circle_traj_x, circle_traj_y), (c_traj_x, c_traj_y) = circle_test(radius, T)
    ax2.plot(circle_traj_x[0], circle_traj_y[0], color="green")
    ax2.plot(c_traj_x[0], c_traj_y[0], color="orange")
    ax2.grid()
    ax2.set_xlabel("x"), ax2.set_ylabel("y")
    ax2.legend(["Noisy meas.", "KF est."])
    ax2.set_title("Circle")

    for t in range(1, T):
        # Axis 1 = Linear
        ax1.plot(truth_traj_x[0:t], truth_traj_y[0:t], color="green")
        ax1.plot(m_traj_x[0:t], m_traj_y[0:t], color="orange")

        # Axis 2 = Circle
        ax2.plot(circle_traj_x[0:t], circle_traj_y[0:t], color="green")
        ax2.plot(c_traj_x[0:t], c_traj_y[0:t], color="orange")
        plt.pause(0.005)
        plt.show()
        
    plt.pause(5)
