#! /usr/bin/env python3

from collections import deque
import numpy as np
from math import sin, cos, atan2
import rospy
from el2425_standards.msg import PersonState, PersonStateArray # TODO
from geometry_msgs.msg import Pose
from rsu_msgs.msg import StampedObjectPoseArray
from scipy.optimize import curve_fit
from kalman_filter import KF


class PersonPrediction:
    """ Class that estimates the states of each detected person 
        (x, y, v, phi) by interpolating the locations upto 
        'max_history_len'. """

    person_tracker_dict = dict()
    person_states = dict()
    kf_dict = dict()
    kf_state_tracker = dict()
    frequency = 14.5
    max_history_len = 6  # Used for trajectory prediction
    drop_id_after = 4  # frames
    threshold = 0.5  # TODO: Set threshold better?

    def __init__(self):
        rospy.init_node("object_state_estimation", anonymous=True)
        self.pub = rospy.Publisher("~person_states",
                                   PersonStateArray,
                                   queue_size=10)
        self.start()

    def __listener(self):
        """ Subscribes to the topic containing only detected 
            persons and applies the function __callback. """
        rospy.Subscriber("classification_splitter/persons",
                         StampedObjectPoseArray,
                         self.__callback)
        while not rospy.is_shutdown():
            rospy.spin()

    def __callback(self, msg):
        """ The function for interpolating persons locations and 
            then publishing is to a separate topic 'estimations/person_states'. 
            This implementation keeps sending the states of persons who have 
            dropped out of frame, because we don't know if the person has gone 
            out of the camera's view or dropped randomly, regardless, 
            we still send information about this.  """
        personstatearray_msg = PersonStateArray()
        personstatearray_msg.header = msg.header
        for person in msg.objects:
            # Get the person's ID and current location
            person_id = person.object.id
            person_loc = (person.pose.pose.position.x,
                          person.pose.pose.position.y,
                          person.pose.pose.position.z)
            # Check if new measurement is close to previous stored measurements and
            # set person_id if criteria is met.
            # TODO: Fix this.
            #   -   Scenario: If 2 people are in the frame and close to each other
            #       and 1 of them is suddenly dropped, then new measurements of the
            #       dropped person would probably be too close to the one that is
            #       still tracked, thus only tracking one person.
            if self.person_tracker_dict:
                min_dist = 1000
                for p_id, p_loc in self.person_tracker_dict.items():
                    dist = np.linalg.norm(np.array(person_loc[0:2]) -
                                          np.array(p_loc[-1][0:2]))
                    if dist < min_dist:
                        min_dist = dist
                        if min_dist < self.threshold:
                            person_id = p_id
            # Append the current person's location in the dict
            if person_id in self.person_tracker_dict:
                self.person_tracker_dict[person_id].append(person_loc)
            else:
                # Initiate a deque that only can contain the wanted length of historical locations
                self.person_tracker_dict[person_id] = deque([person_loc],
                                                            maxlen=self.max_history_len)
            # Estimate the state
            if len(self.person_tracker_dict[person_id]) == self.max_history_len:
                # Get the velocity and heading used for kalman filter estimation
                v, phi = self.fit(self.person_tracker_dict[person_id])
                # Run the Kalman filter
                if not self.kf_dict.get(person_id):
                    self.kf_dict[person_id] = KF(person_id,
                                                 [person_loc[0], person_loc[1]],
                                                 v, phi, self.frequency)
                else:
                    self.kf_dict[person_id].predict()
                    z = [person_loc[0], person_loc[1], v, phi]
                    self.kf_dict[person_id].update(z)
                kf_state = self.kf_dict[person_id].x
                # Check if the person_id is in the kf_state_tracker, append its (x,y) pos
                # or create new deque list.
                if person_id in self.kf_state_tracker:
                    self.kf_state_tracker[person_id].append((kf_state[0],
                                                            kf_state[1]))
                else:
                    self.kf_state_tracker[person_id] = deque([(kf_state[0], kf_state[1])],
                                                             maxlen=self.max_history_len)
                # Use the KF estimate for calculating the velocity and heading
                if len(self.kf_state_tracker[person_id]) == self.max_history_len:
                    # If the kf state tracker is large enough, we can fit the new heading and
                    # velocity using the pedestrian locations from the Kalman filter.
                    v, phi = self.fit(self.kf_state_tracker[person_id])
                else:
                    # Otherwise, we just use the bad estimate
                    v, phi = kf_state[2], kf_state[3]
                state = PersonState()
                pose = Pose()
                euler_angle = np.array([[cos(phi), -sin(phi), 0],
                                        [sin(phi), cos(phi), 0],
                                        [0, 0, 1]])
                quaternion = rotationMatrixToQuaternion1(euler_angle)
                pose.position.x, pose.position.y, pose.position.z = (kf_state[0],
                                                                     kf_state[1],
                                                                     person_loc[2])
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quaternion
                state.id = person_id
                state.pose = pose
                state.velocity = v
                state.direction = phi
                state.counter = msg.header.seq
                # Update the dictionary with {ID: PersonState}
                self.person_states[person_id] = state

        # Cleanup the dictionary of persons
        self.__clean_up_dict(msg.header.seq)
        # Put the list of personstates in the message and publish it
        personstatearray_msg.personstates = list(
            self.person_states.values())
        self.pub.publish(personstatearray_msg)

    def interpolate_data(self, Q: deque):
        """         DEPRECATED; use fit instead.
            Interpolate locations to velocity and heading.
            IN  :   queue of locations
            OUT :   (velocity, heading) """
        x1, y1, _ = np.array(Q[-1])
        x0, y0, _ = np.array(Q[0])
        v = np.linalg.norm(
            np.array([x1, y1])-np.array([x0, y0])) * self.frequency / self.max_history_len
        phi = atan2(y1-y0, x1-x0)
        return float(v), float(phi)

    def fit(self, Q: deque):
        """ Fit the trajectory of the previous positions in order to get a 
            better estimate of current velocity and heading.
            IN  :   queue of locations
            OUT :   (velocity, heading) """
        t = np.linspace(0, len(Q), self.max_history_len) * 1/self.frequency
        x, y = np.zeros(len(Q)), np.zeros(len(Q))
        for i, q in enumerate(Q):
            x[i], y[i] = q[0], q[1]
        poptx, *_ = curve_fit(objective, t, x)
        popty, *_ = curve_fit(objective, t, y)
        ax, bx, cx = poptx
        ay, by, cy = popty
        xs = objective(t, ax, bx, cx)
        ys = objective(t, ay, by, cy)
        c0, c1 = (xs[-2], ys[-2]), (xs[-1], ys[-1])
        return self.__interpolate_coords(c0, c1)

    def __interpolate_coords(self, c0, c1):
        """ Coordinate interpolation
            IN  :   tuple(x0,z0), tuple(x1,z1) 
            OUT :   velocity, heading"""
        x1, y1 = np.array(c1)
        x0, y0 = np.array(c0)
        v = np.linalg.norm(
            np.array([x1, y1])-np.array([x0, y0])) * self.frequency / self.max_history_len
        phi = atan2(y1-y0, x1-x0)
        return float(v), float(phi)

    def __clean_up_dict(self, current_count):
        id_to_drop = []
        for id, state in self.person_states.items():
            c = state.counter
            if current_count - c >= self.drop_id_after:
                id_to_drop.append(id)
        for id in id_to_drop:
            self.person_states.pop(id)
            self.kf_dict.pop(id)

    def start(self):
        self.__listener()


def objective(x, a, b, c):
    """ The objective function that is used for fitting 
        previous positions. """
    return a*x**2 + b*x + c


def rotationMatrixToQuaternion1(m):
    t = np.matrix.trace(m)
    q = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    if (t > 0):
        t = np.sqrt(t + 1)
        q[3] = 0.5 * t
        t = 0.5/t
        q[0] = (m[2, 1] - m[1, 2]) * t
        q[1] = (m[0, 2] - m[2, 0]) * t
        q[2] = (m[1, 0] - m[0, 1]) * t

    else:
        i = 0
        if (m[1, 1] > m[0, 0]):
            i = 1
        if (m[2, 2] > m[i, i]):
            i = 2
        j = (i+1) % 3
        k = (j+1) % 3

        t = np.sqrt(m[i, i] - m[j, j] - m[k, k] + 1)
        q[i] = 0.5 * t
        t = 0.5 / t
        q[3] = (m[k, j] - m[j, k]) * t
        q[j] = (m[j, i] + m[i, j]) * t
        q[k] = (m[k, i] + m[i, k]) * t
    q[1] = -q[1]

    return q


if __name__ == "__main__":
    predictions = PersonPrediction()
