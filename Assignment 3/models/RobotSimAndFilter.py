import random
import numpy as np

class RobotSim:
    def __init__(self, t_m, s_m):
        self.t_m = t_m
        self.s_m = s_m

        self.rows, self.cols, self.head = self.s_m.get_grid_dimensions()
        # Direction arrays
        self.dir_ls = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]
        self.dir_ls_2 = [(0, -2), (-1, -2), (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
                         (-1, 2), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (2, -1), (2, -2), (1, -2)]

    # Return the headings that the robot can move in
    def get_headings(self, tsX, tsY):
        headings = []
        if tsX > 0:
            headings.append(0)  # North
        if tsY < self.rows - 1:
            headings.append(1)  # East
        if tsX < self.cols - 1:
            headings.append(2)  # South
        if tsY > 0:
            headings.append(3)  # West

        return headings

    # Generates a random next state according to rules
    def new_state(self, true_state):
        tsX, tsY, tsH = self.s_m.state_to_pose(true_state)
        headings = self.get_headings(tsX, tsY)

        prob = random.random()
        # Picks a heading according to the robot strategy
        if tsH in headings:
            if prob <= 0.3:
                headings.remove(tsH)
                tsH = random.choice(headings)
        else:
            tsH = random.choice(headings)

        # Moves one step in the direction of heading
        if tsH == 0:  # North
            tsX -= 1
        elif tsH == 1:  # East
            tsY += 1
        elif tsH == 2:  # South
            tsX += 1
        elif tsH == 3:  # West
            tsY -= 1

        return self.s_m.pose_to_state(tsX, tsY, tsH)

    def sensor_reading(self, tsX, tsY):
        # use direction array to calculate ls and ls2
        ls = [(tsX + i, tsY + j) for i, j in self.dir_ls]
        ls_2 = [(tsX + i, tsY + j) for i, j in self.dir_ls_2]

        # remove elements that is out of bounds of the grid
        ls = self.filter(ls)
        ls_2 = self.filter(ls_2)

        prob = random.random()
        # According to what the sensor reports
        ls_prob = 0.1 + len(ls) * 0.05
        if prob <= 0.1:
            return self.s_m.position_to_reading(tsX, tsY)
        elif prob <= ls_prob:
            choice = random.choice(ls)
            return self.s_m.position_to_reading(choice[0], choice[1])
        elif prob <= ls_prob + len(ls_2) * 0.025:
            choice = random.choice(ls_2)
            return self.s_m.position_to_reading(choice[0], choice[1])

        return None

    def filter(self, l):
        for index in l.copy():
            if index[0] >= self.rows or index[0] < 0 or index[1] >= self.cols or index[1] < 0:
                l.remove(index)
        return l

class HMMFilter:
    def __init__(self, t_m, o_m):
        self.t_m = t_m
        self.o_m = o_m

    # Forward filtering according to 14.3.1 in the course book
    def forward_filter(self, last_probs, reading):
        f = self.t_m.get_T_transp()
        o_reading = self.o_m.get_o_reading(reading)
        probs = o_reading @ f @ last_probs
        # Normalized vector
        alpha = 1 / np.linalg.norm(probs)
        return alpha * probs
