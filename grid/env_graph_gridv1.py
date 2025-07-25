from copy import copy
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from scipy.special import softmax
import gym
from gym import spaces
from matplotlib import cm, colors


class GoalWrapper:
    def __init__(self, env, trayectories):
        self.trayectories = trayectories
        self.env = env

    def step(self, actions, emb):
        obs, _, terminated, info = self.env.step(actions, emb)


class GraphEnv(gym.Env):
    def __init__(
        self,
        config,
        goal,
        max_time=23,
        board_size=10,
        sensing_range=6,
        pad=4,
        starting_positions=None,
        obstacles=None,
    ):
        super(GraphEnv, self).__init__()
        """
        :starting_positions: np.array-> [nb_agents, positions]; positions == [X,Y]
                            [[0,0],
                             [1,1]]
        """
        self.config = config
        self.max_time = self.config["max_time"]
        self.min_time = self.config["min_time"]
        self.board_size = self.config["board_size"][0]
        if obstacles is not None:
            self.obstacles = obstacles
        else:
            self.obstacles = None
        self.goal = goal
        self.board = np.zeros((self.board_size, self.board_size))
        self.pad = pad
        self.starting_positions = starting_positions
        self.action_list = {
            1: (1, 0),  # Right
            2: (0, 1),  # Up
            3: (-1, 0),  # Left
            4: (0, -1),  # Down
            0: (0, 0),  # Idle
        }
        nb_agents = self.config["num_agents"]
        self.positionX = np.zeros((nb_agents, 1), dtype=np.int32)
        self.positionY = np.zeros((nb_agents, 1), dtype=np.int32)
        self.nb_agents = nb_agents
        self.sensing_range = sensing_range
        self.obs_shape = self.nb_agents * 4
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=-15, high=15, shape=((self.obs_shape,)), dtype=np.float32
        )
        self.headings = None
        self.embedding = np.ones(self.nb_agents)
        norm = colors.Normalize(vmin=0.0, vmax=1.4, clip=True)
        self.mapper = cm.ScalarMappable(norm=norm, cmap=cm.inferno)
        self.time = 0
        _ = self.reset()

    def reset(self):
        self.time = 0
        self.avilable_pos = np.arange(self.board_size)
        self.board = np.zeros((self.board_size, self.board_size))
        if self.obstacles is not None:
            self.board[self.obstacles[:, 1], self.obstacles[:, 0]] = 2
        if self.starting_positions is not None:
            assert (
                self.starting_positions.shape[0] == self.nb_agents
            ), f"Agents and positions are not equal"
            self.positionX = self.starting_positions[:, 0]
            self.positionY = self.starting_positions[:, 1]
        else:
            self.avilable_pos_x = np.arange(self.board_size)
            self.avilable_pos_y = np.arange(self.board_size)
            if self.obstacles is not None:
                mask_x = np.isin(self.avilable_pos_x, self.obstacles[:, 0])
                mask_y = np.isin(self.avilable_pos_y, self.obstacles[:, 1])
                self.avilable_pos_x = self.avilable_pos_x[~mask_x]
                self.avilable_pos_y = self.avilable_pos_y[~mask_y]
            self.positionX = np.random.choice(
                self.avilable_pos_x, size=(self.nb_agents)
            )
            self.positionY = np.random.choice(
                self.avilable_pos_y, size=(self.nb_agents)
            )

        self.goal_paded = self.goal + self.pad
        self.headings = np.random.uniform(-3.14, 3.14, size=(self.nb_agents))
        self.embedding = np.ones(self.nb_agents).reshape((self.nb_agents, 1))
        self.reached_goal = np.zeros(self.nb_agents)
        self._computeDistance()
        return self.getObservations()

    def getObservations(self):
        obs = {
            # "positionX": self.positionX,
            # "positionY": self.positionY,
            # "headings": self.headings,
            "board": self.updateBoardGoal(),
            "fov": self.preprocessObs(),
            "adj_matrix": self.adj_matrix,
            "distances": self.distance_matrix,
            "embeddings": self.embedding,
        }
        return obs

    def getGraph(self):
        return self.adj_matrix

    def getEmbedding(self):
        return copy(self.embedding)

    def getPositions(self):
        return np.array([self.positionX, self.positionY]).T

    def _computeDistance(self):
        # Create Matrices from positions and heading
        X1, XT = np.meshgrid(self.positionX, self.positionX)
        Y1, YT = np.meshgrid(self.positionY, self.positionY)

        # Calculate distance matrix
        D_ij_x = X1 - XT
        D_ij_y = Y1 - YT
        D_ij = np.sqrt(np.multiply(D_ij_x, D_ij_x) + np.multiply(D_ij_y, D_ij_y))
        D_ij[D_ij >= self.sensing_range] = 0

        self.distance_matrix = D_ij
        # Get only closest 4
        self.adj_matrix = self._computeClosest(D_ij)
        self.adj_matrix[self.adj_matrix != 0] = 1

    def computeMetrics(self):
        # Agent-level success: fraction of agents that reached their goals
        positions = np.array([self.positionX, self.positionY]).T
        agent_success = np.all(positions == self.goal, axis=1)
        success_rate = float(np.mean(agent_success))
        # count how many agents reached their goals
        num_reached = int(np.sum(agent_success))
        total_agents = self.nb_agents

        flow_time = self.computeFlowTime()

        return success_rate, flow_time, total_agents, num_reached

    def checkAllInGoal(self):
        last_state = np.array([self.positionX, self.positionY]).T
        # return np.sum(success[0]) == self.nb_agents
        return np.array_equal(last_state, self.goal)

    def check_goals(self):
        positions = np.array([self.positionX, self.positionY]).T
        positions = np.where(positions == self.goal, self.goal, positions)
        # self.positionX = np.where(self.positionX_temp == self.goal[:,0],self.goal[:,0], self.positionX)
        # self.positiony = np.where(self.positionY_temp == self.goal[:, 1],self.goal[:,1], self.positionY)
        self.positionX, self.positionY = positions[:, 0], positions[:, 1]

    def computeFlowTime(self):
        if self.checkAllInGoal():
            return self.time
        else:
            return self.nb_agents * self.max_time

    @staticmethod
    def _computeClosest(A):
        for i in range(len(A)):
            temp = np.sort(A[i][A[i] != 0])
            if len(temp) < 4:
                temp = np.concatenate((np.zeros(4 - len(temp)), temp))
            A[i][A[i] > temp[3]] = 0
        return A

    def step(self, actions, emb):
        """
        Actions: {
          vx[list], shape(nb_agents)
          vy[list], shape(nb_agents)
        }
        """
        done = False
        self._updateEmbedding(emb)
        self._updatePositions(actions)
        self._computeDistance()
        obs = self.getObservations()
        self.time += 1
        if self.checkAllInGoal():
            done = True
        return obs, {}, done, {}

    def _updatePositions(self, actions):

        action_x = np.array([self.action_list[act][0] for act in actions])
        action_y = np.array([self.action_list[act][1] for act in actions])
        self.positionX_temp = copy(self.positionX)
        self.positionY_temp = copy(self.positionY)
        self.positionX += action_x
        self.positionY += action_y
        self.check_goals()
        if self.obstacles is not None:
            self.check_collision_obstacle()
        self.check_boundary()
        self.check_collisions()
        self.updateBoard()
        # print(self.board)
        # self.positionX += actions["vx"]
        # self.positionY += actions["vy"]
        # self.headings  += actions["headings"]

    def _updateEmbedding(self, H):
        self.embedding = H

    def map_goal(self, agent):
        # Check if it's in the FOV
        if (
            self.goal_paded[agent][0] < self.posx[agent] + self.pad - 1
            and self.goal_paded[agent][0] > self.posx[agent] - self.pad + 1
        ):
            goal_x = -(self.posx[agent] - self.goal_paded[agent][0]) + self.pad - 1

        # Check if it's in the left or right of the FOV
        elif self.goal_paded[agent][0] <= self.posx[agent] - self.pad + 1:
            goal_x = 0
        elif self.goal_paded[agent][0] >= self.posx[agent] + self.pad - 1:
            goal_x = 1 + self.pad

        # Same for Y
        if (
            self.goal_paded[agent][1] < self.posy[agent] + self.pad - 1
            and self.goal_paded[agent][1] >= self.posy[agent] - self.pad + 1
        ):
            goal_y = (self.posy[agent] - self.goal_paded[agent][1]) + self.pad - 1

        elif self.goal_paded[agent][1] <= self.posy[agent] - self.pad + 1:
            goal_y = 1 + self.pad

        elif self.goal_paded[agent][1] >= self.posy[agent] + self.pad - 1:
            goal_y = 0

        goal = np.array([goal_y, goal_x])  # Reversed for numpy
        return goal[0], goal[1]

    def preprocessObs(self):
        self.posx = self.positionX + self.pad
        self.posy = self.positionY + self.pad
        map_padded = np.pad(self.board, (self.pad, self.pad))
        FOV = np.zeros((self.nb_agents, 2, (self.pad * 2) - 1, (self.pad * 2) - 1))

        for agent in range(self.nb_agents):
            FOV[agent, 0, :, :] = np.flip(
                map_padded[
                    self.positionY[agent] : self.positionY[agent] + 7,
                    self.positionX[agent] : self.positionX[agent] + 7,
                ],
                axis=0,
            )
            gx, gy = self.map_goal(agent=agent)
            FOV[agent, 1, gx, gy] = 3

        return FOV

    def check_boundary(self):
        self.positionX[self.positionX > self.board_size - 1] = self.board_size - 1
        self.positionY[self.positionY > self.board_size - 1] = self.board_size - 1
        self.positionX[self.positionX < 0] = 0
        self.positionY[self.positionY < 0] = 0

    def updateBoard(self):
        self.board[self.positionY_temp, self.positionX_temp] = 0
        self.board[self.positionY, self.positionX] = 1

    def updateBoardGoal(self):
        board = copy(self.board)
        board[self.goal[:, 1], self.goal[:, 0]] += 4
        return board

    def check_collisions(self):
        """
        Iterate over X:
        if 2 equal, check their Y:
            If equal, revert to position in temp
        """
        ck = {}
        for i in range(len(self.positionX)):
            hash = str((self.positionX[i], self.positionY[i]))
            if hash in ck:
                self.positionX[i] = self.positionX_temp[i]
                self.positionY[i] = self.positionY_temp[i]
                self.positionX[int(ck[hash])] = self.positionX_temp[int(ck[hash])]
                self.positionY[int(ck[hash])] = self.positionY_temp[int(ck[hash])]
                continue
            ck[hash] = i

    def check_collision_obstacle(self):
        ck = {
            str((self.obstacles[i][0], self.obstacles[i][1])): i
            for i in range(len(self.obstacles))
        }
        for i in range(len(self.positionX)):
            hash = str((self.positionX[i], self.positionY[i]))
            if hash in ck:
                self.positionX[i] = self.positionX_temp[i]
                self.positionY[i] = self.positionY_temp[i]

    def printBoard(self):
        self.updateBoard()
        return f"Game Board:\n{self.board}"

    def render(self, agentId=0, printNeigh=False, printFOV=False, mode="plot"):

        plt.axis("off")
        if agentId is not None:
            column = np.where(self.adj_matrix[agentId])
            column = column[0]
            for i in range(len(column)):
                plt.plot(
                    [self.positionX[agentId], self.positionX[column[i]]],
                    [self.positionY[agentId], self.positionY[column[i]]],
                    color="black",
                )
                if printNeigh:
                    neig_column = np.where(self.adj_matrix[column[i]])
                    neig_column = neig_column[0]
                    for j in range(len(neig_column)):
                        plt.plot(
                            [self.positionX[column[i]], self.positionX[neig_column[j]]],
                            [self.positionY[column[i]], self.positionY[neig_column[j]]],
                            color="black",
                            ls="--",
                        )
        else:
            for agent in range(self.nb_agents):
                column = np.where(self.adj_matrix[agent])
                column = column[0]
                for i in range(len(column)):
                    plt.plot(
                        [self.positionX[agent], self.positionX[column[i]]],
                        [self.positionY[agent], self.positionY[column[i]]],
                        color="black",
                    )
                    if printNeigh:
                        neig_column = np.where(self.adj_matrix[column[i]])
                        neig_column = neig_column[0]
                        for j in range(len(neig_column)):
                            plt.plot(
                                [
                                    self.positionX[column[i]],
                                    self.positionX[neig_column[j]],
                                ],
                                [
                                    self.positionY[column[i]],
                                    self.positionY[neig_column[j]],
                                ],
                                color="black",
                            )

        if mode == "plot":
            plt.scatter(
                self.positionX,
                self.positionY,
                s=150,
                color=self.mapper.to_rgba(self.embedding),
            )
            plt.scatter(
                self.goal[:, 0], self.goal[:, 1], color="blue", marker="*", s=150
            )
            if self.obstacles is not None:
                plt.scatter(
                    self.obstacles[:, 0],
                    self.obstacles[:, 1],
                    color="black",
                    marker="s",
                    s=150,
                )
        if mode == "photo":
            plt.imshow(self.board, cmap="Greys")

        # printing FOV
        if printFOV:
            plt.plot(
                [
                    self.positionX[agentId] - self.sensing_range * 3 / 4,
                    self.positionX[agentId] + self.sensing_range * 3 / 4,
                ],
                [
                    self.positionY[agentId] - self.sensing_range * 3 / 4,
                    self.positionY[agentId] - self.sensing_range * 3 / 4,
                ],
                color="red",
            )
            plt.plot(
                [
                    self.positionX[agentId] - self.sensing_range * 3 / 4,
                    self.positionX[agentId] - self.sensing_range * 3 / 4,
                ],
                [
                    self.positionY[agentId] - self.sensing_range * 3 / 4,
                    self.positionY[agentId] + self.sensing_range * 3 / 4,
                ],
                color="red",
            )
            plt.plot(
                [
                    self.positionX[agentId] - self.sensing_range * 3 / 4,
                    self.positionX[agentId] + self.sensing_range * 3 / 4,
                ],
                [
                    self.positionY[agentId] + self.sensing_range * 3 / 4,
                    self.positionY[agentId] + self.sensing_range * 3 / 4,
                ],
                color="red",
            )
            plt.plot(
                [
                    self.positionX[agentId] + self.sensing_range * 3 / 4,
                    self.positionX[agentId] + self.sensing_range * 3 / 4,
                ],
                [
                    self.positionY[agentId] - self.sensing_range * 3 / 4,
                    self.positionY[agentId] + self.sensing_range * 3 / 4,
                ],
                color="red",
            )
        # Printing env stuff
        plt.axis([-2, self.board_size + 5, -2, self.board_size + 5])
        plt.plot(
            [-1, self.board_size],
            [
                -1,
                -1,
            ],
            color="black",
        )
        plt.plot(
            [
                -1,
                -1,
            ],
            [self.board_size, -1],
            color="black",
        )
        plt.plot(
            [-1, self.board_size], [self.board_size, self.board_size], color="black"
        )
        plt.plot(
            [self.board_size, self.board_size], [self.board_size, -1], color="black"
        )
        plt.pause(0.1)
        plt.clf()
        plt.axis("off")


########## utils ##########
def create_goals(board_size, num_agents, obstacles=None):
    avilable_pos_x = np.arange(board_size[0])
    avilable_pos_y = np.arange(board_size[1])
    if obstacles is not None:
        mask_x = np.isin(avilable_pos_x, obstacles[:, 0])
        mask_y = np.isin(avilable_pos_y, obstacles[:, 1])
        avilable_pos_x = avilable_pos_x[~mask_x]
        avilable_pos_y = avilable_pos_y[~mask_y]
    goals_x = np.random.choice(avilable_pos_x, size=num_agents, replace=False)
    goals_y = np.random.choice(avilable_pos_y, size=num_agents, replace=False)
    goals = np.array([goals_x, goals_y]).T
    return goals


def create_obstacles(board_size, nb_obstacles):
    avilable_pos = np.arange(board_size[0])
    obstacles_x = np.random.choice(avilable_pos, size=nb_obstacles, replace=False)
    obstacles_y = np.random.choice(avilable_pos, size=nb_obstacles, replace=False)
    obstacles = np.array([obstacles_x, obstacles_y]).T
    return obstacles


if __name__ == "__main__":
    agents = 2
    board_size = 16
    config = {
        "num_agents": agents,
        "board_size": [board_size],
        "max_time": 23,
        "min_time": 16,
    }
    sensing = 4
    start = np.array([[6, 6], [3, 3]])
    goals = np.array([[4, 3], [7, 7]])
    obstacles = np.array([[2, 2], [2, 3]])
    env = GraphEnv(
        config,
        goal=goals,
        board_size=board_size,
        sensing_range=sensing,
        starting_positions=start,
        obstacles=obstacles,
    )
    emb = np.ones(agents).reshape((agents, 1))
    obs = env.reset()
    actions = np.zeros((7, agents))
    plt.ion()
    actions[:, 0] = np.array([4, 4, 4, 3, 3, 3, 3]).T
    actions[:, 1] = np.array([0, 0, 0, 3, 4, 4, 4]).T
    for i in range(7):
        """
        1:(1,0), # Right
        2:(0,1), # Up
        3:(-1,0),# Left
        4:(0,-1), # Down
        0:(0,0)  # Idle
        """
        obs, _, _, _ = env.step(actions[i, :], emb)
        env.render(agentId=0, printNeigh=True)
