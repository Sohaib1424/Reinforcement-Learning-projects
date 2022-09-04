import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import gym

# Agent Class for Classic Control Problems
class Agent:

    def __init__(self, env):

        self.Env = env
        
        # booleans to check methods used for training
        self.trained = {
            "q_learning": False,
            "sarsa": False,
            "d_sarsa": False,
            "actor_critic": False
        }

        self.q_table = {
            "table": None, # the actual Q-table
            "sizes": None # window sizes
        }

        self.weights = {
            "sarsa": None,
            "d_sarsa": None,
            "actor_critic": None
        }

        self.theta = {
            "actor_critic": None
        }

        # received reward per episode for each method
        self.rewards = {
            "q_learning": [],
            "sarsa": [],
            "d_sarsa": [],
            "actor_critic": []
        }

        self.tilings = {
            "sarsa": None,
            "d_sarsa": None,
            "actor_critic": None
        }

        self.tiles = {
            "sarsa": 0,
            "d_sarsa": 0,
            "actor_critic": 0
        }




    def SolveWith_Q_Learning(self, alpha=0.01, state_discretization=[20, 20],
                            episodes=30000, display_every=2000,
                            discount=0.95, epsilon=0.05):

        self.Env["env"] = gym.make(self.Env["name"])
        
        if self.Env["dim"] != len(state_discretization):
            raise EnvironmentError("dimensions mismatch.")

        if state_discretization.__contains__(0):
            raise EnvironmentError("discretization by zero is not possible.")

        if (not 0 < alpha <= 1) or (not 0 < discount <= 1) or (not 0 <= epsilon <= 1) or episodes <= 0 or display_every < 0:
            raise EnvironmentError("wrong inputs.")

        self.trained["q_learning"] = True
        self.rewards["q_learning"] = []

        self.q_table["sizes"] = (self.Env["max"] - self.Env["min"]) / np.array(state_discretization)
        self.q_table["table"] = Create_Q_Table(state_discretization + [self.Env["num_actions"]])

        # training
        for i in range(episodes):

            done = False # boolean to check episode completion
            current_state = self.Env["env"].reset()
            discrete_current_state = GetDiscreteStates(current_state, self.Env["min"], self.q_table["sizes"])
            current_action = self.Env["env"].action_space.sample() if np.random.random() < epsilon else np.argmax(self.q_table["table"][discrete_current_state])

            reward_per_episode = 0

            while not done:
                
                new_state, reward, done, _ = self.Env["env"].step(current_action)
                discrete_new_state = GetDiscreteStates(new_state, self.Env["min"], self.q_table["sizes"])
                new_action = np.argmax(self.q_table["table"][discrete_new_state])

                if not done:
                    # updating Q-table using the bellman equation
                    self.q_table["table"][discrete_current_state][current_action] += alpha * (reward + 
                                                                            discount * self.q_table["table"][discrete_new_state][new_action] - 
                                                                            self.q_table["table"][discrete_current_state][current_action])

                    discrete_current_state = deepcopy(discrete_new_state)
                    current_action = self.Env["env"].action_space.sample() if np.random.random() < epsilon else new_action


                elif self.Env["goal_pos"] != None and new_state[0] >= self.Env["goal_pos"]:
                    # updating Q-table using the bellman equation
                    self.q_table["table"][discrete_current_state][current_action] += alpha * (reward - 
                                                                            self.q_table["table"][discrete_current_state][current_action])

                reward_per_episode += reward

                if i % display_every == 0:
                    self.Env["env"].render()

            self.rewards["q_learning"].append(reward_per_episode)

        self.Env["env"].close()


    def SolveWith_SARSA(self, alpha=0.1, num_tilings=8, num_tiles=10,
                             episodes=3000, display_every=300, epsilon=0.05, discount=0.95):

        self.Env["env"] = gym.make(self.Env["name"])

        if num_tilings <= 0 or num_tiles <= 0:
            raise EnvironmentError("discretization by zero is not possible.")

        if (not 0 < alpha <= 1) or (not 0 < discount <= 1) or (not 0 <= epsilon <= 1) or episodes <= 0 or display_every < 0:
            raise EnvironmentError("wrong inputs.")

        alpha /= num_tilings
        
        self.tiles["sarsa"] = num_tiles
        self.trained["sarsa"] = True
        self.rewards["sarsa"] = []

        # uniform offset tiling
        sizes = (self.Env["max"] - self.Env["min"]) / (num_tilings * num_tiles)
        offsets = np.array([(i - num_tilings/2) * sizes for i in range(num_tilings)])

        # initializing tilings and weights
        self.tilings["sarsa"] = CreateTilings(self.Env["min"], self.Env["max"], num_tilings, num_tiles, offsets, self.Env["dim"])
        self.weights["sarsa"] = np.zeros((self.Env["num_actions"], num_tilings * num_tiles**self.Env["dim"]))

        # training
        for i in range(episodes):
            
            reward_per_episode = 0

            done = False # boolean to check episode completion
            current_state = self.Env["env"].reset()
            current_active_tiles = GetActiveTiles(current_state, self.tilings["sarsa"], num_tiles, self.Env["dim"])
            current_action, current_action_values = SelectAction(current_active_tiles, self.weights["sarsa"], epsilon, self.Env["num_actions"])
            
            while not done:
                
                new_state, reward, done, _ = self.Env["env"].step(current_action)
                new_active_tiles = GetActiveTiles(new_state, self.tilings["sarsa"], num_tiles, self.Env["dim"])
                new_action, new_action_values = SelectAction(new_active_tiles, self.weights["sarsa"], epsilon, self.Env["num_actions"])
                
                if not done:

                    TD_error = reward + discount * new_action_values[new_action] - current_action_values[current_action]
                    # binary representation of active tiles for the selected action
                    x = np.zeros_like(self.weights["sarsa"]).astype(np.int64)
                    x[current_action][current_active_tiles] = 1
                
                    self.weights["sarsa"] += alpha * TD_error * x
                    
                    # or we could've just wrote
                    # self.weights["sarsa"][current_action][current_active_tiles] += alpha * (reward + discount * new_action_values[new_action] - current_action_values[current_action])
                    
                    current_state = new_state
                    current_active_tiles = deepcopy(new_active_tiles)
                    current_action = new_action
                    current_action_values = deepcopy(new_action_values)
                
                elif self.Env["goal_pos"] != None and new_state[0] >= self.Env["goal_pos"]:

                    TD_error = reward - current_action_values[current_action]
                    # binary representation of active tiles for the selected action
                    x = np.zeros_like(self.weights["sarsa"]).astype(np.int64)
                    x[current_action][current_active_tiles] = 1
                
                    self.weights["sarsa"] += alpha * TD_error * x
                
                reward_per_episode += reward

                if i % display_every == 0:
                    self.Env["env"].render()

            self.rewards["sarsa"].append(reward_per_episode)

        self.Env["env"].close()



    def SolveWith_Diff_SARSA(self, num_tilings=8, alpha=0.1, beta=0.01, num_tiles=10,
                            episodes=3000, display_every=300, epsilon=0.05, discount=1.):

        self.Env["env"] = gym.make(self.Env["name"])

        alpha /= num_tilings

        self.tiles["d_sarsa"] = num_tiles
        self.trained["d_sarsa"] = True
        self.rewards["d_sarsa"] = []
        # uniform offset tiling
        sizes = (self.Env["max"] - self.Env["min"]) / (num_tilings * num_tiles)
        offsets = np.array([(i - num_tilings/2) * sizes for i in range(num_tilings)])

        # initializing tilings and weights
        self.tilings["d_sarsa"] = CreateTilings(self.Env["min"], self.Env["max"], num_tilings, num_tiles, offsets, self.Env["dim"])
        self.weights["d_sarsa"] = np.zeros((self.Env["num_actions"], num_tilings * num_tiles**self.Env["dim"]))


        avg_reward = 0

        # training
        for i in range(episodes):
            
            reward_per_episode = 0

            done = False # boolean to check episode completion
            current_state = self.Env["env"].reset()
            current_active_tiles = GetActiveTiles(current_state, self.tilings["d_sarsa"], num_tiles, self.Env["dim"])
            current_action, current_action_values = SelectAction(current_active_tiles, self.weights["d_sarsa"], epsilon, self.Env["num_actions"])
            
            while not done:
                
                new_state, reward, done, _ = self.Env["env"].step(current_action)
                new_active_tiles = GetActiveTiles(new_state, self.tilings["d_sarsa"], num_tiles, self.Env["dim"])
                new_action, new_action_values = SelectAction(new_active_tiles, self.weights["d_sarsa"], epsilon, self.Env["num_actions"])
                
                if not done:

                    # TD-error
                    delta = reward - avg_reward + discount * new_action_values[new_action] - current_action_values[current_action]
                    avg_reward += beta * delta

                    # binary representation of active tiles for the selected action
                    x = np.zeros_like(self.weights["d_sarsa"]).astype(np.int64)
                    x[current_action][current_active_tiles] = 1
                
                    self.weights["d_sarsa"] += alpha * delta * x
                    
                    # or we could've just wrote
                    # self.weights["d_sarsa"][current_action][current_active_tiles] += alpha * (reward + discount * new_action_values[new_action] - current_action_values[current_action])
                    
                    current_state = new_state
                    current_active_tiles = deepcopy(new_active_tiles)
                    current_action = new_action
                    current_action_values = deepcopy(new_action_values)
                
                elif self.Env["goal_pos"] != None and new_state[0] >= self.Env["goal_pos"]:

                    # TD-error
                    delta = reward - avg_reward + discount * new_action_values[new_action] - current_action_values[current_action]
                    avg_reward += beta * delta

                    # binary representation of active tiles for the selected action
                    x = np.zeros_like(self.weights["d_sarsa"]).astype(np.int64)
                    x[current_action][current_active_tiles] = 1
                
                    self.weights["d_sarsa"] += alpha * delta * x


                reward_per_episode += reward

                if i % display_every == 0:
                    self.Env["env"].render()

            self.rewards["d_sarsa"].append(reward_per_episode)

        self.Env["env"].close()

                
        
    def SolveWith_Actor_Critic(self, step_sizes=[.3, .3, .1], num_tilings=10, num_tiles=10,
                                     discount=1., episodes=3000, display_every=300, softmax=True):

        self.Env["env"] = gym.make(self.Env["name"])
        

        step_sizes = np.array(step_sizes) / num_tilings
        
        self.tiles["actor_critic"] = num_tiles
        self.trained["actor_critic"] = True
        self.rewards["actor_critic"] = []
        # uniform offset tiling
        sizes = (self.Env["max"] - self.Env["min"]) / (num_tilings * num_tiles)
        offsets = np.array([(i - num_tilings/2) * sizes for i in range(num_tilings)])

        # initializing tilings and weights
        self.tilings["actor_critic"] = CreateTilings(self.Env["min"], self.Env["max"], num_tilings, num_tiles, offsets, self.Env["dim"])
        self.weights["actor_critic"] = np.zeros((1, num_tilings * num_tiles**self.Env["dim"]))
        self.theta["actor_critic"] = np.zeros((self.Env["num_actions"], num_tilings * num_tiles**self.Env["dim"]))


        avg_reward = 0

        # training
        for i in range(episodes):
            
            reward_per_episode = 0

            done = False # boolean to check episode completion
            current_state = self.Env["env"].reset()
            current_active_tiles = GetActiveTiles(current_state, self.tilings["actor_critic"], num_tiles, self.Env["dim"])
            current_action, current_action_values = SelectAction(current_active_tiles, self.theta["actor_critic"], 0, self.Env["num_actions"], softmax=softmax)
            
            # binary representation of active tiles
            x = np.zeros_like(self.weights["actor_critic"][0]).astype(np.int64)
            x[current_active_tiles] = 1
            
            current_state_value = (self.weights["actor_critic"] @ x)[0]
            # or we could've just wrote
            # current_state_value = np.sum(self.ACS_w[0][current_active_tiles])

            while not done:
                
                new_state, reward, done, _ = self.Env["env"].step(current_action)
                new_active_tiles = GetActiveTiles(new_state, self.tilings["actor_critic"], num_tiles, self.Env["dim"])

                # binary representation of new active tiles
                x = np.zeros_like(self.weights["actor_critic"][0]).astype(np.int64)
                x[new_active_tiles] = 1

                new_state_value = (self.weights["actor_critic"] @ x)[0]
                # or we could've just wrote
                # new_state_value = np.sum(self.weights["actor_critic"][0][new_active_tiles])

                if not done:

                    # TD-error
                    delta = reward - avg_reward + discount * new_state_value - current_state_value
                    avg_reward += step_sizes[1] * delta

                    # binary representation of active tiles
                    x = np.zeros_like(self.weights["actor_critic"]).astype(np.int64)
                    x[0][current_active_tiles] = 1

                    self.weights["actor_critic"] += step_sizes[0] * delta * x
                    # or we could've just wrote
                    # self.weights["actor_critic"][current_action][current_active_tiles] += alpha * (reward + discount * new_state_value - current_state_value
                    
                    # binary representation of active tiles
                    x = np.zeros_like(self.theta["actor_critic"]).astype(np.int64)
                    x[:, current_active_tiles] = 1
                    
                    z = np.zeros_like(self.theta["actor_critic"]).astype(np.int64)
                    z[current_action][current_active_tiles] = 1
                    
                    derivatives = z - current_action_values * x
                                        
                    self.theta["actor_critic"] += step_sizes[1] * delta * derivatives 
                    # or we could've just wrote
                    # self.theta["actor_critic"][:, current_active_tiles] += step_sizes[1] * delta * derivatives

                    # selecting new action AFTER the upadte
                    current_state = deepcopy(new_state)
                    current_state_value = new_state_value
                    current_active_tiles = deepcopy(new_active_tiles)
                    current_action, current_action_values = SelectAction(current_active_tiles, self.theta["actor_critic"], 0, self.Env["num_actions"], softmax=softmax)
                
                elif self.Env["goal_pos"] != None and new_state[0] >= self.Env["goal_pos"]:

                    # TD-error
                    delta = reward - avg_reward - current_state_value
                    avg_reward += step_sizes[1] * delta

                    # binary representation of active tiles
                    x = np.zeros_like(self.weights["actor_critic"]).astype(np.int64)
                    x[0][current_active_tiles] = 1

                    self.weights["actor_critic"] += step_sizes[0] * delta * x
                    # or we could've just wrote
                    # self.weights["actor_critic"][current_action][current_active_tiles] += alpha * (reward + discount * new_state_value - current_state_value
                    
                    # binary representation of active tiles
                    x = np.zeros_like(self.theta["actor_critic"]).astype(np.int64)
                    x[:, current_active_tiles] = 1
                    
                    z = np.zeros_like(self.theta["actor_critic"]).astype(np.int64)
                    z[current_action][current_active_tiles] = 1
                    
                    derivatives = z - current_action_values * x
                                        
                    self.theta["actor_critic"] += step_sizes[1] * delta * derivatives 
                    # or we could've just wrote
                    # self.theta["actor_critic"][:, current_active_tiles] += step_sizes[1] * delta * derivatives

                reward_per_episode += reward

                if i % display_every == 0:
                    self.Env["env"].render()

            self.rewards["actor_critic"].append(reward_per_episode)

        self.Env["env"].close()


    def Test(self, method):

        self.Env["env"] = gym.make(self.Env["name"])
        
        if method == "q_learning":
            done = False
            # current state
            discrete_current_state = GetDiscreteStates(self.Env["env"].reset(), self.Env["min"], self.q_table["sizes"])
            # sleceting an action to take according to the desired method
            current_action = np.argmax(self.q_table["table"][discrete_current_state])
        
            while not done:
                
                new_state, reward, done, _ = self.Env["env"].step(current_action)
                # discretizing new state
                discrete_new_state = GetDiscreteStates(new_state, self.Env["min"], self.q_table["sizes"])
                new_action = np.argmax(self.q_table["table"][discrete_new_state])
                
                if not done:
                    discrete_current_state = deepcopy(discrete_new_state)
                    current_action = new_action

                self.Env["env"].render()
            self.Env["env"].close()

        else:
            done = False
            current_state = self.Env["env"].reset()
            current_active_tiles = GetActiveTiles(current_state, self.tilings[method], self.tiles[method], self.Env["dim"])
            current_action, _ = SelectAction(current_active_tiles, self.weights[method] if method != "actor_critic" else self.theta["actor_critic"], 0, self.Env["num_actions"], True)
            
            while not done:
                
                new_state, _, done, _ = self.Env["env"].step(current_action)
                new_active_tiles = GetActiveTiles(new_state, self.tilings[method], self.tiles[method], self.Env["dim"])
                new_action, _ = SelectAction(new_active_tiles, self.weights[method] if method != "actor_critic" else self.theta["actor_critic"], 0, self.Env["num_actions"], True)
                
                if not done:
                    current_active_tiles = deepcopy(new_active_tiles)
                    current_action = new_action

                    self.Env["env"].render()

            self.Env["env"].close()

    def plot(self, plot_max=True, plot_min=True):

        plt.figure(figsize=(16, 10))
        plt.title(self.Env["name"])

        if self.trained["q_learning"]:
            plt.plot([i for i in range(len(self.rewards["q_learning"])//100)],
                [np.mean(self.rewards["q_learning"][i:i+100]) for i in range(0, len(self.rewards["q_learning"]), 100)],
                 color="blue", label="mean q_learning")
            if plot_max:
                plt.plot([i for i in range(len(self.rewards["q_learning"])//100)],
                    [np.max(self.rewards["q_learning"][i:i+100]) for i in range(0, len(self.rewards["q_learning"]), 100)],
                    color="blue", linestyle=":", label="max q_learning")
            if plot_min:
                plt.plot([i for i in range(len(self.rewards["q_learning"])//100)],
                    [np.min(self.rewards["q_learning"][i:i+100]) for i in range(0, len(self.rewards["q_learning"]), 100)],
                    color="blue", linestyle="--", label="min q_learning")

        if self.trained["sarsa"]:
            plt.plot([i for i in range(len(self.rewards["sarsa"])//100)],
             [np.mean(self.rewards["sarsa"][i:i+100]) for i in range(0, len(self.rewards["sarsa"]), 100)],
              color="green", label="mean sarsa")

            if plot_max:
                plt.plot([i for i in range(len(self.rewards["sarsa"])//100)],
                    [np.max(self.rewards["sarsa"][i:i+100]) for i in range(0, len(self.rewards["sarsa"]), 100)],
                    color="green", linestyle=":", label="max sarsa")

            if plot_min:
                plt.plot([i for i in range(len(self.rewards["sarsa"])//100)],
                    [np.min(self.rewards["sarsa"][i:i+100]) for i in range(0, len(self.rewards["sarsa"]), 100)],
                    color="green", linestyle="--", label="min sarsa")

        if self.trained["d_sarsa"]:
            plt.plot([i for i in range(len(self.rewards["d_sarsa"])//100)],
             [np.mean(self.rewards["d_sarsa"][i:i+100]) for i in range(0, len(self.rewards["d_sarsa"]), 100)],
              color="red", label="mean d_sarsa")

            if plot_max:
                plt.plot([i for i in range(len(self.rewards["d_sarsa"])//100)],
                    [np.max(self.rewards["d_sarsa"][i:i+100]) for i in range(0, len(self.rewards["d_sarsa"]), 100)],
                    color="red", linestyle=":", label="max d_sarsa")

            if plot_min:
                plt.plot([i for i in range(len(self.rewards["d_sarsa"])//100)],
                    [np.min(self.rewards["d_sarsa"][i:i+100]) for i in range(0, len(self.rewards["d_sarsa"]), 100)],
                    color="red", linestyle="--", label="min d_sarsa")

        if self.trained["actor_critic"]:
            plt.plot([i for i in range(len(self.rewards["actor_critic"])//100)],
             [np.mean(self.rewards["actor_critic"][i:i+100]) for i in range(0, len(self.rewards["actor_critic"]), 100)],
              color="black", label="mean actor_critic")

            if plot_max:
                plt.plot([i for i in range(len(self.rewards["actor_critic"])//100)],
                    [np.max(self.rewards["actor_critic"][i:i+100]) for i in range(0, len(self.rewards["actor_critic"]), 100)],
                    color="black", linestyle=":", label="max actor_critic")

            if plot_min:
                plt.plot([i for i in range(len(self.rewards["actor_critic"])//100)],
                    [np.min(self.rewards["actor_critic"][i:i+100]) for i in range(0, len(self.rewards["actor_critic"]), 100)],
                    color="black", linestyle="--", label="min actor_critic")     

        plt.legend()   
        plt.show()



# function for creating q-table
def Create_Q_Table(size):
    return np.random.uniform(low=-2, high=0, size=size)

# function for discretizing states for q_table
def GetDiscreteStates(state, minimum, sizes):
    discrete_state = ((state - minimum) / sizes).astype(np.int64)
    return tuple(x for x in discrete_state)

 # function for creating tilings
def CreateTilings(start, stop, num_tilings, num_tiles, offsets, dimension):
    
    default_tiling = np.array([np.linspace(start=start[i], stop=stop[i], num=num_tiles-1)[:num_tiles-2] for i in range(dimension)])
    
    tilings = []
    # creating the rest of the tilings
    for i in range(num_tilings):
        tilings.append(np.array([default_tiling[j] + offsets[i][j] for j in range(dimension)]))
        
    return np.array(tilings)


def GetActiveTiles(state, tilings, num_tiles, dimension):
    active_tiles = []

    for i, tiling in enumerate(tilings):
        tile = np.zeros(dimension)
        
        for j in range(dimension):
                        
            if tiling[j][0] < state[j] < tiling[j][-1]:
                tile[j] = int(num_tiles * (state[j] - tiling[j][0]) / (tiling[j][-1] - tiling[j][0]))
                        
            elif tiling[j][-1] < state[j]:
                tile[j] = len(tiling[j]) - 1

        
        active_tile = i * num_tiles**dimension
                        
        for k in range(dimension):
            active_tile += tile[k] * num_tiles**k

        active_tiles.append(active_tile)
            
    return np.array(active_tiles).astype(np.int64)


def SelectAction(active_tiles, weights, epsilon, num_actions, softmax=False):
    action = 0
    action_values = []
    
    # loops `num_actions` times
    # calculates W.T @ X for each action where X is the binary representation of the active and deactive tiles
    # for simplicity we do not preform W.T @ X, instead we perform the equivalent operation below        
    for action_value in weights:
        # binary representation of tiles, activ tiles are `1` and deactive tiles are `0`
        x = np.zeros_like(action_value).astype(np.int64)
        x[active_tiles] = 1
        # W.T @ X
        WX = action_value.T @ x       
        action_values.append(WX)
        # or we could just write the code below            
        # action_values.append(np.sum(action_value[active_tiles]))
    
    action_values = np.array(action_values).reshape(num_actions, 1)
                        
    if softmax:
        action_values = Softmax(action_values)
        action = np.random.choice(num_actions, p=action_values[:, 0])

    else:
        if np.random.random() < epsilon:
            action = np.random.choice(num_actions)
        else:
            action = np.argmax(action_values)

    return action, action_values


def Softmax(x):
    x = x - np.max(x, keepdims=True)
    exp_x = np.exp(x)
    exp_x /= np.sum(exp_x, axis=0, keepdims=True)
    return exp_x