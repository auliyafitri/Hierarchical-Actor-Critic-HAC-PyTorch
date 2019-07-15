import torch
import gym
import numpy as np
from HAC import HAC

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():
    #################### Hyperparameters ####################
    # env_name = "MountainCarContinuous-v0"
    env_name = "FetchPickAndPlace-v1"
    save_episode = 10               # keep saving every n episodes
    max_episodes = 1000             # max num of training episodes
    random_seed = 0
    render = True
    
    env = gym.make(env_name)
    env = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation'])

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    goal_dim = env.goal.shape[0]

    print("state dim {} action dim {}".format(state_dim, action_dim))
    
    """
     Actions (both primitive and subgoal) are implemented as follows:
       action = ( network output (Tanh) * bounds ) + offset
       clip_high and clip_low bound the exploration noise
    """
    
    # primitive action bounds and offset
    # action_bounds = env.action_space.high[0]
    # action_offset = np.array([0.0])
    action_bounds = np.ones(action_dim) * 2.0
    action_bounds = torch.FloatTensor(action_bounds.reshape(1, -1)).to(device)
    action_offset = np.ones(action_dim) * 0.01
    action_offset = torch.FloatTensor(action_offset.reshape(1, -1)).to(device)
    # action_clip_low = np.array([-1.0 * action_bounds])
    # action_clip_high = np.array([action_bounds])
    action_clip_low = env.action_space.low
    action_clip_high = env.action_space.high

    """
    observation:
    grip_pos,               3
    object_pos.ravel(),     3
    object_rel_pos.ravel(), 3
    gripper_state,            2
    object_rot.ravel(),     3
    object_velp.ravel(),    3   positional(cartesian) velocity in world frame whereas xvelr is rotational velocity
    object_velr.ravel(),    3
    grip_velp,              3
    gripper_vel               2

    ###
    observation bound
    3  x [low_valid_area.x, high_valid_area.x]
         [low_valid_area.y, high_valid_area.y]
         [low_valid_area.z, high_valid_area.z]
    2  x [0, 0.05]
    3  x [-pi,pi]
    11 x [-1,1]
    """

    valid_area_high = np.array([1.369, 1.387, 1.314])
    valid_area_low = np.array([0.725, 0.1, 0.686])

    # state bounds and offset
    state_bounds = np.append(
        np.array([
            [0.725, 1.369],
            [0.1, 1.387],
            [0.686, 1.314],
            [0.725, 1.369],
            [0.1, 1.387],
            [0.686, 1.314],
            [0.725, 1.369],
            [0.1, 1.387],
            [0.686, 1.314],
            [0, 0.05],
            [0, 0.05],
            [-np.pi, np.pi],
            [-np.pi / 2, np.pi / 2],
            [-np.pi, np.pi]
        ]),
        np.ones([11, 2]) * [-1, 1], axis=0)
    # state_bounds_np = np.array([0.9, 0.07])
    # state_bounds = torch.FloatTensor(state_bounds_np.reshape(1, -1)).to(device)
    state_offset = np.ones(state_dim)
    # state_offset =  np.array([-0.3, 0.0])
    state_offset = torch.FloatTensor(state_offset.reshape(1, -1)).to(device)
    # state_clip_low = np.array([-1.2, -0.07])
    # state_clip_high = np.array([0.6, 0.07])
    state_clip_low = state_bounds[:, 0]
    state_clip_high = state_bounds[:, -1]

    # subgoal bounds and offset
    # subgoal_bounds = np.zeros(state_dim)
    # subgoal_offset = np.zeros(state_dim)
    subgoal_bounds = np.zeros(goal_dim)
    subgoal_offset = np.zeros(goal_dim)
    for i in range(goal_dim):
        subgoal_bounds[i] = (valid_area_high[i] - valid_area_low[i])/2
        subgoal_offset[i] = valid_area_high[i] - subgoal_bounds[i]

    # for i in range(state_dim):
    #     subgoal_bounds[i] = (state_bounds[i][1] - state_bounds[i][0]) / 2
    #     subgoal_offset[i] = state_bounds[i][1] - subgoal_bounds[i]

    subgoal_bounds = torch.FloatTensor(subgoal_bounds.reshape(1, -1)).to(device)
    subgoal_offset = torch.FloatTensor(subgoal_offset.reshape(1, -1)).to(device)
    subgoal_clip_low = valid_area_low
    subgoal_clip_high = valid_area_high
    # subgoal_clip_low = state_clip_low
    # subgoal_clip_high = state_clip_high

    # exploration noise std for primitive action and subgoals
    # exploration_action_noise = np.array([0.1])
    # exploration_state_noise = np.array([0.02, 0.01])
    exploration_action_noise = np.ones(action_dim) * 0.1
    exploration_state_noise = np.ones(state_dim) * 0.1
    exploration_subgoal_noise = np.ones(goal_dim) * 0.1
    
    # goal_state = np.array([0.48, 0.04])        # final goal state to be achieved
    # threshold = np.array([0.01, 0.02])         # threshold value to check if goal state is achieved

    # TODO: meaningful threshold
    """
    observation threshold
    9  x 0.05
    2  x 0.01
    3  x np.pi/18
    11 x [-1,1]
    """
    pos_threshold = 0.05
    gripper_st_threshold = 0.01
    angle_threshold = np.pi/18
    vel_threshold = 1

    endgoal_thresholds = np.array([pos_threshold for _ in range(3)])
    subgoal_thresholds = np.concatenate((
        np.array([pos_threshold for _ in range(9)]),
        np.array([gripper_st_threshold for _ in range(2)]),
        np.array([angle_threshold for _ in range(3)]),
        np.array([vel_threshold for _ in range(11)]),
         ))

    # HAC parameters:
    k_level = 2                 # num of levels in hierarchy
    H = 30                      # time horizon to achieve subgoal
    lamda = 0.3                 # subgoal testing parameter
    
    # DDPG parameters:
    gamma = 0.95                # discount factor for future rewards
    n_iter = 100                # update policy n_iter times in one DDPG update
    batch_size = 100            # num of transitions sampled from replay buffer
    lr = 0.001
    
    # save trained models
    directory = "/home/robotics/projects/Hierarchical-Actor-Critic-HAC-PyTorch/preTrained/{}/{}level".format(env_name, k_level)
    filename = "HAC_{}".format(env_name)
    #########################################################
    
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    # creating HAC agent and setting parameters
    agent = HAC(k_level, H, state_dim, action_dim, render, endgoal_thresholds,
                action_bounds, action_offset, state_bounds, state_offset, lr,
                goal_dim, subgoal_bounds, subgoal_offset, subgoal_thresholds)

    agent.set_parameters(lamda, gamma, action_clip_low, action_clip_high,
                         state_clip_low, state_clip_high, exploration_action_noise, exploration_state_noise,
                         subgoal_clip_low, subgoal_clip_high, exploration_subgoal_noise)

    # logging file:
    log_f = open("log.txt","w+")
    
    # training procedure 
    for i_episode in range(1, max_episodes+1):
        agent.reward = 0
        agent.timestep = 0

        state = env.reset()
        goal_state = env.goal
        # collecting experience in environment
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXX RUN HAC with goal {}".format(goal_state))
        last_state, done = agent.run_HAC(env, k_level-1, state, goal_state, False)
        
        if agent.check_goal(last_state, goal_state, endgoal_thresholds):
            print("################ Solved! ################ ")
            name = filename + '_solved'
            agent.save(directory, name)
        
        # update all levels
        agent.update(n_iter, batch_size)
        
        # logging updates:
        log_f.write('{},{}\n'.format(i_episode, agent.reward))
        log_f.flush()
        
        if i_episode % save_episode == 0:
            agent.save(directory, filename)
        
        print("Episode: {}\t Reward: {}".format(i_episode, agent.reward))
        
    
if __name__ == '__main__':
    train()
 
