import torch
import numpy as np
from DDPG import DDPG
from utils import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HAC:
    def __init__(self, k_level, H, state_dim, action_dim, render, endgoal_thresholds,
                 action_bounds, action_offset, state_bounds, state_offset, lr,
                 goal_dim, subgoal_bounds, subgoal_offset, subgoal_thresholds):

        # adding lowest level
        print("LOWEST LEVEL")
        self.HAC = [DDPG(state_dim, state_dim, action_dim, action_bounds, action_offset, lr, H)]
        print("FINISH BUILDING HAC for lowest level")
        self.replay_buffer = [ReplayBuffer()]

        # adding remaining levels
        for i in range(k_level - 1):
            print("building HAC for level-{}".format(i + 1))
            # self.HAC.append(DDPG(state_dim, action_dim, state_bounds, state_offset, lr, H))
            self.HAC.append(DDPG(state_dim, goal_dim, state_dim, subgoal_bounds, subgoal_offset, lr, H))
            print("finished HAC for level-{}".format(i + 1))
            self.replay_buffer.append(ReplayBuffer())

        # set some parameters
        self.k_level = k_level
        self.H = H
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.endgoal_thresholds = endgoal_thresholds
        self.subgoal_thresholds = subgoal_thresholds
        self.render = render

        # logging parameters
        self.goals = [None] * self.k_level
        self.reward = 0
        self.timestep = 0

    def set_parameters(self, lamda, gamma, action_clip_low, action_clip_high,
                       state_clip_low, state_clip_high, exploration_action_noise, exploration_state_noise,
                       subgoal_clip_low, subgoal_clip_high, exploration_subgoal_noise):

        self.lamda = lamda
        self.gamma = gamma
        self.action_clip_low = action_clip_low
        self.action_clip_high = action_clip_high
        self.state_clip_low = state_clip_low
        self.state_clip_high = state_clip_high
        self.exploration_action_noise = exploration_action_noise
        self.exploration_subgoal_noise = exploration_state_noise
        self.subgoal_clip_low = subgoal_clip_low
        self.subgoal_clip_high = subgoal_clip_high
        # self.exploration_subgoal_noise = exploration_subgoal_noise

    """
    """

    def check_goal(self, state, goal, threshold):
        # print("state shape {}".format(state.shape))
        # print("action shape {}".format(self.action_dim))
        # print("goal shape {}".format(goal.shape))
        # print("threshold shape {}".format(threshold.shape))
        # state here is only checking the achieved goal, in PickAndPlace is indexed [3:6]
        obj_state = state[3:6]
        for i in range(self.goal_dim):
            if abs(obj_state[i] - goal[i]) > threshold[i]:
                return False
        return True

    """
    for level>0
    """

    def check_goal_up(self, state, goal, threshold):
        # state here is only checking the subgoal and goal
        # print("====================   LEVEL-{}  ===========================".format(self.k_level-1))
        # print("UPPER LVL state shape {}".format(state.shape))
        # print("UPPER LVL action shape {}".format(self.action_dim))
        # print("UPPER LVL goal shape {}".format(goal.shape))
        # print("UPPER LVL threshold shape {}".format(threshold.shape))
        for i in range(self.goal_dim):
            if abs(state[i] - goal[i]) > threshold[i]:
                return False
        return True

    """
    """

    def run_HAC(self, env, i_level, state, goal, subgoal_test):
        next_state = None
        done = None
        goal_transitions = []

        # logging updates
        self.goals[i_level] = goal
        print("<<<<<<<<<<<<<<<<<<<<<<<<<< GOALS in run_HAC {}".format(self.goals))

        # H attempts
        for _ in range(self.H):
            next_subgoal_test = subgoal_test

            action = self.HAC[i_level].select_action(state, goal)
            # print("ACTION in run HAC for level-{} >>>>> {}".format(i_level, action))
            #
            #   <================ high level policy ================>
            if i_level > 0:
                # print("action from nn for level-1 >>>>> {}".format(action))
                # action should be subgoal
                # add noise or take random action if not subgoal testing
                if not subgoal_test:
                    if np.random.random_sample() > 0.2:
                        action = action + np.random.normal(0, self.exploration_subgoal_noise)
                        action = action.clip(self.subgoal_clip_low, self.subgoal_clip_high)
                    else:
                        action = np.random.uniform(self.subgoal_clip_low, self.subgoal_clip_high)

                # Determine whether to test subgoal (action)
                if np.random.random_sample() < self.lamda:
                    next_subgoal_test = True

                # Pass subgoal to lower level
                # subgoal is action
                next_state, done = self.run_HAC(env, i_level - 1, state, action, next_subgoal_test)

                # only take achieved goal
                achieved_goal = next_state

                # if subgoal was tested but not achieved, add subgoal testing transition
                if next_subgoal_test and not self.check_goal_up(action, achieved_goal, self.endgoal_thresholds):
                    # print("ACTION inserted to replay buffer @ 129 {} at level-{}".format(action.shape, i_level))
                    self.replay_buffer[i_level].add((state, action, -self.H, next_state, goal, 0.0, float(done)))

                # for hindsight action transition
                # TODO:
                action = achieved_goal
                # print("HINDSIGHT ACTION action shape {}".format(action.shape))

            #   <================ low level policy ================>
            else:
                # add noise or take random action if not subgoal testing
                if not subgoal_test:
                    if np.random.random_sample() > 0.2:
                        action = action + np.random.normal(0, self.exploration_action_noise)
                        action = action.clip(self.action_clip_low, self.action_clip_high)
                    else:
                        print("======================== SUBGOAL TESTING ============================")
                        action = np.random.uniform(self.action_clip_low, self.action_clip_high)

                # take primitive action
                next_state, rew, done, _ = env.step(action)
                # print("=============================================")
                # print("Observation {}".format(np.around(next_state, decimals=3)))

                if self.render:
                    env.render()

                    if self.k_level == 2:
                        env.render_goal(self.goals[0], self.goals[1])
                    elif self.k_level == 3:
                        env.render_goal_2(self.goals[0], self.goals[1], self.goals[2])

                    for _ in range(1000000):
                        continue

                # this is for logging
                self.reward += rew
                self.timestep += 1

            # check if goal is achieved
            goal_achieved = self.check_goal(next_state, goal, self.endgoal_thresholds)

            # hindsight action transition
            if goal_achieved:
                # print("ACTION inserted to replay buffer @ 172 {} at level-{}".format(action.shape, i_level))
                self.replay_buffer[i_level].add((state, action, 0.0, next_state, goal, 0.0, float(done)))
            else:
                # print("ACTION inserted to replay buffer @ 175 {} at level-{}".format(action.shape, i_level))
                self.replay_buffer[i_level].add((state, action, -1.0, next_state, goal, self.gamma, float(done)))

            # copy for goal transition
            goal_transitions.append([state, action, -1.0, next_state, None, self.gamma, float(done)])

            state = next_state

            if done or goal_achieved:
                break

        # hindsight goal transition
        # last transition reward and discount is 0
        goal_transitions[-1][2] = 0.0
        goal_transitions[-1][5] = 0.0
        for transition in goal_transitions:
            # last state is goal for all transitions
            if i_level > 0:
                transition[4] = next_state[3:6]
            else:
                transition[4] = next_state
            # print("ACTION inserted to replay buffer @ 196 {} at level-{}".format(transition[1].shape, i_level))
            self.replay_buffer[i_level].add(tuple(transition))

        return next_state, done

    """
    """

    def update(self, n_iter, batch_size):
        for i in range(self.k_level):
            self.HAC[i].update(self.replay_buffer[i], n_iter, batch_size)

    """
    """

    def save(self, directory, name):
        for i in range(self.k_level):
            self.HAC[i].save(directory, name + '_level_{}'.format(i))

    """
    """

    def load(self, directory, name):
        for i in range(self.k_level):
            self.HAC[i].load(directory, name + '_solved_level_{}'.format(i))
