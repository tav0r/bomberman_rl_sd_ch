"""
Created on Sun Feb 17 22:22:43 2019
Fundamentals of Machine Learning | Winter Semester 2018/2019 | Bomberman Final Project
Solved by S.Deesamutara and C.Hansel
"""
import numpy as np

import os
import sys
import json

from settings import s as settings

# TODO get rid of this nasty stuff
GOT_KILLED = 14

# general settings
softmax = True
# settings for task 1
if settings.crate_density == 0:  # and len(agent.game_state['others']) == 0
    # kill = False
    num_actions = 5  # only moments # 8 for all actions

# settings for task 2
if settings.crate_density != 0:
    # kill = True
    num_actions = 6  # only moments # 8 for all actions

# settings for task 3
# if False:


def read_file(agent):
    """
    load learned q-table with the probabilities for the actions in different states
    """
    with open('q_table.json', 'r') as json_file:
        agent.q_table = json.load(json_file)


def write_file(agent):
    with open('q_table.json', 'w') as output:
        output.write(json.dumps(agent.q_table))


def state_to_str(state):
    """
    Turns a state representation of list to a string.
    The strings is used as keys for the q_table dictionary.
    :param state:
    :return: joint string of the objects describing the state
    """

    return ','.join(state)


def accessible_states(agent):
    """
    Scan direct surroundings of the agent for accessible sates. The options for the actions are in this function only
    the movements
    :param agent:
    :return: curr_state
    """
    try:
        options = [(-1, 0, 'left'), (0, -1, 'up'), (1, 0, 'right'), (0, 1, 'down'),
                   (0, 0, 'self')]  # compare to environment.py->perform_agent_action
        curr_state = list()
        agent_pos = np.array(agent.game_state['self'][0:2])

        for column, row, direction in options:
            # position of the field currently inspected
            scanning = np.array(agent_pos + np.array([column, row]))

            # check arena: walls and crates
            if agent.game_state['arena'][scanning[0], scanning[1]] == -1:
                curr_state.append('wall')
                continue

            elif agent.game_state['arena'][scanning[0], scanning[1]] == 1:
                curr_state.append('crate')
                continue

            # check opponent
            opponent = np.array(agent.game_state['others'])
            if len(opponent):
                opponent_pos = opponent[:, [0, 1]].flatten().astype(np.int)
                opponents_distance = np.linalg.norm(opponent_pos - scanning)
                if len(opponent[opponents_distance == 0]) > 0:
                    curr_state.append('opponent')
                    continue

            # check bombs
            bombs = np.array(agent.game_state['bombs'])
            if len(bombs):
                bombs_distance = np.linalg.norm(bombs[:, [0, 1]] - scanning, axis=1)
                if len(bombs[bombs_distance == 0]) > 0:
                    curr_state.append('bomb')
                    continue
                elif direction == 'self':
                    curr_state.append('empty')
                    continue
            elif direction == 'self':
                curr_state.append('empty')
                continue

            # check ongoing explosion
            if agent.game_state['explosions'][scanning[0], scanning[1]] > 1:
                curr_state.append('bomb')  # review why > 1
                continue

            # check ticking bomb
            bombs = np.array([[x, y] for x, y, t in agent.game_state['bombs'] if
                              (x == scanning[0] or y == scanning[1])])
            if len(bombs):
                bombs_distance = np.linalg.norm(bombs[:, [0, 1]] - scanning, axis=1)
                bombs = bombs[bombs_distance < 4]
                if len(bombs):
                    curr_state.append('danger') if np.min(bombs) > 2 else curr_state.append('bomb')
                    continue

            # check coins
            coins = np.array(agent.game_state['coins'])
            if len(coins):
                # distance from agent to coin
                coin_distance = np.linalg.norm(coins - scanning, axis=1)
                if len(coins[coin_distance == 0]) > 0:
                    curr_state.append('coin')
                    continue

            # otherwise the state is empty
            curr_state.append('empty')

            # TODO nothing useful to do, look for coins or enemy's close by

        # in case the q_table does not contain the state already, add an initial state of zeros
        string_state = state_to_str(curr_state)
        try:
            agent.q_table[string_state]
        except:
            agent.q_table[string_state] = list(np.zeros(num_actions))

        return curr_state

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb.frame.f_code.co_filename)[1]
        print('accessible_states : ', e, exc_type, fname, exc_tb.tb_lineno)



def setup(agent):
    """
    Called once before a settings of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    :param agent:
    :return:
    """

    try:
        agent.logger.debug('Successfully entered setup code')

        agent.num_games = 0  # for q_table update count in the console and for debugging
        agent.prev_state = None
        agent.curr_state = None
        agent.rewards_tmp = dict()  # individual rewards after each action, to write the whole settings of actions later in the q_table

        agent.relevant_actions = [0, 1, 2, 3, 4, 6, 7]  # discriminate bomb # TODO make if more consistent
        agent.rewards = {'bomb': -80,
                         'danger': -40,
                         'coin': 100,
                         'opponent': -40,
                         'wall': -80,
                         'empty': -20,
                         'dead': -100,
                         'crate': -70}
        agent.actions = ['LEFT', 'UP', 'RIGHT', 'DOWN', 'WAIT', 'BOMB']
        agent.train_flag = True  # training on or off

        # === epsilon-greedy ===
        # agent.gamma =  0.66
        # agent.alpha = 0.9 # 0.66
        agent.epsilon = 1.0
        agent.epsilon_min = 0.20
        agent.epsilon_decay = 0.9993

        # === SARSA ===
<<<<<<< HEAD
        agent.gamma = 0.9  # 0.66           # discount coefficient
        agent.alpha = 0.9  # 0.66           # learning rate
        agent.temperature = 200             # first guess of Boltzmann temp.
        agent.temperature_anneal = 0.99995  # anneal rate of Boltzmann temperature
        agent.lowest_temperature = 1        # lower bound to prevent the mathematical error
=======
        agent.gamma = 0.9 # 0.65           # discount coefficient
        agent.alpha = 0.9 # 0.65           # learning rate
        agent.temperature = 200              # first guess of Boltzmann temp.
        agent.temperature_anneal = 0.9      # anneal rate of Boltzmann temperature
        agent.lowest_temperature = 1         # lower bound to prevent the mathematical error

        # Fixed length FIFO queues to avoid repeating the same actions
        agent.coordinate_history = deque([], 20)
        # While this timer is positive, agent will not hunt/attack opponents
        agent.ignore_others_timer = 0

        # reduce the problem the surroundings of the agent
        # in case a tile of state is "empty".
        agent.radius = 0
        agent.radius_anneal = 2.5
        # weigths for diffrent kinds of obejcts
        agent.weights = {"arena": 1, "coins": 2, "others": 3} # TODO not used yet
>>>>>>> chris_train

        # Load the q_table:
        try:  # If the q_table.json file doesn't exit yet.
            read_file(agent)
        except Exception as ex:
            agent.q_table = dict()

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print('setup(): ', e, exc_type, fname, exc_tb.tb_lineno)


def act(agent):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via
    self.game_state, which is a dictionary. Consult 'get_state_for_agent'
    in environment.py to see what it contains.

    Set the action you wish to perform by assigning the relevant string to
    self.next_action. You can assign to this variable multiple times during
    your computations. If this method takes longer than the time limit
    specified in settings.py, execution is interrupted by the game and the
    current value of self.next_action will be used. The default value is
    'WAIT'.
    :param agent:
    :return:
    """
    agent.logger.info('Picking action according to rule settings')

    try:
        agent.curr_state = accessible_states(agent)
        state_string = state_to_str(agent.curr_state)
        q_values = np.array(agent.q_table[state_string][0:num_actions])

        if agent.train_flag:
            if softmax:
                # === softmax policy/ Max-Boltzmann Exploration Rule ===
                policies = np.exp(-q_values / agent.temperature)  # Boltzmann distribution
                prob = policies / sum(policies)
                rand_index = int(np.random.choice(num_actions, 1, replace=False, p=prob))
                action = rand_index

                # review not sure about the annealing here or only in end_of_episode
                # if agent.temperature >= agent.lowest_temperature:
                #     agent.temperature *= agent.temperature_anneal

            if not softmax:
                # === epsilon greedy === # seems to be working
                if np.random.uniform(0, 1) > agent.epsilon:
                    state_string = state_to_str(agent.curr_state)
                    q_values = np.array(agent.q_table[state_string][0:num_actions])
                    action = np.random.choice(np.flatnonzero(q_values == q_values.max()))

                else:
                    action = np.random.randint(0, num_actions)
                # anneal epsilon
                if agent.epsilon >= agent.epsilon_min:
                    agent.epsilon = agent.epsilon * agent.epsilon_decay

        else:
            try:
                state_string = state_to_str(agent.curr_state)
                curr_q = np.array(agent.q_table[state_string])
                argmax = np.where(curr_q == max(curr_q))[0]
                action = np.random.choice(argmax)
            except KeyError as ke:
                action = np.random.randint(0, num_actions)
                print('KeyError in act()')

        agent.next_action = agent.actions[action]

    except Exception as ex:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print('act: ', ex, exc_type, fname, exc_tb.tb_lineno)
        pass


def reward_update(agent):
    """Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occured during the previous step.
    Consult settings.py to see what events are tracked. You can hand out
    rewards to your agent based on these events and your knowledge of the
    (new) game state. In contrast to act, this method has no time limit.
    """
    try:
        agent.logger.debug(f'Encountered {len(agent.events)} game event(s)')

        if len(agent.events) == 0:
            return

        if agent.game_state['step'] == 2:  # due to if self.fake_self.game_state['step'] > 1:... in agents.py
            # review check how the 1 step is generally defined
            agent.train_flag = True
            agent.rewards_tmp = dict()
            agent.prev_action = np.random.randint(0, num_actions)
            agent.prev_state = accessible_states(agent)[0:num_actions]
            agent.prev_reward = 0  # np.random.random_sample() * max(agent.rewards.values()) # review not sure about that

        curr_pos = [agent.game_state['self'][0], agent.game_state['self'][1]]
        state_string = state_to_str(agent.curr_state)
        Rt = 0 # review
        action = 0 # review
        for event in agent.events:
            Rt = 0

            # exit for waiting only
            if event == 5:
                continue

            # next action for the reward
            action = agent.actions.index(agent.next_action)

            # movements only
            if action < 4:
                curr_action = agent.curr_state[action]  # [0:num_actions]
                Rt += agent.rewards[curr_action]

            # drop bomb
            elif action == 4:
                if ('crate' or 'enemy') in agent.curr_state:
                    Rt += 10
                else:
                    Rt += -20

            # Wait
            else:
                if agent.curr_state[4] == 'bomb':
                    Rt += agent.rewards['bomb']
                elif (agent.game_state['explosions'][curr_pos[0], curr_pos[1]]) > 0:
                    Rt += agent.rewards['danger']
                else:
                    Rt += -10

        # === SARSA algorithm ===
        try:
            prev_state_str = state_to_str(agent.prev_state)
            a, g = agent.alpha, agent.gamma
            try:
                Qsa = agent.q_table[prev_state_str][agent.prev_action]
            except KeyError as ke1:
                key = agent.prev_state[agent.prev_action]
                Qsa = agent.rewards[key]

            # review make 'dead' important in the rewards
            if GOT_KILLED == agent.events[-1]:
                Qs1a1 = agent.rewards['dead']
            else:
                try:
                    Qs1a1 = max(agent.q_table[state_string])
                except KeyError as ke2:
                    key = agent.prev_state[agent.prev_action]
                    Qs1a1 = agent.rewards[key]

            curr_reward = Qsa + a * (agent.prev_reward + g * Qs1a1 - Qsa)

            # add reward to a temporal dict of states and rewards for actions
            # this will be averaged and written into the q_table in end_of_episode after a whole episode/game

            try:
                try:
                    # append current reward to dict when the combination of state and action occurred before
                    agent.rewards_tmp[state_string][action].append(curr_reward)
                except Exception as ex:
                    # overwrite
                    agent.rewards_tmp[state_string][action] = [curr_reward]
            except Exception as ex:
                # otherwise new entry in dict
                agent.rewards_tmp[state_string] = {action: [curr_reward]}

        except Exception as ex:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print('Reward_update <--: ', ex, exc_type, fname, exc_tb.tb_lineno)
            pass

        agent.prev_state = agent.curr_state
        agent.prev_action = action
        agent.prev_reward = Rt

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print('Reward_update -->: ', e, exc_type, fname, exc_tb.tb_lineno)


def end_of_episode(agent):
    """
    Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    :param agent:
    :return:
    """

    # reduce temperature # review whether that needs to be done in each step or only each episode
    if agent.temperature >= agent.lowest_temperature:
        agent.temperature *= agent.temperature_anneal

    for state in agent.rewards_tmp:  # review rewards_tmp should have given a dict of num_actions x num_actions not num_actions x 1
        for action in agent.rewards_tmp[state]:
            if agent.q_table[state][action] != 0:
                agent.rewards_tmp[state][action] += [agent.q_table[state][action]]
            agent.q_table[state][action] = np.mean(agent.rewards_tmp[state][action])

    write_file(agent)
    print('q_table updated in game number ' + str(agent.num_games))

    agent.num_games += 1
