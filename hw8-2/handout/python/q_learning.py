from environment import MountainCar
import csv
import numpy as np
import math
import sys
import copy
import random
""" ---------------------------------------------------------------------------
                            Initialization Routines
--------------------------------------------------------------------------- """

def create_state(state, state_space):
    numpy_state = np.zeros(state_space)

    for key in state:
        numpy_state[key] = state[key]

    return numpy_state
        
""" ---------------------------------------------------------------------------
                            Calculation Routines
--------------------------------------------------------------------------- """

def calculate_q(state, action, b):
    return np.matmul(state, action) + b

def find_old_q(epsilon, action_space, cur_state, b, weights):
    p = random.random()

    if p < epsilon:
        action = random.randint(0,action_space-1) 
        q = calculate_q(cur_state, weights[action], b)

    else:
        action = 0
        q = 0

        #reverse ensure min index is used
        for i in range(0, action_space):
            cur_action = i
            cur_q = calculate_q(cur_state, weights[cur_action], b)

            if (cur_q > q or i == 0):
                q = cur_q
                action = cur_action

    return q, action

def find_new_q(state, action_space, b, weights):
    q = 0

    for i in range(0, action_space):
        cur_q = calculate_q(state, weights[i], b)

        if (cur_q > q or i == 0):
            q = cur_q

    return q

def find_q_grad(state, action, action_space, state_space):

    grad = np.zeros((action_space, state_space))
    i = action
    grad[i] = state

    return grad

def calculate_grad(alpha, gamma, reward, old_q, new_q, q_grad):

    td_target = reward + gamma*new_q
    td_error = old_q - td_target
    b_grad = alpha*td_error
    grad = q_grad * b_grad
    return b_grad, grad

""" ---------------------------------------------------------------------------
                            Print Routines
--------------------------------------------------------------------------- """
def write_returns(outPath, returns):
    f = open(outPath, "w")

    for value in returns:
        f.write("%3.1f\n" % (value))
    f.close()

def write_weights(outPath, weights, b):
    f = open(outPath, "w")

    weights = np.transpose(weights)

    f.write("%f\n" % (b))

    for row in weights:
        if type(row) == np.float64:
            f.write("%f\n" % (row))
        else:
            for val in row:
                f.write("%f\n" % (val))
    f.close()

""" ---------------------------------------------------------------------------
                            Main Run Routines
--------------------------------------------------------------------------- """

def initialize(data, argv):
    data.mode = argv[1]
    data.weights_outpath = argv[2]
    data.returns_outpath = argv[3]
    data.episodes = int(argv[4])
    data.max_iterations = int(argv[5])
    data.epsilon = float(argv[6])
    data.gamma = float(argv[7])
    data.alpha = float(argv[8])

    data.car = MountainCar(data.mode)
    
    data.a_space = data.car.action_space
    data.s_space = data.car.state_space
    data.weights = np.zeros((data.a_space, data.s_space))
    data.b = 0

    data.returns = []

def run_q_function_approx_alg(data):
    
    for e in range(0, data.episodes):
        cur_return = 0

        for i in range(0, data.max_iterations):
            if (i == 0):
                cur_state = create_state(data.car.reset(), data.s_space)
                data.car.reset()

            old_q, action = find_old_q(data.epsilon, data.a_space, cur_state, data.b, data.weights)
            
            new_state_dict, reward, terminal = data.car.step(action)
            new_state = create_state(new_state_dict, data.s_space)

            new_q = find_new_q(new_state, data.a_space, data.b, data.weights)
            q_grad = find_q_grad(cur_state, action, data.a_space, data.s_space)

            b_grad, weights_grad = calculate_grad(data.alpha, data.gamma, reward, old_q, new_q, q_grad)

            data.b -= b_grad
            data.weights -= weights_grad
            cur_return += reward
            cur_state = new_state

            if (terminal):
                break
        
        if (e % 25 == 0):
             print(np.average(data.returns))
        data.returns.append(cur_return)

def print_output(data):
    write_returns(data.returns_outpath, data.returns)
    write_weights(data.weights_outpath, data.weights, data.b)
def main(argv):

    class Struct(object): pass
    data = Struct()

    initialize(data, argv)
    run_q_function_approx_alg(data)
    print_output(data)

if __name__ == "__main__":
    main(sys.argv)