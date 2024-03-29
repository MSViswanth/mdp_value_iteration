import copy
import math


rs = [-100, -3, 0, 3]
r = -100
action_space = ["l", "u", "r", "d"]
discount_factor = 0.99


grid_world_rewards = [[r, -1, 10], [-1, -1, -1], [-1, -1, -1]]
# class MDPObject:


def q_value(mdp, s, a, U):
    possible_actions = [
        mdp.actions[mdp.actions.index(a) % 4 + 1],
        mdp.actions[mdp.actions.index(a) % 4 - 1],
        a
    ]
    q_value = 0
    for action in possible_actions:
        if action == a:
            p = 0.8
        else:
            p = 0.1
        match action:
            case 'l':
                s_ = (s[0]-1, s[1])
            case 'r':
                s_ = (s[0]+1, s[1])
            case 'u':
                s_ = (s[0], s[1]+1)
            case 'd':
                s_ = (s[0], s[1]-1)
        q_value += p * (r[*s_] + mdp.gamma * U[*s_])


def value_iteration(mdp, e):
    S = mdp.states
    A = mdp.actions
    U = {}
    U.keys = S
    U = {s: 0 for s in S}
    U_ = copy.deepcopy(U)
    gamma = mdp.gamma

    while True:
        U = U_
        delta = 0
        for s in S:
            U_[s] = max([q_value(mdp, s, a, U) for a in A])
            if math.abs(U_[s] - U[s]) > delta:
                delta = math.abs(U_[s] - U[s])
        if delta <= e * ((1 - gamma) / gamma):
            break
    return U
