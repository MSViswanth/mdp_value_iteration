import math

rs = [-100, -3, 0, 3]
r = -100
grid_world = [[r, -1, 10], [-1, -1, -1], [-1, -1, -1]]
action_space = ["l", "r", "u", "d"]
discount_factor = 0.99

def value_iteration(mdp, e):
    U = [] 
    U_ = []
    S = mdp.states
    A = mdp.actions
    delta = 0
    gamma = mdp.gamma


    while(True):
        U = U_
        delta = 0
        for s in S:
            U_[s] = max([q_value(mdp, s, a, U) for a in A])
            if math.abs(U_[s] - U[s]) > delta:
                delta = math.abs(U_[s] - U[s])
        if delta <= e*((1- gamma)/gamma):
            break
    return U

def q_value():
    return 0