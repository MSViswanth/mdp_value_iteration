import copy


rs = [-100, -3, 0, 3]
r = -100
discount_factor = 0.99


class MDPObject:
    def __init__(self, m, n):
        self.states = []
        self.rewards = []
        for i in range(0, m):
            self.rewards.append([])
            for j in range(0, n):
                self.states.append((i, j))
                self.rewards[i].append(-0.04)
        self.actions = ["l", "u", "r", "d"]
        self.states.remove((1, 1))
        self.rewards[3][2] = +1
        self.rewards[3][1] = -1
        # self.gamma = 0.99
        self.gamma = 1
        # self.rewards = [[r, -1, 10], [-1, -1, -1], [-1, -1, -1]]


def q_value(mdp, s, a, U):
    # print(mdp.actions.index(a) % 4 + 1)
    possible_actions = [
        mdp.actions[(mdp.actions.index(a) + 1) % 4],
        mdp.actions[(mdp.actions.index(a) - 1) % 4],
        a,
    ]
    r = mdp.rewards
    q_value = 0
    for action in possible_actions:
        if action == a:
            p = 0.8
        else:
            p = 0.1
        match action:
            case "l":
                s_ = (s[0] - 1, s[1])
            case "r":
                s_ = (s[0] + 1, s[1])
            case "u":
                s_ = (s[0], s[1] + 1)
            case "d":
                s_ = (s[0], s[1] - 1)
        if s_ not in mdp.states:
            s_ = s
        q_value += p * (r[s_[0]][s_[1]] + mdp.gamma * U[s_])
    return q_value


def value_iteration(mdp, e):
    S = mdp.states
    A = mdp.actions
    U = {}
    U = {s: 0 for s in S}
    U_ = copy.deepcopy(U)
    gamma = mdp.gamma

    while True:
        # print('in while')
        U = copy.deepcopy(U_)
        delta = 0
        for s in S:
            U_[s] = max([q_value(mdp, s, a, U) for a in A])
            # print(U_[s])
            if (U_[s] - U[s]) > delta:
                delta = (U_[s] - U[s])
        print(delta)
        if delta <= e * ((1 - gamma) / gamma):
            break
    return U


mdp = MDPObject(4, 3)
u = value_iteration(mdp, 0.0000000001)
print(u)
