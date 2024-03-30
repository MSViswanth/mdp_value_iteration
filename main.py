import copy

class MDPObject:
    def __init__(self, m, n, r):
        self.states = []
        for i in range(0, m):
            for j in range(0, n):
                self.states.append((i, j))
        self.actions = ["l", "u", "r", "d"]
        self.gamma = 0.99
        self.rewards = [[r, -1, 10], [-1, -1, -1], [-1, -1, -1]]


def q_value(mdp, s, a, U):
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
        s_ = move(mdp, s, action)
        q_value += p * (r[s_[0]][s_[1]] + mdp.gamma * U[s_])
    return q_value


def move(mdp, s, a):
    match a:
        case "l":
            s_ = (s[0], s[1] - 1)
        case "r":
            s_ = (s[0], s[1] + 1)
        case "d":
            s_ = (s[0] + 1, s[1])
        case "u":
            s_ = (s[0] - 1, s[1])
    if s_ not in mdp.states:
        s_ = s
    return s_


def value_iteration(mdp, e):
    S = mdp.states
    A = mdp.actions
    U = {}
    U_ = {s: 0 for s in S}
    gamma = mdp.gamma

    while True:
        U = copy.deepcopy(U_)
        delta = 0
        for s in S:
            U_[s] = max([q_value(mdp, s, a, U) for a in A])
            if abs(U_[s] - U[s]) > delta:
                delta = abs(U_[s] - U[s])
        if delta <= e * ((1 - gamma) / gamma):
            break
    return U

def generate_policy(mdp, U, g):
        policy = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for u in U:
            reward = {a: U[move(mdp, u, a)] - U[u] for a in mdp.actions}
            policy[u[0]][u[1]] = max(reward, key=reward.get)
        policy[g[0]][g[1]] = "G"
        return policy

def print_matrix(matrix):
        n = len(matrix)
        for i in range(0, n):
            for j in range(0, n):
                print('{:>2s}'.format(str(matrix[i][j])), end=' ')
            print('')

rs = [-100, -3, 0, 3]
for r in rs:
    mdp = MDPObject(3, 3, r)
    u = value_iteration(mdp, 0.001)

    print(f'Rewards for when r = {r}')
    print_matrix(mdp.rewards)
    print('Policy')
    print_matrix(generate_policy(mdp, u, (0, 2)))