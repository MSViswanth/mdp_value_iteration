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


def value_iteration(mdp, e, g):
    S = mdp.states
    A = mdp.actions
    U = {}
    U_ = {s: 0 for s in S}
    gamma = mdp.gamma
    policies = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    while True:
        U = copy.deepcopy(U_)
        delta = 0
        for s in S:
            if s == g:
                U_[s] = U[s]
            else:
                q_values = {a:q_value(mdp, s, a, U) for a in A}
                policy = max(q_values, key=q_values.get)
                U_[s] = q_values[policy]
            policies[s[0]][s[1]] = policy

            if abs(U_[s] - U[s]) > delta:
                delta = abs(U_[s] - U[s])
        if delta <= e * ((1 - gamma) / gamma):
            break
    U[g] = 10
    return U, policies

def print_stuff(*args):
    title = ['Reward Matrix', 'Utility Matrix','Policy Matrix']
    n = len(args[0])
    if (len(args) == 3):
        print(
            f'{{:^{n*6 + n}s}} {{:^{n*3}s}} {{:^{n*6 + n}s}} {{:^{n*3}s}} {{:^{n*6 + n}s}}'.format(title[0], '|', title[1],'|', title[2]))
        print(f'{'-' * 30 * n}')
        for i in range(0, n):
            for k, arg in enumerate(args):
                for j in range(0, n):
                    print('{:>6s}'.format(str(arg[i][j])), end=' ')
                print(f'{{:<{n*3}s}}'.format(''), end=' ')
            print('')
rs = [-100, -3, 0, 3]
for r in rs:
    mdp = MDPObject(3, 3, r)
    u, policy = value_iteration(mdp, 0.0001, (0,2))
    U = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for u_ in u:
        U[u_[0]][u_[1]] = round(u[u_], 2)
    print(f'{'-' * 30 * 3}')
    print_stuff(mdp.rewards,U, policy)