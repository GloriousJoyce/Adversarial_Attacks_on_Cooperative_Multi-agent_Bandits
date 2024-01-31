import numpy as np
from matplotlib import pyplot as plt
import random
import copy
from scipy import stats
from math import pi


# Best arm
def best(m):
    Li = np.argsort(m)[-1:]
    Li = np.flipud(Li)
    return Li


# Attack
def Attack():
    index = K - 1
    beta = np.sqrt(2 * sigma*sigma * np.log(pi * pi * K * n[index] * n[index] / (3 * delta)) / n[index])
    ret = mu[At] * n[At] + sum(cache_m0) + sample - (n[At] + M) * (mu[index] - 2 * beta - Delta0)
    ret = max(ret, 0)
    return ret


def Attack_all():
    index = K - 1
    beta = np.sqrt(0.5 * np.log(pi * pi * K * n[index] * n[index] / (3 * delta)) / n[index])
    ret = mu[At] * n[At] + sum(cache_m0) - (n[At] + M) * (mu[index] - 2 * beta - Delta0)
    ret = max(ret, 0)
    return ret


Repeat = 1
T = 100000
K = 20
M = 20
Delta0 = 0.1
delta = 0.1
alpha = 4
sigma = 0.1


Mu = [0] * K
index = 1
for i in range(K):
    seed = random.uniform(index, index + 0.1)
    Mu[i] = seed
    index += 0.2
Mu = sorted(Mu, reverse=True)
Mu = np.array(Mu)
print("Mu = " + str(Mu))

MINDELTA = 10
for i in range(K - 1):
    if Mu[i]-Mu[i+1] <= MINDELTA:
        MINDELTA = Mu[i]-Mu[i+1]
print("MinDelta = " + str(MINDELTA))
print()


Regret = np.zeros((Repeat, T))
Cost = np.zeros((Repeat, T))

for counter in range(Repeat):
    print("The " + str(counter) + "'s repeat")

    mu = np.array([Mu[0] + 3*sigma] * K)
    m0 = np.array([Mu[0] + 3*sigma] * K)
    n = np.ones(K)
    r = np.zeros(K)
    UCB = np.zeros(K)

    for t in range(T):

        r = np.sqrt(alpha * np.log(max(t, 1)) / (2 * n))
        UCB = mu + r
        cache_mu = []
        cache_m0 = []

        At = best(UCB)

        for agent in range(M):

            sample = stats.norm.rvs(Mu[At], sigma)
            if agent == M - 1 and At != K - 1:
                attack = Attack()
                Cost[counter][t] += attack
                cache_m0.append(sample)
                cache_mu.append(sample - attack)
            else:
                cache_m0.append(sample)
                cache_mu.append(sample)

        mu[At] = (mu[At] * n[At] + sum(cache_mu)) / (n[At] + M)
        m0[At] = (m0[At] * n[At] + sum(cache_m0)) / (n[At] + M)
        n[At] += M

        for arm in range(K):
            Regret[counter][t] = (Mu[0] * M * (t+1) - sum(Mu * n)) / (t+1)
        if t > 0:
            Cost[counter][t] += Cost[counter][t-1]

Regret = np.average(Regret, axis=0)
for i in range(Repeat):
    for j in range(T):
        Cost[i][j] /= (j+1)
Cost = np.average(Cost, axis=0)


AllRegret = np.zeros((Repeat, T))
AllCost = np.zeros((Repeat, T))

for counter in range(Repeat):
    print("The " + str(counter) + "'s repeat")

    mu = np.array([Mu[0] + 3*sigma] * K)
    m0 = np.array([Mu[0] + 3*sigma] * K)
    n = np.ones(K)
    r = np.zeros(K)
    UCB = np.zeros(K)

    for t in range(T):

        r = np.sqrt(alpha * np.log(max(t, 1)) / (2 * n))
        UCB = mu + r
        cache_mu = []
        cache_m0 = []

        At = best(UCB)

        for agent in range(M):

            sample = stats.norm.rvs(Mu[At], sigma)
            if At != K - 1:
                attack = Attack_all()
                AllCost[counter][t] += attack
                cache_m0.append(sample)
                cache_mu.append(sample - attack)
            else:
                cache_m0.append(sample)
                cache_mu.append(sample)

        mu[At] = (mu[At] * n[At] + sum(cache_mu)) / (n[At] + M)
        m0[At] = (m0[At] * n[At] + sum(cache_m0)) / (n[At] + M)
        n[At] += M

        for arm in range(K):
            AllRegret[counter][t] = (Mu[0] * M * (t+1) - sum(Mu * n)) / (t+1)
        if t > 0:
            AllCost[counter][t] += AllCost[counter][t-1]

AllRegret = np.average(AllRegret, axis=0)
for i in range(Repeat):
    for j in range(T):
        AllCost[i][j] /= (j+1)
AllCost = np.average(AllCost, axis=0)


NORegret = np.zeros((Repeat, T))

for counter in range(Repeat):
    print("The " + str(counter) + "'s repeat")

    mu = np.array([Mu[0] + 3*sigma] * K)
    n = np.ones(K)
    r = np.zeros(K)
    UCB = np.zeros(K)

    for t in range(T):

        r = np.sqrt(alpha * np.log(max(t, 1)) / (2 * n))
        UCB = mu + r
        cache_mu = []
        cache_m0 = []

        At = best(UCB)

        for agent in range(M):

            sample = stats.norm.rvs(Mu[At], sigma)
            cache_m0.append(sample)
            cache_mu.append(sample)

        mu[At] = (mu[At] * n[At] + sum(cache_mu)) / (n[At] + M)
        n[At] += M

        for arm in range(K):
            NORegret[counter][t] = (Mu[0] * M * (t+1) - sum(Mu * n)) / (t+1)

NORegret = np.average(NORegret, axis=0)


plt.figure()
plt.grid(True)
x = np.linspace(0, T, T)
plt.plot(x, Regret, label="Attack one agent", color="red", linewidth="2")
plt.plot(x, AllRegret, label="Attack all agents", color="blue", linewidth="2", linestyle=":")
plt.plot(x, NORegret, label="No attack", color="green", linewidth="2", linestyle="--")
plt.xlabel("Round", fontsize=20)
plt.ylabel("Average Regret", fontsize=20)
plt.legend(fontsize=12)
plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("regret.png", dpi=600, bbox_inches='tight')
plt.show()


plt.figure()
plt.grid(True)
x = np.linspace(0, T, T)
plt.plot(x, Cost, label="Attack one agent", color="red", linewidth="2")
plt.plot(x, AllCost, label="Attack all agents", color="blue", linewidth="2", linestyle=":")
plt.xlabel("Round", fontsize=20)
plt.ylabel("Average Cost", fontsize=20)
plt.legend(fontsize=12)
plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("cost.png", dpi=600, bbox_inches='tight')
plt.show()
