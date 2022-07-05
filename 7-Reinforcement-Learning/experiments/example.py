import gym
from src import MultiArmedBandit, QLearning
import matplotlib.pyplot as plt
import numpy as np

def problem_2a():
    env_name = 'SlotMachines-v0'
    avg1 = []
    avg5 = []
    avg10 = []
    for i in range(10):
        env = gym.make(env_name)
        agent = MultiArmedBandit()
        action_values, rewards = agent.fit(env, steps=100000)

        if i < 1:
            avg1.append(rewards)

        if i < 5:
            avg5.append(rewards)
        
        if i < 10:
            avg10.append(rewards)

    avg1 = np.array(avg1)
    print(avg1.shape)
    avg5 = np.array(avg5)
    print(avg5.shape)
    avg10 = np.array(avg10)
    print(avg10.shape)
    avg1 = np.mean(avg1, axis=0)
    avg5 = np.mean(avg5, axis=0)
    avg10 = np.mean(avg10, axis=0)

    step = np.linspace(0,99,100)
    plt.plot(step, avg1, label='1 trial')
    plt.plot(step, avg5, label='5 trials')
    plt.plot(step, avg10, label='10 trials')

    plt.xlabel('average step')
    plt.ylabel('average reward')
    plt.legend(loc='upper right')

    plt.savefig('2a')


def problem_2b():
    env_name = 'SlotMachines-v0'
    avgB = []
    avgQ = []
    for i in range(10):
        env = gym.make(env_name)
        agent = MultiArmedBandit()
        action_values, rewards = agent.fit(env, steps=100000)
        avgB.append(rewards)

    for i in range(10):
        env = gym.make(env_name)
        agent = QLearning()
        action_values, rewards = agent.fit(env, steps=100000)
        avgQ.append(rewards)

    
    avgB = np.array(avgB)
    avgQ = np.array(avgQ)

    avgB = np.mean(avgB, axis=0)
    avgQ = np.mean(avgQ, axis=0)
    step = np.linspace(0,99,100)
    plt.plot(step, avgB, label='MultiArmedBandit')
    plt.plot(step, avgQ, label='QLearning')

    plt.xlabel('average step')
    plt.ylabel('average reward')
    plt.legend(loc='lower right')

    plt.savefig('2b')

        

def problem_3a():
    env_name = 'FrozenLake-v0'
    avg001 = []
    avg05 = []
    for i in range(10):
        env = gym.make(env_name)
        agent = QLearning(epsilon=0.01)
        action_values, rewards = agent.fit(env, steps=100000)
        avg001.append(rewards)

    for i in range(10):
        env = gym.make(env_name)
        agent = QLearning(epsilon=0.5)
        action_values, rewards = agent.fit(env, steps=100000)
        avg05.append(rewards)

    avg001 = np.array(avg001)
    avg05 = np.array(avg05)

    avg001 = np.mean(avg001, axis=0)
    avg05 = np.mean(avg05, axis=0)
    step = np.linspace(0,99,100)
    plt.plot(step, avg001, label='epsilon=0.01')
    plt.plot(step, avg05, label='epsilon=0.5')

    plt.xlabel('average step')
    plt.ylabel('average reward')
    plt.legend(loc='upper right')

    plt.savefig('3a')


    
def problem_3d():
    env_name = 'FrozenLake-v0'
    avg001 = []
    avg05 = []
    for i in range(10):
        env = gym.make(env_name)
        agent = QLearning(epsilon=0.2)
        action_values, rewards = agent.fit(env, steps=100000)
        avg001.append(rewards)

    for i in range(10):
        env = gym.make(env_name)
        agent = QLearning(epsilon=0.2, adaptive=True)
        action_values, rewards = agent.fit(env, steps=100000)
        avg05.append(rewards)

    avg001 = np.array(avg001)
    avg05 = np.array(avg05)

    avg001 = np.mean(avg001, axis=0)
    avg05 = np.mean(avg05, axis=0)
    step = np.linspace(0,99,100)
    plt.plot(step, avg001, label='0.2 not adaptive')
    plt.plot(step, avg05, label='0.2 adaptive')

    plt.xlabel('average step')
    plt.ylabel('average reward')
    plt.legend(loc='upper right')

    plt.savefig('3d')






if __name__ == '__main__':
    # problem_2a()
    #problem_2b()
    # problem_3a()
    problem_3d()
