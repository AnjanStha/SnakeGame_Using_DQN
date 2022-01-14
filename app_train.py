import numpy as np

from parameters import *
import environment
import agent
import q_learning

net = q_learning.Neural_Network()
env = environment.Environment(ROW)
buffer = agent.ExperienceBuffer(REPLAY_SIZE)
agent = agent.Agent(env, buffer)
dqn = q_learning.DQN(net, buffer, agent, load=LOAD)

flag = True
count = 0
#ep = 0
while True:
    dqn.simulate()
   # ep = 0
   #ep += 1
   # print("training==", ep)

    if dqn.super_light_api() == "Finished":
        break

    if count % 1000 == 0:
        epsilon, mean_reward, steps, generation, score = dqn.light_api()
        print("Generation", generation, "Mean reward", mean_reward, "Epsilon", epsilon, "score=", score)
        #dqn.save()
        count = 0

    count += 1
