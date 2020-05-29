
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as patches


k=0

def render1():
    global k
    MAP = np.zeros((5, 5, 3)) #init as all white
    MAP.fill(1)
    length =env.length
    for i in range(4):
        #a means passenger
        if env.passenger_status & (1<<i):
            x,y = env.possible_passenger_loc[i]
#             print(x,y)
            MAP[x, y] =(1,0,0)
            
            
#     print(MAP)
    fig = plt.figure(frameon=False, figsize=(1, 1), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(MAP)
    #state 4 is empty  
    if env.taxi_status == 4:
        dot = patches.Circle((env.x,env.y),0.3,linewidth=1,edgecolor='b',facecolor='b')
        ax.add_patch(dot)
    else:
        dot = patches.Circle((env.x,env.y),0.3,linewidth=1,edgecolor='g',facecolor='g')
        ax.add_patch(dot)
        dest_x,dest_y = env.possible_passenger_loc[env.taxi_status]
        dot1 = patches.Rectangle((dest_x-0.2,dest_y-0.2),0.4,0.4,linewidth=1,edgecolor='black',facecolor='black')
        ax.add_patch(dot1)
#     plt.show()
    print(k)
    plt.savefig('images_b/val/'+str(k)+'.png')
    plt.close()
    k+=1
#     for line in MAP:
#         print(''.join([i.decode('UTF-8') for i in line]))
    if env.taxi_status == 4:
        print('Empty Taxi')
    else:
        x,y = env.possible_passenger_loc[env.taxi_status]
        print('Taxi destination:({},{})'.format(x,y))
        

        
def render2():
    f=open('matrix_b/val/mat.npy','ab')
    global k
    MAP = np.zeros((5, 5, 3))
    #1st layer: passengers pop-up; 2nd: taxi empty or not; 3rd: taxi destination
    for i in range(4):    
        if env.passenger_status & (1<<i):
            x,y = env.possible_passenger_loc[i]
#             print('passengers at ', x, y)
            MAP[x, y] =(1,0,0)
    
    if env.taxi_status == 4:
        MAP[env.x,env.y] =(0,1,0)
    else:
        MAP[env.x,env.y] =(0,2,0)
        dest_x,dest_y = env.possible_passenger_loc[env.taxi_status]
        MAP[dest_x, dest_y] = (0,0,1)
#     print(MAP[:,:,0])
#     print(MAP[:,:,1])
#     print(MAP[:,:,2])
    print(k)
    k+=1
#     if env.taxi_status == 4:
#         print('Empty Taxi')
#     else:
#         x,y = env.possible_passenger_loc[env.taxi_status]
#         print('Taxi destination:({},{})'.format(x,y))
    np.save(f, MAP)
    f.close()
        
        
        
def roll_out(state_num, env, policy, num_trajectory, truncate_size):
    SASR = []
    total_reward = 0.0
    frequency = np.zeros(state_num)
    for i_trajectory in range(num_trajectory):
        state = env.reset()
        for i_t in range(truncate_size):
            env.render()
            p_action = policy[state, :]
            # action = np.random.choice(p_action.shape[0], 1, p=p_action)[0]
            action = np.random.choice(list(range(p_action.shape[0])),
                                      p=p_action)
            next_state, reward = env.step(action)

            SASR.append((state, action, next_state, reward))
            frequency[state] += 1
            total_reward += reward
            print (i_trajectory, i_t, reward)
            # a = input()
            state = next_state
    mean_reward = total_reward / (num_trajectory * truncate_size)
    return SASR, frequency, mean_reward


from environment import taxi
import numpy as np
import random
import torch 

length = 5
env = taxi(length)
n_state = env.n_state
n_action = env.n_action
print_freq=20
env.render=render2

alpha = np.float(0.6)
pi_eval = np.load('taxi-policy/pi19.npy')
pi_behavior = np.load('taxi-policy/pi3.npy')
pi_behavior = alpha * pi_eval + (1-alpha) * pi_behavior
# roll_out(n_state, env, pi_behavior, 1, 100)
SASR_b, b_freq, _ = roll_out(n_state, env, pi_behavior, 1, 250000)  #1, 1000000
np.save('matrix_b/val/SASR.npy', np.asarray(SASR_b))