# Import required packages
import gymnasium as gym
import mani_skill.envs
import time
import matplotlib.pyplot as plt
import numpy as np
from mani_skill.utils.wrappers import RecordEpisode

# def move_forward():
#     # Define forward action
#     forward_action = [0.1, 0, 0, -1]
#     action = np.array(forward_action).reshape(1,4)


# Create the environment for right direction movement
env = gym.make("PickCube-v1", render_mode="rgb_array", control_mode = "pd_ee_delta_pos",
                    obs_mode="state_dict")
env = RecordEpisode(env, output_dir="Videos/TableTopRobot.mp4", save_trajectory=False,
                    save_video=True, video_fps=30, max_steps_per_video=100)

# Reset the environment
obs, _ = env.reset(seed=0)
env.unwrapped.print_sim_details()  # Print verbose details about the configuration

done = False
truncated = False

# Get the current end effector position
eeposition = np.array(obs['extra']['tcp_to_obj_pos'])
print("My pos is:", eeposition)
eeposx = eeposition[0][0]
eeposy = eeposition[0][1]
eeposz = eeposition[0][2]

# Define actions
forward_action = [0.1, 0, 0, -1]
backward_action = [-0.1, 0, 0, -1]
right_action = [0, 0.1, 0, -1]
left_action = [0, -0.1, 0, -1]
up_action = [0, 0, 0.1, -1]
do_nothing = [0, 0, 0, 0]

# Define grip action
grip_action = [0,0,0,-1]
grip_action = np.array(grip_action).reshape(1,4)

# Take user input
from language_interpretor import classify_text
from language_interpretor import input_classifier
direction = input_classifier()
#direction = input('What direction should I move in?\n')
print('Direction is :', direction)

if direction == 0:
    action = np.array(forward_action).reshape(1,4)

elif direction == 1:
    action = np.array(backward_action).reshape(1,4)

elif direction == 2:
    action = np.array(right_action).reshape(1,4)

elif direction == 3:
    action = np.array(left_action).reshape(1,4)

elif direction == 4:
    action = np.array(up_action).reshape(1,4)

else:
    action = np.array(do_nothing).reshape(1,4)
    print("I am sorry; I don't know that direction. I will not move the cube.")


# Run the environment loop
# First move to the cube position
while not done and not truncated:
    eeposition = np.array(obs['extra']['tcp_to_obj_pos'])
    eeposx = eeposition[0][0]
    eeposy = eeposition[0][1]
    eeposz = eeposition[0][2]
    move_action = [eeposx,eeposy,eeposz,1]
    move_action = np.array(move_action).reshape(1,4)
    obs, rew, done, truncated, info = env.step(move_action)

# Grip the cube
done = False
truncated = False
while not done and not truncated:
    obs, rew, done, truncated, info = env.step(grip_action)

# Move the action
done = False
truncated = False
i = 0
while not done and not truncated:
    obs, rew, done, truncated, info = env.step(action)
    if i < 20:
        done = False
        truncated = False
    i = i+1
# Close the environment
env.close()