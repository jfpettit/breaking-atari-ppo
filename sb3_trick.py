import streamlit as st

import matplotlib.pyplot as plt
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from gym.wrappers import Monitor
import os


st.write("""
         # Breaking a Pong-playing PPO agent

        With this demo, you can try to break a Pong-playing agent trained using the [PPO](https://arxiv.org/abs/1707.06347) algorithm.
        Of course, it's really easy to break, since neural networks tend to be really brittle.
        The sliders in the sidebar let you set different attributes of a patch applied to the observations.

        ## What was the workflow here?

        This agent was trained using PPO, implemented in [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/).
        The pretrained model gets around 19 reward in evaluation. So, if its getting much lower than that, then you're successfully breaking it.

        ## Tips

        If you want to replicate the shape of the pong ball, set the pixel width at 2 and height at 5. Set the "Pixel intensity to fill mask with" so that the mask is white-toned, to better replicate the ball. I found a pixel intensity of 226 is pretty close.

        If you want to download the video produced by your mask, just right-click on it and select the download option from the menu.

        ## Example image

        Below is an example of what the mask looks like on the image. It should update as you change the values via the sliders, so use it to set your
        mask up how you want to.
""")

class ImageObsMask(gym.Wrapper):
    def __init__(self, env, fill_val=0, width=25, height=25, xloc=75, yloc=75):
        super().__init__(env)

        self.width = width
        self.height = height
        self.xloc = xloc
        self.yloc = yloc
        self.fill_val = fill_val

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        obs[self.yloc-self.height//2:self.yloc+self.height//2, self.xloc-self.width//2:self.xloc+self.width//2] = self.fill_val
        return obs, rew, done, info

    def render(self, mode='human'):
        img = self.env.ale.getScreenRGB2()
        img[self.yloc-self.height//2:self.yloc+self.height//2, self.xloc-self.width//2:self.xloc+self.width//2] = self.fill_val
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

def build_env(fill_val, width, height, xloc, yloc):
    env = gym.make("PongNoFrameskip-v4")
    env = ImageObsMask(env, fill_val, width, height, xloc, yloc)
    env_fn = lambda: Monitor(env, f"atari_vids/", force=True)
    env = make_atari_env(env_fn, n_envs=1)
    env = VecFrameStack(env, n_stack=4)
    return env

episodes = st.sidebar.number_input("Number of episodes to evaluate over.", value=10, min_value=1, max_value=1000)
fill_val = st.sidebar.slider("Pixel intensity to fill mask with", value=0, min_value=0, max_value=255)
width = st.sidebar.slider("Width of the mask.", value=25, min_value=0, max_value=159)
height = st.sidebar.slider("Height of the mask.", value=25, min_value=0, max_value=209)
xloc = st.sidebar.slider("Where to center the mask in the x direction.", value=75, min_value=0, max_value=159)
yloc = st.sidebar.slider("Where to center the mask in the y direction.", value=75, min_value=0, max_value=209)
def run(episodes, fill_val, width, height, xloc, yloc):

    env = build_env(fill_val, width, height, xloc, yloc)
    model = PPO.load("models/PongNoFrameskip-v4_1618178168.zip", env=env)
    mean_rew, std_rew = evaluate_policy(model, model.get_env(), n_eval_episodes=episodes, deterministic=True)
    st.write(f"Mean evaluation reward: {mean_rew}\tStd evaluation reward: {std_rew}")

    path_list = os.listdir(os.getcwd() + "/atari_vids/")
    for pth in path_list:
        if ".mp4" in pth:
            video_path = "atari_vids/" + pth
            st.video(video_path)
            break

dummy_env = gym.make("PongNoFrameskip-v4")
dummy_env = ImageObsMask(dummy_env, fill_val, width, height, xloc, yloc)
obs = dummy_env.reset()
obs, _, _, _ = dummy_env.step(dummy_env.action_space.sample())
fig, ax = plt.subplots()
ax.imshow(obs)
st.pyplot(fig)


if st.sidebar.button("Run!"):
    with st.spinner("Evaluating the pretrained PPO agent on your mask setup..."):
        run(episodes, fill_val, width, height, xloc, yloc)
