import argparse
import time

from Common.env import ReacherWrapper 
from Common.logger import Logger
import os
from sac import SAC
import numpy as np


def make_dir(dir_path):
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError:
        pass
    return dir_path

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--target_type', default='visual_reacher', type=str)
    parser.add_argument('--image_height', default=125, type=int)
    parser.add_argument('--image_width', default=200, type=int)
    parser.add_argument('--stack_frames', default=3, type=int)
    parser.add_argument('--tol', default=0.036, type=float)
    parser.add_argument('--image_period', default=1, type=int)
    parser.add_argument('--episode_length_step', default=75, type=int)

    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    parser.add_argument('--rad_offset', default=0.01, type=float)

    # train
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--env_steps', default=24000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--max_updates_per_step', default=1, type=float)

    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=1, type=int)
    parser.add_argument('--bootstrap_terminal', default=0, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_update_freq', default=1, type=int)
    # encoder
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)

    # misc
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_model_freq', default=1000, type=int)
    parser.add_argument('--load_model', default=-1, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--lock', default=False, action='store_true')
    parser.add_argument('--save_path', default='', type=str, help="For saving SAC buffer")
    parser.add_argument('--load_path', default='', type=str, help="Path to SAC buffer file")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    args.work_dir += f'/results/jax_run3/'

    make_dir(args.work_dir)

    model_dir = make_dir(os.path.join(args.work_dir, 'model'))
    args.model_dir = model_dir
    L = Logger(args.work_dir, use_tb=args.save_tb)

    image_shape = (args.image_height, args.image_width, 3*args.stack_frames)

    env = ReacherWrapper(args.tol, image_shape, args.image_period, use_ground_truth=True, chw=False)

    args.image_shape = env.image_space.shape
    args.proprioception_shape = env.proprioception_space.shape
    args.action_dim = env.action_space.shape[0]
    args.actor_input_dim = [(1, *image_shape), (1, *args.proprioception_shape)]
    args.critic_input_dim = [(1, *image_shape), (1, *args.proprioception_shape), (1, args.action_dim)]
    args.max_action = 1


    agent = SAC(args)

    # print(args.image_shape)
    # print(args.proprioception_shape)
    # print(args.action_dim) 
    # print(args.actor_input_dim)
    # print(args.critic_input_dim)
    # print(args.max_action)

    ## Define agent, sync weights
    
    episode, episode_reward, episode_step, done = 0, 0, 0, True
    image, propri = env.reset()
    # print(args.image_shape, image.shape, propri.shape)
    train_times = []
    # agent.send_init_ob((image, propri))
    start_time = time.time()
    for step in range(args.env_steps + args.init_steps):
        action = agent.select_action(image, propri).clip(-args.max_action, args.max_action)
        next_image, next_propri, reward, done, _ = env.step(action)

        episode_reward += reward
        episode_step += 1

        agent.replay_buffer.add(image, propri, action, reward, next_image, next_propri, 1.0 - float(done))
        
        if done or (episode_step == args.episode_length_step):
            L.log('train/duration', time.time() - start_time, step)
            L.log('train/episode_reward', episode_reward, step)
            L.dump(step)
            L.log('train/episode', episode+1, step)

            next_image, next_propri = env.reset() 
            episode_reward = 0
            episode_step = 0
            episode += 1
            start_time = time.time()
        
        image = next_image
        propri = next_propri

        if agent.replay_buffer.count > args.init_steps:
            sample = agent.replay_buffer.sample()
            t1 = time.time()
            agent.train(*sample)
            t2 = time.time()
            train_times.append(t2-t1)
            L.log('train/update_time', t2-t1, step)

    env.close()
    num_trains = len(train_times)
    avg_time = sum(train_times) / num_trains
    print(f'Number of training steps: {num_trains}')
    print(f'Average time per training step: {avg_time}')
    print('Train finished')

if __name__ == '__main__':
    main()


# Run 1
# Number of training steps: 24000
# Average time per training step: 0.009829623361428579

# Run 2
# Number of training steps: 24000
# Average time per training step: 0.009988719403743743