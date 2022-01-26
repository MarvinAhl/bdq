import sys
sys.path.insert(1, '../..')

from bdq import BDQ

import pygame
import physics
from rocket import Rocket

import torch

import numpy as np
from matplotlib import pyplot as plt

class Game:
    def __init__(self, render=True, agent_play=True, agent_train=True, agent_file='rocket_game_land', save_episodes=100, eval_every_episodes=25, eval_episodes=25,
                 step_limit=2000, device='cpu'):
        self.running = True
        self.display_surf = None
        self.size = (self.width, self.height) = (1280, 720)

        self.rocket = Rocket()
        self.last_dirty_rects = []

        self.agent_play = agent_play
        self.agent_train = agent_train
        self.agent_file = agent_file

        if agent_play:
            sampling_period = 0.01
            lookahead_horizon = 5.0
            gamma = np.exp(-sampling_period/lookahead_horizon)  # Calculate discount factor
            new_actions_prob = 5 * sampling_period  # Calculate e-greedy new action probability
            
            self.agent = BDQ(6, (2, 5, 3), new_actions_prob=new_actions_prob, gamma=gamma, device=device)
            
            if not agent_train:
                self.agent.load_net('nets/' + agent_file + '.net')
        else:
            self.agent = None

        self.last_shaping = None  # For potential based reward shaping
        
        self.save_episodes = save_episodes
        self.eval_every_episodes = eval_every_episodes
        self.eval_episodes = eval_episodes
        self.step_limit = step_limit

        self.render = render

        # For measuring frame time
        self.last_time_ms = 0

    def _init(self):
        pygame.init()
        self.display_surf = pygame.display.set_mode(self.size, pygame.SCALED, vsync=1)
        pygame.display.set_caption('Rocket Game')

        self.rocket.init()

    def _on_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        
        # Controls for human play
        if not self.agent_play:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.rocket.set_f1(1.0)  # Activate Booster
                elif event.key == pygame.K_a:
                    self.rocket.set_theta(1.0)  # Angle Booster left
                elif event.key == pygame.K_d:
                    self.rocket.set_theta(-1.0)  # Angle Booster right
                elif event.key == pygame.K_LEFT:
                    self.rocket.set_f2(-1.0)  # Fire right control nozzle
                elif event.key == pygame.K_RIGHT:
                    self.rocket.set_f2(1.0)  # Fire left control nozzle

            elif event.type == pygame.KEYUP:
                pressed_keys = pygame.key.get_pressed()
                if event.key == pygame.K_SPACE:
                    self.rocket.set_f1(0.0)  # Booster engine off
                elif event.key == pygame.K_a:
                    if pressed_keys[pygame.K_d]:
                        self.rocket.set_theta(-1.0)  # Angle Booster right
                    else:
                        self.rocket.set_theta(0.0)  # Angle Booster center
                elif event.key == pygame.K_d:
                    if pressed_keys[pygame.K_a]:
                        self.rocket.set_theta(1.0)  # Angle Booster left
                    else:
                        self.rocket.set_theta(0.0)  # Angle Booster center
                elif event.key == pygame.K_LEFT:
                    if pressed_keys[pygame.K_RIGHT]:
                        self.rocket.set_f2(1.0)  # Fire left control nozzle
                    else:
                        self.rocket.set_f2(0.0)  # Stop firing control nozzle
                elif event.key == pygame.K_RIGHT:
                    if pressed_keys[pygame.K_LEFT]:
                        self.rocket.set_f2(-1.0)  # Fire right control nozzle
                    else:
                        self.rocket.set_f2(0.0)  # Stop firing control nozzle

    def _update(self, dt, actions):
        if self.agent_play:
            self.rocket.control(actions)

        state = self.rocket.update(dt)

        # Place to differntiate between shape and observation
        obsv = state.copy()

        done = False

        reward = 0.0 if not self.agent_play or actions[0] == 0 else -0.02  # Reward if booster is off or on

        engine_on_ground = 0.0 <= (obsv[1] + self.rocket.l1 * np.cos(obsv[4]))
        nose_on_ground = 0.0 <= (obsv[1] - self.rocket.l2 * np.cos(obsv[4]))
        
        out_of_bounds_left = obsv[0] < -(self.width/2 / physics.pixel_per_meter)
        out_of_bounds_right = obsv[0] > (self.width/2 / physics.pixel_per_meter)
        out_of_bounds_top = obsv[1] < -(self.height / physics.pixel_per_meter)

        flying_upwards = obsv[3] < 0.0

        if out_of_bounds_left or out_of_bounds_right or out_of_bounds_top:
            done = True
            reward += -100.0  # Reward for flying out of bounds
        elif engine_on_ground or nose_on_ground:
            done = True

            x_good = obsv[0] < 10.0 and obsv[0] > -10.0  # Landing on pad
            x_v_good = obsv[2] < 5.0 and obsv[2] > -5.0
            y_v_good = obsv[3] < 10.0 and obsv[3] > -10.0
            phi_good = obsv[4] < 0.2 and obsv[4] > -0.2
            phi_v_good = obsv[5] < 0.4 and obsv[5] > -0.4

            rocket_landed = x_good and x_v_good and y_v_good and phi_good and phi_v_good
            if rocket_landed:
                reward += 100.0  # Reward for landing

                # Rewards for being on point
                reward += self._gauss_reward(30.0, 10.0, 0.4, obsv[0])
                reward += self._gauss_reward(20.0, 3.0, 0.15, obsv[2])
                reward += self._gauss_reward(20.0, 1.0, 0.15, obsv[3])
                reward += self._gauss_reward(20.0, 0.1, 0.15, obsv[4])
                reward += self._gauss_reward(20.0, 0.2, 0.15, obsv[5])
            
            else:
                reward += -100.0
        elif flying_upwards:
            done = True
            reward += -100.0  # Flying upwards is not permitted due to waste of fuel

        shaping = self._calc_shaping(obsv)
        reward += shaping - self.last_shaping
        self.last_shaping = shaping

        return obsv, reward, done

    def _calc_shaping(self, state):
        # Square potentials scaled to have about the same impact
        position = -0.8 * np.sqrt(state[0]**2 + state[1]**2)
        velocity = -1.7 * np.sqrt(state[2]**2 + state[3]**2)
        angle = -5.0 * state[4]**2
        ang_vel = -2 * state[5]**2

        # Scaled importance
        return position + velocity + angle + ang_vel

    def _render(self):
        ppm = physics.pixel_per_meter

        dirty_rects = []  # Store alle areas that changed in here
        self.display_surf.fill((255, 255, 255))

        (r_x1, r_y1), (r_x2, r_y2), (r_xb, r_yb), (r_xn, r_yn) = self.rocket.render()

        # The coordinate systems origin is at the bottom center of the screen
        x1 = r_x1 + self.width / 2
        y1 = r_y1 + self.height
        x2 = r_x2 + self.width / 2
        y2 = r_y2 + self.height

        xb = r_xb + self.width / 2
        yb = r_yb + self.height
        xn = r_xn + self.width / 2
        yn = r_yn + self.height

        burn_line = pygame.draw.line(self.display_surf, (239, 151, 0), (x1, y1), (xb, yb), 3)
        nozzle_line = pygame.draw.line(self.display_surf, (239, 151, 0), (x2, y2), (xn, yn), 1)

        rocket_circ1 = pygame.draw.circle(self.display_surf, (0, 0, 0), (x1, y1), 3)
        rocket_circ2 = pygame.draw.circle(self.display_surf, (0, 0, 0), (x2, y2), 3)
        rocket_line = pygame.draw.line(self.display_surf, (0, 0, 0), (x1, y1), (x2, y2), 5)

        # Ground and landing pad
        pygame.draw.rect(self.display_surf, (0, 0, 0), (0, self.height - physics.ground_height * ppm, self.width, physics.ground_height * ppm))
        pygame.draw.line(self.display_surf, (239, 151, 0), (self.width/2 - 10 * ppm, self.height - physics.ground_height * ppm), (self.width/2 + 10 * ppm, self.height - physics.ground_height * ppm), 5)

        new_dirty_rects = [rocket_circ1, rocket_circ2, rocket_line, burn_line, nozzle_line]
        dirty_rects = self.last_dirty_rects + new_dirty_rects

        pygame.display.update(dirty_rects)

        self.last_dirty_rects = new_dirty_rects

    def _cleanup(self):
        pygame.quit()

    def play(self):
        if self._init() == False:
            self.running = False

        episode_rewards = []
        steps = []
        final_obsvs = []
        eval_rewards = []

        episode = 0

        # Loop over all Episodes:
        while self.running:
            episode += 1

            if self.agent_play and self.agent_train and episode % self.eval_every_episodes == 0:
                eval_reward = self._evaluate()
                eval_rewards.append(eval_reward)

            print(f'Episode {episode} started')

            state, done = self.rocket.reset(), False

            obsv = state.copy()

            self.last_shaping = self._calc_shaping(obsv)  # Calculate initial potential

            episode_reward = 0.0
            step_count = 0

            self._init_time()
            # Loop for one Episode:
            while self.running and not done:
                dt = 0.01  # Will always calculate 100 steps per second, even if realtime simulation is slower

                for event in pygame.event.get():
                    self._on_event(event)
                
                if self.agent_play and self.agent_train:
                    actions = self.agent.act(obsv)  # Exploration strategy for training
                elif self.agent_play and not self.agent_train:
                    actions = self.agent.act_optimally(obsv)  # Optimal strategy for evaluation
                elif not self.agent_play:
                    actions = None  # No action because human inputs work with pygame events

                next_obsv, reward, done = self._update(dt, actions)
                
                if self.agent_play and self.agent_train:
                    self.agent.experience(obsv, actions, reward, next_obsv, done)
                    self.agent.train()

                if self.render:
                    self._render()
                    self._delay(dt)  # Wait in render mode after each calculation to ensure simulation is in real time
                
                obsv = next_obsv

                episode_reward += reward
                step_count += 1

                # Stop Episode if time limit is reached
                if step_count >= self.step_limit and self.agent_play:
                    done = True
            
            if self.agent_play:
                print(f'Reward: {episode_reward}')
            else:
                print(f'x: {obsv[0]}')
                print(f'y: {obsv[1]}')
                print(f'x_v: {obsv[2]}')
                print(f'y_v: {obsv[3]}')
                print(f'phi: {obsv[4]}')
                print(f'phi_v: {obsv[5]}')
                print(f'Reward: {episode_reward}')
                print('')

            episode_rewards.append(episode_reward)
            overall_steps = step_count if len(steps) < 1 else steps[-1] + step_count
            steps.append(overall_steps)
            final_obsvs.append(obsv)

            # Print training stats
            if (episode % self.save_episodes == 0 or not self.running) and self.agent_play and self.agent_train:
                self.agent.save_net('nets/' + self.agent_file + f'_e{episode}' + '.net')

                # Plot results
                plt.clf()  # To prevent overlapping of old plots

                figure, axis = plt.subplots(3, 1)
                figure.suptitle('Training Stats')

                axis[0].plot(np.arange(episode) + 1, episode_rewards, 'k-')
                axis[0].grid(True)
                axis[0].set_ylabel('Reward')

                episode_steps = [steps[0]]
                episode_steps += [steps[i+1] - steps[i] for i in range(episode-1)]
                axis[1].plot(np.arange(episode) + 1, episode_steps, 'k-')
                axis[1].grid(True)
                axis[1].set_ylabel('Episode Steps')

                axis[2].plot(np.arange(episode) + 1, steps, 'k-')
                axis[2].grid(True)
                axis[2].set_ylabel('Cumulative Steps')
                axis[2].set_xlabel('Episodes')

                plt.savefig('stats/' + f'train_stats_e{episode}.png')

                plt.close(figure)

                # Plot final obsvs
                plt.clf()  # To prevent overlapping of old plots

                figure, axis = plt.subplots(3, 2)
                figure.suptitle('Final Observations')

                axis[0, 0].plot(np.arange(episode) + 1, [obsv[0] for obsv in final_obsvs], 'k.')
                axis[0, 0].grid(True)
                axis[0, 0].set_ylabel('x/y Position')

                axis[0, 1].plot(np.arange(episode) + 1, [obsv[1] for obsv in final_obsvs], 'k.')
                axis[0, 1].grid(True)

                axis[1, 0].plot(np.arange(episode) + 1, [obsv[2] for obsv in final_obsvs], 'k.')
                axis[1, 0].grid(True)
                axis[1, 0].set_ylabel('x/y Velocity')

                axis[1, 1].plot(np.arange(episode) + 1, [obsv[3] for obsv in final_obsvs], 'k.')
                axis[1, 1].grid(True)

                axis[2, 0].plot(np.arange(episode) + 1, [obsv[4] for obsv in final_obsvs], 'k.')
                axis[2, 0].grid(True)
                axis[2, 0].set_ylabel('phi Angle/Velocity')
                axis[2, 0].set_xlabel('Episodes')

                axis[2, 1].plot(np.arange(episode) + 1, [obsv[5] for obsv in final_obsvs], 'k.')
                axis[2, 1].grid(True)
                axis[2, 1].set_xlabel('Episodes')

                plt.savefig('stats/' + f'train_final_obsvs_e{episode}.png')

                plt.close(figure)

                plt.clf()

                plt.plot((np.arange(len(eval_rewards))+1) * self.eval_every_episodes, eval_rewards, 'k-')
                plt.grid(True)
                plt.title(f'Evaluation Results every {self.eval_every_episodes} episodes for {self.eval_episodes} Runs using Greedy Policy')
                plt.xlabel('Episodes')
                plt.ylabel('Reward')

                plt.savefig('stats/' + f'train_eval_rewards_e{episode}.png')

                print(f'Episode {episode} saved')

        self._cleanup()
    
    def _evaluate(self):
        overall_reward = 0.0
        for _ in range(self.eval_episodes):
            episode_reward, step_count = 0.0, 0
            state, done = self.rocket.reset(), False
            obsv = state.copy()

            self.last_shaping = self._calc_shaping(obsv)  # Calculate initial potential

            self._init_time()
            # Loop for one Episode:
            while self.running and not done:
                dt = 0.01  # Will always calculate 100 steps per second, even if realtime simulation is slower
                actions = self.agent.act_optimally(obsv)
                next_obsv, reward, done = self._update(dt, actions)
                obsv = next_obsv

                episode_reward += reward
                step_count += 1

                # Stop Episode if time limit is reached
                if step_count >= self.step_limit:
                    done = True
            
            overall_reward += episode_reward
        return overall_reward / self.eval_episodes

    
    def _init_time(self):
        self.old_time_ms = pygame.time.get_ticks()

    def _delay(self, dt):
        new_time_ms = pygame.time.get_ticks()
        delta_time_ms = new_time_ms - self.old_time_ms
        self.old_time_ms = new_time_ms

        d_d_time_ms = int(dt * 1000 - delta_time_ms)

        if d_d_time_ms > 0:
            pygame.time.delay(d_d_time_ms)

    def _gauss_reward(self, a, b, c, x):
        """
        Calculate unnormalized Gauss Curve with maximum height of a at x=0 that
        decays to c*a at x=b. Return value at point x. Is used for the reward function.
        """
        temp = np.sqrt(-np.log(c**2))
        alpha = np.sqrt(2*np.pi) * a * b / temp
        sigma = b / temp

        return alpha / sigma / np.sqrt(2*np.pi) * np.exp(-x**2 / 2 / sigma**2)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {device}')

    game = Game(render=True, agent_play=True, agent_train=False, agent_file='rocket_game_land_e1900', step_limit=2000, device=device)
    game.play()