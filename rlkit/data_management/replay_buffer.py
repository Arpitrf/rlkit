import abc
import h5py
import random
from filelock import FileLock
import numpy as np


class ReplayBuffer(object, metaclass=abc.ABCMeta):
    """
    A class used to save and replay data.
    """

    @abc.abstractmethod
    def add_sample(self, observation, action, reward, next_observation,
                   terminal, **kwargs):
        """
        Add a transition tuple.
        """
        pass

    @abc.abstractmethod
    def terminate_episode(self):
        """
        Let the replay buffer know that the episode has terminated in case some
        special book-keeping has to happen.
        :return:
        """
        pass

    @abc.abstractmethod
    def num_steps_can_sample(self, **kwargs):
        """
        :return: # of unique items that can be sampled.
        """
        pass

    # def add_path(self, path):
    #     """
    #     Add a path to the replay buffer.

    #     This default implementation naively goes through every step, but you
    #     may want to optimize this.

    #     NOTE: You should NOT call "terminate_episode" after calling add_path.
    #     It's assumed that this function handles the episode termination.

    #     :param path: Dict like one outputted by rlkit.samplers.util.rollout
    #     """
    #     for i, (
    #             obs,
    #             action,
    #             reward,
    #             next_obs,
    #             terminal,
    #             agent_info,
    #             env_info
    #     ) in enumerate(zip(
    #         path["observations"],
    #         path["actions"],
    #         path["rewards"],
    #         path["next_observations"],
    #         path["terminals"],
    #         path["agent_infos"],
    #         path["env_infos"],
    #     )):
    #         self.add_sample(
    #             observation=obs,
    #             action=action,
    #             reward=reward,
    #             next_observation=next_obs,
    #             terminal=terminal,
    #             agent_info=agent_info,
    #             env_info=env_info,
    #         )
        # self.terminate_episode()

    def add_paths(self, paths):
        # add sample to the replay buffer that is stored on disk
        with FileLock(self.hdf5_path + ".lock"):
            with h5py.File(self.hdf5_path, 'a') as file:
                for path in paths:
                    last_key = None
                    for last_key in file:
                        pass
                    # print("LAST KEY: ", last_key)
                    key_idx = int(last_key.split('_')[0])\
                        if last_key is not None else 0
                    while True:
                        group_key = f'{key_idx:09d}'
                        if (group_key + '_step00') not in file\
                                and (group_key + '_step00_last') not in file:
                            break
                        key_idx += 1 
                    # print("GROUP_KEY: ", group_key)
                    
                    # Loop over one episode
                    for step in range(len(path["observations"])):   
                        step_key = group_key + f'_step{step:02d}'
                        try:
                            group = file.create_group(step_key)
                        except Exception as e:
                            print(e, step_key)
                            group = file.create_group(
                                step_key + '_' +
                                str(random.randint(0, int(1e5))))
                        for key, value in path.items():
                            # print("key, value: ", key, type(value[step]), len(value[step])) 
                            try:
                                if key in ['agent_infos', 'env_infos']:
                                    continue
                                step_value = value[step]
                                # print("step_valueeee: ", len(step_value))
                                if type(step_value) == float\
                                        or type(step_value) == np.float64\
                                        or type(step_value) == str\
                                        or type(step_value) == int:
                                    group.attrs[key] = step_value
                                elif type(step_value) == list or isinstance(step_value, np.ndarray):
                                    # subgroup = group.create_group(key)
                                    # for i, item in enumerate(step_value):
                                    group.create_dataset(
                                        name=key,
                                        data=step_value,
                                        compression='gzip')
                                        # compression_opts=9)
                                else:
                                    group.create_dataset(
                                        name=key,
                                        data=step_value,
                                        compression='gzip',
                                        compression_opts=9)
                            except Exception as e:
                                print(f'[Memory] Dump key {key} error:', e)
                                # print(value)
                    
        # for path in paths:
        #     self.add_path(path)

    # @abc.abstractmethod
    # def random_batch(self, batch_size):
    #     """
    #     Return a batch of size `batch_size`.
    #     :param batch_size:
    #     :return:
    #     """
    #     pass

    def get_diagnostics(self):
        return {}

    def get_snapshot(self):
        return {}

    def end_epoch(self, epoch):
        return

