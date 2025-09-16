#!/usr/bin/env python
from __future__ import print_function

import csv
import pandas as pd
import glob
import os
import random
import time
import json
import re
from enum import IntEnum
from networkx.readwrite import json_graph
import numpy as np
from airlift.envs import AirliftEnv
from airlift.utils.running_stats import RunningStat


class Status(IntEnum):
    SUCCESSFUL_CREATION = 0
    STOPPED_TOO_MANY_MISSED = 1
    FINISHED_ALL_SCENARIOS = 2


"""
Assumes the scenarios follow this structure:

    .
    ├── Test_0
    │   ├── Level_1.pkl
    │   ├── .......
    │   ├── .......
    │   └── Level_99.pkl
    ├── Test_1
    │   ├── Level_1.pkl
    │   ├── .......
    │   ├── .......
    │   └── Level_99.pkl
    ├── ...... 
"""


class LocalEvaluationService:
    def __init__(
            self,
            test_env_folder="/tmp",
            output_dir="./",
            submission_id=-1,
            fail_threshold=0.3,
            first_step_timeout_in_seconds=600,
            step_timeout_in_seconds=60,
            render=False,
    ):

        self.test_env_folder = test_env_folder
        self.output_dir = output_dir
        self.submission_id = submission_id
        self.FAIL_THRESHOLD = fail_threshold
        self.FIRST_STEP_TIMEOUT_IN_SECONDS = first_step_timeout_in_seconds
        self.STEP_TIMEOUT_IN_SECONDS = step_timeout_in_seconds
        self.render = render

        self.overall_start_time = 0
        self.step_time_start = -1
        self.num_tests_completed = 0

        self.status = Status.SUCCESSFUL_CREATION
        self._env = None

        self.simulation_scores = []
        self.simulation_scores_normalized = []
        self.simulation_missed_deliveries = []
        self.simulation_percentage_missed_per_test = {}
        self.simulation_steps = []
        self.simulation_times = []
        self.begin_simulation = False

        self.simulation_count = -1
        self.current_step = 0
        self.current_test = None
        self.current_level = -1

        self.test_ids = []
        self.level_ids = []
        self.env_filenames = []
        self.seeds = []
        self.baseline_scores = []
        self.random_scores = []
        self.read_metadata()

        self.num_scenarios = len(self.test_ids)

        self.episode_step_time_stat = RunningStat()

    def read_metadata(self):
        df = pd.read_csv(self.test_env_folder + "/metadata.csv")
        for index, row in df.iterrows():
            self.test_ids.append(df.at[index, 'test_id'])
            self.level_ids.append(df.at[index, 'level_id'])
            self.env_filenames.append(df.at[index, 'filename'])
            self.seeds.append(int(df.at[index, 'seed']))
            self.baseline_scores.append(float(df.at[index, 'baseline_score']))
            self.random_scores.append(float(df.at[index, 'random_score']))

    def exceeded_missed_threshold(self, test_id):
        assert self.current_test in self.simulation_percentage_missed_per_test, \
            "No environment was finished at all during test {}!".format(test_id)

        mean_missed = np.mean(self.simulation_percentage_missed_per_test[test_id])
        if mean_missed > self.FAIL_THRESHOLD:
            print("The solution missed an average of {0} percent of the deliveries on the last test, which exceeds the threshold of {1}".format(100*mean_missed, 100*self.FAIL_THRESHOLD))
            return True
        else:
            print("The solution missed an average of {0} percent of the deliveries on the last test".format(100*mean_missed))

    def env_create(self):
        """
        Handles a ENV_CREATE command from the client
        """

        # Make sure status is valid to continue
        if self.status == Status.STOPPED_TOO_MANY_MISSED:
            raise Exception(
                "Client attempted to create an environment when too many deliveries have been missed.")
        elif self.status == Status.FINISHED_ALL_SCENARIOS:
            raise Exception(
                "Client attempted to create an environment when all scenarios have been finished.")

        # Very first episode? Start the overall timer
        if self.simulation_count == -1:
            self.overall_start_time = time.time()

            # Create a default scores.txt at first test. All scores are assumed to be zero.
            scores_file = open(self.output_dir + "scores.txt", "w")
            scores_file.write("MEAN_Score: {:.3f}\n".format(0))
            scores_file.write("SUM_Score: {:.3f}\n".format(0))
            scores_file.write("MEAN_Normalized_Score: {:.3f}\n".format(0))
            scores_file.write("MEAN_Percent_Missed: {:.3f}\n".format(0))
            scores_file.write("set1_score: {:.3f}\n".format(0)) # SUM Normalized Score
            scores_file.write("Duration: {:.3f}\n".format(0))
            scores_file.close()


        # Check if test and/or evaluation is complete and do the following:
        # - Set evaluation_done if evaluation is complete (no more scenarios or too many missed deliveries)
        # - Set appropriate return status if evaluation is done
        # - Advance the test counter if the test completed
        evaluation_done = False
        if self.simulation_count + 1 >= self.num_scenarios:
            # We have finished all the available scenarios
            evaluation_done = True
            if self.exceeded_missed_threshold(self.current_test):
                # If missed threshold was exceeded on last test mark as completing due to too many missed
                # and do not consider the test complete
                self.status = Status.STOPPED_TOO_MANY_MISSED
                self.create_scores_file()
            else:
                self.status = Status.FINISHED_ALL_SCENARIOS
                self.num_tests_completed += 1  # Consider the last test complete

                # Account for final test scores.txt
                self.create_scores_file()

        else:
            # Check the next test to see if we have finished the current test
            next_test = self.test_ids[self.simulation_count + 1]
            if self.current_test is not None and self.current_test != next_test:
                if self.exceeded_missed_threshold(self.current_test):
                    evaluation_done = True
                    self.status = Status.STOPPED_TOO_MANY_MISSED
                    self.create_scores_file()
                else:
                    self.num_tests_completed += 1

                    # Account for scores.txt at every test before reaching final test.
                    self.create_scores_file()

        if evaluation_done:
            _observation = False
        else:
            """
            There are still test envs left that are yet to be evaluated
            """
            self.simulation_count += 1
            self.current_test = self.test_ids[self.simulation_count]
            self.current_level = self.level_ids[self.simulation_count]
            test_env_file_path = self.env_filenames[self.simulation_count]

            print("=" * 15)
            print("Evaluating {} ({}/{})".format(test_env_file_path, self.simulation_count + 1, self.num_scenarios))

            del self._env
            self._env = AirliftEnv.load(os.path.join(self.test_env_folder, test_env_file_path))

            self.begin_simulation = time.time()

            # Add placeholders for the new episode
            self.simulation_scores.append(0)
            self.simulation_scores_normalized.append(0)
            self.simulation_missed_deliveries.append(0)
            self.simulation_times.append(0)
            self.simulation_steps.append(0)

            self.current_step = 0
            _observation = self._env.reset(seed=self.seeds[self.simulation_count])

            self.status = Status.SUCCESSFUL_CREATION

        return _observation, None, self.status

    def get_done_status(self):
        return all(self._env.dones.values())

    def env_step(self, command):
        """ Handles a ENV_STEP command from the client
        """
        # add in different time outs (first step is allowed to have up to a 10 min timeout)
        # add in a null action instead of returning
        timeout = False
        if self.step_time_start != -1:
            # give 10 minutes
            if self.current_step == 0:
                if (time.time() - self.step_time_start) > self.FIRST_STEP_TIMEOUT_IN_SECONDS:
                    timeout = True
            elif (time.time() - self.step_time_start) > self.STEP_TIMEOUT_IN_SECONDS:
                timeout = True

        if timeout:
            action = None
        else:
            action = command

        if all(self._env.dones.values()):
            raise Exception(
                "Client attempted to perform an action on an Env which is done")

        step_start_time = time.time()
        _observation, all_rewards, done, info = self._env.step(action)
        self.episode_step_time_stat.update(time.time() - step_start_time)

        for agent in info:
            info[agent]["timeout"] = timeout

        if self.render:
            self._env.render()

        self.current_step += 1

        self.simulation_steps[-1] += 1

        # Is the episode over?
        if all(self._env.dones.values()):
            if self.begin_simulation:
                # If begin simulation has already been initialized at least once
                # This adds the simulation time for the previous episode
                self.simulation_times[-1] = time.time() - self.begin_simulation

            # Compute percentage complete
            percentage_missed = self._env.metrics.missed_deliveries / self._env.metrics.total_cargo_generated
            self.simulation_scores[-1] = self._env.metrics.score
            self.simulation_missed_deliveries[-1] = percentage_missed

            # adds 1.0 so we can add them up
            self.simulation_scores_normalized[-1] += \
                (self.random_scores[self.simulation_count] - self._env.metrics.score) \
                / (self.random_scores[self.simulation_count] - self.baseline_scores[self.simulation_count])

            if self.simulation_count == 0:
                file = open(self.output_dir + "breakdown_results.csv", "w")
                data = "Filename, "
                for name in zip(self._env.metrics._fields):
                    data += str(name[0]) + ","
                data += "Simulation Time ,"
                data += "Normalized Score ,"
                data += "Simulation Steps"
                data += "\n"
                file.write(data)
                file.close()

            file = open(self.output_dir + "breakdown_results.csv", "a")
            data = str(self.env_filenames[self.simulation_count]) + ","

            for i in range(len(self._env.metrics)):
                data += str(self._env.metrics[i]) + ","
            data += str(self.simulation_times[self.simulation_count]) + ","
            data += str(self.simulation_scores_normalized[self.simulation_count]) + ","
            data += str(self.simulation_steps[self.simulation_count])
            data += "\n"
            file.write(data)
            file.close()
            #data = str(self.env_filenames[self.simulation_count]) + "," \
            #       + str(self.simulation_scores[self.simulation_count]) \
            #       + "," + str(self.simulation_scores_normalized[self.simulation_count]) \
            #       + "," + str(self.simulation_missed_deliveries[self.simulation_count]) \
            #       + "," + str(self._env.metrics.total_cost) \
            #       + "," + str(self._env.metrics.average_cost_per_plane) \
            #       + "," + str(self._env.metrics.total_lateness) \
            #       + "," + str(self._env.metrics.average_lateness_per_plane) \
            #       + "," + str(self._env.metrics.total_steps) \
            #       + "," + str(self._env.metrics.average_steps) \
            #       + "," + str(self._env.metrics.total_waiting_steps) \
            #       + "," + str(self._env.metrics.total_malfunctions) \
            #       + "," + str(self._env.metrics.missed_deliveries) \
            #       + "," + str(self._env.metrics.total_rewards_for_all_agents) \
            #       + "," + str(self._env.metrics.average_rewards_for_all_agents) + "\n"


            if self.current_test not in self.simulation_percentage_missed_per_test:
                self.simulation_percentage_missed_per_test[self.current_test] = []
            self.simulation_percentage_missed_per_test[self.current_test].append(percentage_missed)
            print("Percentage of deliveries missed for test {}, level {}: {}".format(self.current_test,
                                                                                     self.current_level,
                                                                                     100*percentage_missed))

            print(
                "Episode finished in {} timesteps, {:.3f} seconds. Percentage deliveries missed: {:.3f}. Normalized score: {:.3f}.".format(
                    self.simulation_steps[-1],
                    self.simulation_times[-1],
                    100*self.simulation_missed_deliveries[-1],
                    self.simulation_scores_normalized[-1],
                ))

            print("Overall score so far: {:.3f}".format(sum(self.simulation_scores_normalized)))

        self.step_time_start = time.time()

        return _observation, all_rewards, done, info

    def submit(self):
        """
        Handles a ENV_SUBMIT command from the client
        """

        # Register simulation time of the last episode
        self.simulation_times.append(time.time() - self.begin_simulation)

        # Compute the evaluation metadata for the last episode
        mean_score, sum_score, mean_normalized_score, sum_normalized_score, mean_cargo_percentage_missed = self.compute_mean_scores()

        print("#" * 100)
        print("EVALUATION COMPLETE !!")
        print("# submission id: " + str(self.submission_id))
        print("#" * 100)
        print("# Mean Episode Score : {}".format(mean_score))
        print("# Sum of Episode Scores : {}".format(sum_score))
        print("# Mean Percentage Cargo Missed over all episodes: {}".format(100*mean_cargo_percentage_missed))
        print("# Mean Normalized Score over all episodes: {}".format(mean_normalized_score))
        print("# Num of Test Folders Completed: {}".format(self.num_tests_completed))
        print("# *** Overall score (sum of Normalized Scores): {} ***".format(sum_normalized_score))
        print("#" * 100)
        print("#" * 100)

        #
        return None

    def print_stats(self):
        ######################################################################
        # Print Local Stats
        ######################################################################
        print("=" * 100)
        print("=" * 100)
        print("## Performance Stats")
        print("=" * 100)
        print("\t - {}\t => min: {} || mean: {} || max: {}".format(
            "episode step time",
            self.episode_step_time_stat.min,
            self.episode_step_time_stat.mean,
            self.episode_step_time_stat.max))
        print("=" * 100)

    def compute_mean_scores(self):
        mean_score = np.array(self.simulation_scores).mean()
        sum_score = np.array(self.simulation_scores).sum()
        mean_normalized_score = np.array(self.simulation_scores_normalized).mean()
        mean_percentage_missed = np.array(self.simulation_missed_deliveries).mean()
        sum_normalized_score = np.array(self.simulation_scores_normalized).sum()

        # Round off the score values
        mean_score = round(mean_score, 2)
        mean_normalized_score = round(mean_normalized_score, 5)
        mean_percentage_missed = round(mean_percentage_missed, 3)

        file = open(self.output_dir + "results_summary_" + str(self.submission_id) + ".csv", "w")
        data = "Mean Evaluation Score, Sum Evaluation Score, Mean Evaluation Score Normalized, Mean Evaluation Percentage Missed Deliveries, Sum of Normalized Evaluation Score, Num Tests Completed \n"
        data += str(mean_score) + "," + str(sum_score) + "," + str(mean_normalized_score) + "," + str(
            mean_percentage_missed) + "," + str(sum_normalized_score) + "," + str(self.num_tests_completed)
        file.write(data)
        file.close()

        return mean_score, sum_score, mean_normalized_score, sum_normalized_score, mean_percentage_missed

    def create_scores_file(self):
        mean_score = np.array(self.simulation_scores).mean()
        sum_score = np.array(self.simulation_scores).sum()
        mean_normalized_score = np.array(self.simulation_scores_normalized).mean()
        mean_percentage_missed = np.array(self.simulation_missed_deliveries).mean()
        sum_normalized_score = np.array(self.simulation_scores_normalized).sum()

        scores_file = open(self.output_dir + "scores.txt", "w")
        scores_file.write("MEAN_Score: {:.3f}\n".format(mean_score))
        scores_file.write("SUM_Score: {:.3f}\n".format(sum_score))
        scores_file.write("MEAN_Normalized_Score: {:.3f}\n".format(mean_normalized_score))
        scores_file.write("MEAN_Percent_Missed: {:.3f}\n".format(mean_percentage_missed))
        scores_file.write("set1_score: {:.3f}\n".format(sum_normalized_score)) # SUM Normalized Score
        scores_file.write("Duration: {:.3f}\n".format(0))
        scores_file.close()
