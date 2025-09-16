import csv
import os
import traceback
import warnings
from operator import itemgetter
from typing import NamedTuple, Callable, List
from pathlib import Path

from airlift.evaluators.local_evaluator import Status
from airlift.evaluators.local_evaluator import LocalEvaluationService
from airlift.solutions import Solution
import multiprocessing

from airlift.envs import AirliftEnv
from airlift.solutions import doepisode
from airlift.solutions.baselines import RandomAgent, ShortestPath
from airlift.evaluators.client import RemoteClient


def doeval(test_folder: Path,
           solution: Solution,
           start_solution_seed: int = 123):
    evaluator = LocalEvaluationService(test_env_folder=str(test_folder))

    solution_seed = start_solution_seed
    episode = 0
    observation = True
    while observation:
        observation, infos, status = evaluator.env_create()
        dones = evaluator._env.dones
        if status == Status.FINISHED_ALL_SCENARIOS:
            # When evaluation is complete, the evaluator responds false for the observation
            print("Evaluation Complete")
            break
        elif status == Status.STOPPED_TOO_MANY_MISSED:
            print("Evaluation Ended due to too many missed cargo.")
            break
        print("Episode : {}".format(episode))

        solution.reset(observation, seed=solution_seed)

        while True:
            action = solution.policies(observation, dones, infos=infos)

            observation, all_rewards, dones, infos = evaluator.env_step(action)
            if all(dones.values()):
                print("Episode {} Done".format(episode))
                episode += 1
                break

        solution_seed += 1

    # if status == Status.FINISHED_ALL_SCENARIOS:
    print("Evaluation Complete...")
    print(evaluator.submit())
    evaluator.print_stats()

def doremoteeval(test_folder: Path,
           solution: Solution,
           start_solution_seed: int=123,
           remote_port: int=6379):
    evaluator = RemoteClient(test_env_folder=str(test_folder),
                                     remote_port=remote_port)

    solution_seed = start_solution_seed
    episode = 0
    observation = True
    while observation:
        observation, infos, status = evaluator.env_create()
        if status == Status.FINISHED_ALL_SCENARIOS:
            # When evaluation is complete, the evaluator responds false for the observation
            print("Evaluation Complete")
            break
        elif status == Status.STOPPED_TOO_MANY_MISSED:
            print("Evaluation Ended due to too many missed cargo.")
            break
        print("Episode : {}".format(episode))
        episode += 1

        solution.reset(observation, seed=solution_seed)

        dones = {a: False for a in observation.keys()} # For the first step assume agents are not done
        while True:
            action = solution.policies(observation, dones, infos=infos)

            observation, all_rewards, dones, infos = evaluator.env_step(action)
            if all(dones.values()):
                print("Episode {} Done".format(episode))
                break

        solution_seed += 1

    #if status == Status.FINISHED_ALL_SCENARIOS:
    print("Evaluation Complete...")
    print(evaluator.submit())
    evaluator.print_stats()

def doeval_single_episode(
           test_pkl_file: Path,
           env_seed,
           solution: Solution,
           solution_seed: int = 123,
           render=False,
           render_sleep_time=0.1,
           render_mode="human",
           capture_step_metrics=False):
    env = AirliftEnv.load(test_pkl_file)
    return \
        doepisode(env,
                  solution=solution,
                  render=render,
                  env_seed=env_seed,
                  solution_seed=solution_seed,
                  render_sleep_time=render_sleep_time,
                  capture_metrics=capture_step_metrics,
                  render_mode=render_mode)



class ScenarioInfo(NamedTuple):
    testnum: int
    levelnum: int
    env: AirliftEnv


def _get_solution_info(solution_env, solution_name, solution_seed):
    return ["{}_solution_seed".format(solution_name),
            "{}_score".format(solution_name),
            "{}_missed_deliveries".format(solution_name),
            "{}_total_lateness".format(solution_name),
            "{}_total_cost".format(solution_name),
            "{}_total_steps".format(solution_name),
            "{}_total_cargo_generated".format(solution_name),
            "{}_total_scaled_cost".format(solution_name),
            "{}_total_scaled_lateness".format(solution_name)
            ], \
           [solution_seed,
            solution_env.metrics.score,
            solution_env.metrics.missed_deliveries,
            solution_env.metrics.total_lateness,
            solution_env.metrics.total_cost,
            solution_env.metrics.total_steps,
            solution_env.metrics.total_cargo_generated,
            solution_env.metrics.total_scaled_cost,
            solution_env.metrics.total_scaled_lateness]


def generate_scenarios(output_path, scenarios: List[ScenarioInfo], multiprocess=False,
                       num_processes=multiprocessing.cpu_count() - 1, run_random=True, run_baseline=True,
                       base_env_seed=44, base_solution_seed=33):
    output_path.mkdir(parents=True, exist_ok=True)

    if list(output_path.glob('*')):
        warnings.warn("Output folder is not empty")

    base_env_seed = base_env_seed
    base_solution_seed = base_solution_seed

    if multiprocess:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=num_processes) \
            (delayed(generate_scenario)(scenario, output_path, base_env_seed + i, base_solution_seed + i,
                                        run_random=run_random, run_baseline=run_baseline)
             for i, scenario in enumerate(scenarios))
        headers, raw_rows = zip(*results)

        # Filter out "None" rows, i.e., rows which encountered an exception
        rows = []
        for i, row in enumerate(raw_rows):
            if row is None:
                warnings.warn("Missing results for row {0}".format(i))
            else:
                rows.append(row)

        # Make sure metadata is properly sorted.
        # Assume test and level are first two fields!
        rows = sorted(rows, key=itemgetter(0, 1))

        print("Writing metadata")
        with open(output_path / "metadata.csv", 'w', newline='') as metadatafile:
            csvwriter = csv.writer(metadatafile)

            assert headers[0] is not None  # This could happen if first scenario generator fails
            csvwriter.writerow(headers[0])
            for row in rows:
                csvwriter.writerow(row)
    else:
        print("Writing metadata")

        # Writes the metadata as the scenarios are generated_debug, so that we can stop the script early and have usable partial results
        with open(output_path / "metadata.csv", 'w', newline='') as metadatafile:
            csvwriter = csv.writer(metadatafile)
            for i, scenario in enumerate(scenarios):
                header, row = generate_scenario(scenario, output_path, base_env_seed + i, base_solution_seed + i,
                                                run_random=run_random, run_baseline=run_baseline)
                if i == 0:
                    assert header is not None  # This could happen if first scenario generator fails
                    csvwriter.writerow(header)
                if row is None:
                    warnings.warn("Missing results for row {0}".format(i))
                else:
                    csvwriter.writerow(row)

    print("Done!")


def generate_scenario(scenario, output_path, env_seed, solution_seed, run_random=True, run_baseline=True):
    try:
        testnum = scenario.testnum
        levelnum = scenario.levelnum
        env = scenario.env

        test_folder = "Test_{}".format(testnum)
        level_filename = "Level_{}.pkl".format(levelnum)
        test_path = output_path / test_folder
        test_path.mkdir(exist_ok=True)
        level_file = test_path / level_filename

        print("Start Generating {}".format(level_file))
        env.save(level_file)

        env.reset(env_seed)
        env_info = env.env_info

        # # For testing purposes
        # if scenario.testnum > 0:
        #     raise Exception("Testing failure")

        # Generate random score
        # Let's reload the environment to make sure it's exactly what the evaluator will evaluate against
        if run_random:
            randomenv = AirliftEnv.load(level_file)
            doepisode(randomenv,
                      solution=RandomAgent(),
                      render=False,
                      env_seed=env_seed,
                      solution_seed=solution_seed)
            assert (all(randomenv.dones.values()))
            random_fields, random_values = _get_solution_info(randomenv, "random", solution_seed)
        else:
            random_fields = []
            random_values = []

        # Generate baseline score
        if run_baseline:
            baselineenv = AirliftEnv.load(level_file)
            doepisode(baselineenv,
                      solution=ShortestPath(),
                      render=False,
                      env_seed=env_seed,
                      solution_seed=solution_seed)
            assert (all(baselineenv.dones.values()))
            baseline_fields, baseline_values = _get_solution_info(baselineenv, "baseline", solution_seed)
        else:
            baseline_fields = []
            baseline_values = []

        print("Done Generating {}".format(level_file))

        if run_baseline and randomenv.metrics.score <= baselineenv.metrics.score:
             warnings.warn("Random solution did as well as or better than baseline solutions - omitting from metadata")
             os.remove(level_file)
             return None, None

        if True:
            header = ["test_id",
                      "level_id",
                      "filename",
                      "seed"] \
                     + list(env_info._fields) \
                     + list(random_fields) \
                     + list(baseline_fields)
            data = [scenario.testnum,
                    scenario.levelnum,
                    test_folder + "/" + level_filename,
                    env_seed] \
                   + list(env_info) \
                   + list(random_values) \
                   + list(baseline_values)

            return header, data
    except:
        warnings.warn("Exception processing test {0} level {1}".format(scenario.testnum, scenario.levelnum))
        traceback.print_exc()
        return None, None
