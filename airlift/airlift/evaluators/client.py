import hashlib
import json
import logging
import os
import random
import time
from networkx.readwrite import json_graph
import msgpack
from airlift.evaluators.local_evaluator import Status
import msgpack_numpy as m
import pickle
import numpy as np
import redis
from airlift.envs.airlift_env import PlaneTypeObservation, ScenarioObservation
from airlift.envs.airlift_env import CargoObservation
from airlift.envs.airlift_env import PlaneState

from airlift.evaluators import messages
from airlift.solutions.baselines import RandomAgent


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
m.patch()


class TimeoutException(StopAsyncIteration):
    """ Custom exception for evaluation timeouts. """
    pass


def unpack_named_tuples(obs):

    for i in range(len(obs['a_0']['globalstate']['plane_types'])):
        obs['a_0']['globalstate']['plane_types'][i] = PlaneTypeObservation \
            (id=obs['a_0']['globalstate']['plane_types'][i][0],
             max_weight=obs['a_0']['globalstate']['plane_types'][i][1])
    for i in range(len(obs['a_0']['globalstate']['active_cargo'])):
        obs['a_0']['globalstate']['active_cargo'][i] = CargoObservation \
            (id=obs['a_0']['globalstate']['active_cargo'][i][0],
             location=obs['a_0']['globalstate']['active_cargo'][i][1],
             destination=obs['a_0']['globalstate']['active_cargo'][i][2],
             weight=obs['a_0']['globalstate']['active_cargo'][i][3],
             earliest_pickup_time=obs['a_0']['globalstate']['active_cargo'][i][4],
             is_available=obs['a_0']['globalstate']['active_cargo'][i][5],
             soft_deadline=obs['a_0']['globalstate']['active_cargo'][i][6],
             hard_deadline=obs['a_0']['globalstate']['active_cargo'][i][7])
    for i in range(len(obs['a_0']['globalstate']['scenario_info'])):
        obs['a_0']['globalstate']['scenario_info'][i] = ScenarioObservation \
            (processing_time=obs['a_0']['globalstate']['scenario_info'][i][0])
    for i in range(len(obs['a_0']['globalstate']['event_new_cargo'])):
        obs['a_0']['globalstate']['event_new_cargo'][i] = CargoObservation \
            (id=obs['a_0']['globalstate']['event_new_cargo'][i][0],
             location=obs['a_0']['globalstate']['event_new_cargo'][i][1],
             destination=obs['a_0']['globalstate']['event_new_cargo'][i][2],
             weight=obs['a_0']['globalstate']['event_new_cargo'][i][3],
             earliest_pickup_time=obs['a_0']['globalstate']['event_new_cargo'][i][4],
             is_available=obs['a_0']['globalstate']['event_new_cargo'][i][5],
             soft_deadline=obs['a_0']['globalstate']['event_new_cargo'][i][6],
             hard_deadline=obs['a_0']['globalstate']['event_new_cargo'][i][7])
    for agent in obs:
        old_value = obs[agent]['state']
        obs[agent]['state'] = PlaneState(obs[agent]['state'])
        assert old_value == obs[agent]['state'].value

def unpack_observations(obs):
    unpack_named_tuples(obs)
    for i, agent_idx in enumerate(obs):
        if i == 0:
            first_agent_idx = agent_idx
            for item in obs[agent_idx]['globalstate']['route_map']:
                obs[agent_idx]['globalstate']['route_map'][item] = json_graph.node_link_graph(
                    obs[agent_idx]['globalstate']['route_map'][item])
        else:
            obs[agent_idx]['globalstate'] = obs[first_agent_idx]['globalstate']


def pack_actions(actions):
    if actions is not None:
        for agent_idx in actions:
            # adjust observation frozen sets to lists
            if actions[agent_idx] is not None:
                for item in actions[agent_idx]:
                    if isinstance(actions[agent_idx][item], frozenset):
                        actions[agent_idx][item] = list(actions[agent_idx][item])


class RemoteClient(object):
    """
        Redis client to interface with airlift-rl remote-evaluation-service
        The Docker container hosts a redis-server inside the container.
        This client connects to the same redis-server,
        and communicates with the service.
        The service eventually will reside outside the docker container,
        and will communicate
        with the client only via the redis-server of the docker container.
        On the instantiation of the docker container, one service will be
        instantiated parallely.
        The service will accepts commands at "`service_id`::commands"
        where `service_id` is either provided as an `env` variable or is
        instantiated to "airlift_rl_service_id"
    """

    def __init__(self,
                 remote_host='127.0.0.1',
                 remote_port=6379,
                 remote_db=0,
                 remote_password=None,
                 test_env_folder=None,
                 verbose=False,
                 use_pickle=False):
        self.use_pickle = use_pickle
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.remote_db = remote_db
        self.remote_password = remote_password
        self.redis_pool = redis.ConnectionPool(
            host=remote_host,
            port=remote_port,
            db=remote_db,
            password=remote_password)
        self.redis_conn = redis.Redis(connection_pool=self.redis_pool)
        self.test_envs_root = None
        self.namespace = "airlift-rl"
        self.service_id = -1  # not sure if we need this?
        self.command_channel = "{}::{}::commands".format(
            self.namespace,
            self.service_id
        )

        # for timeout messages sent out-of-band
        self.error_channel = "{}::{}::errors".format(
            self.namespace, self.service_id)

        if test_env_folder:
            self.test_envs_root = test_env_folder

        self.current_env_path = None

        self.verbose = verbose

        self.env = None
        self.ping_pong()

        self.env_step_times = []
        self.stats = {}

    def get_redis_connection(self):
        return self.redis_conn

    def _generate_response_channel(self):
        random_hash = hashlib.md5(
            "{}".format(
                random.randint(0, 10 ** 10)
            ).encode('utf-8')).hexdigest()
        response_channel = "{}::{}::response::{}".format(self.namespace,
                                                         self.service_id,
                                                         random_hash)
        return response_channel

    def _remote_request(self, _request, blocking=True):
        """
            request:
                -command_type
                -payload
                -response_channel
            response: (on response_channel)
                - RESULT
            * Send the payload on command_channel (self.namespace+"::command")
                ** redis-left-push (LPUSH)
            * Keep listening on response_channel (BLPOP)
        """

        assert isinstance(_request, dict)

        _request['response_channel'] = self._generate_response_channel()
        _request['timestamp'] = time.time()

        _redis = self.get_redis_connection()
        """
            The client always pushes in the left
            and the service always pushes in the right
        """
        if self.verbose:
            print("Request : ", _request)

        # check for errors (essentially just timeouts, for now.)
        error_bytes = _redis.rpop(self.error_channel)
        if error_bytes is not None:
            if self.use_pickle:
                error_dict = pickle.loads(error_bytes)
            else:
                error_dict = msgpack.unpackb(
                    error_bytes,
                    object_hook=m.decode,
                    strict_map_key=False,  # new for msgpack 1.0?
                    raw=False
                    # encoding="utf8"  # remove for msgpack 1.0
                )
            print("Error received: ", error_dict)
            raise TimeoutException(error_dict["type"])

        # Push request in command_channels
        # Note: The patched msgpack supports numpy arrays
        if self.use_pickle:
            payload = pickle.dumps(_request)
        else:
            payload = msgpack.packb(_request, default=m.encode, use_bin_type=True)
        _redis.lpush(self.command_channel, payload)

        if blocking:
            # Wait with a blocking pop for the response
            _response = _redis.blpop(_request['response_channel'])[1]
            if self.verbose:
                print("Response : ", _response)
            if self.use_pickle:
                _response = pickle.loads(_response)
            else:
                _response = msgpack.unpackb(
                    _response,
                    object_hook=m.decode,
                    strict_map_key=False,  # new for msgpack 1.0?
                    raw=False
                    # encoding="utf8"  # remove for msgpack 1.0
                )
            if _response['type'] == messages.AIRLIFT_RL.ERROR:
                raise Exception(str(_response["payload"]))
            else:
                return _response

    def ping_pong(self):
        """
            Official Handshake with the evaluation service
            Send a PING
            and wait for PONG
            If not PONG, raise error
        """
        print("Pinging")
        _request = {}
        _request['type'] = messages.AIRLIFT_RL.PING
        _request['payload'] = {
            "version": "1.0"
        }
        _response = self._remote_request(_request)
        if _response['type'] != messages.AIRLIFT_RL.PONG:
            raise Exception(
                "Unable to perform handshake with the evaluation service. \
                Expected PONG; received {}".format(json.dumps(_response)))
        else:
            print("received response")
            return True

    # def env_create(self, obs_builder_object):
    def env_create(self):
        """
            Create a local env and remote env on which the
            local agent can operate.
            The observation builder is only used in the local env
            and the remote env uses a DummyObservationBuilder
        """
        time_start = time.time()
        _request = {}
        _request['type'] = messages.AIRLIFT_RL.ENV_CREATE
        _request['payload'] = {}
        _response = self._remote_request(_request)
        observation = _response['payload']['observation']
        info = _response['payload']['info']
        status = _response['payload']['status']

        if not observation:
            # If the observation is False,
            # then the evaluations are complete
            # hence return false
            return observation, info, status

        local_observation = observation

        # We use the last_env_step_time as an approximate measure of the inference time
        self.last_env_step_time = time.time()
        unpack_observations(local_observation)
        return local_observation, info, status

    def env_step(self, action):
        """
            Respond with [observation, reward, done, info]
        """
        # We use the last_env_step_time as an approximate measure of the inference time
        approximate_inference_time = time.time() - self.last_env_step_time
        # self.update_running_stats("inference_time(approx)", approximate_inference_time)

        pack_actions(action)

        _request = {}
        _request['type'] = messages.AIRLIFT_RL.ENV_STEP
        _request['payload'] = {}
        _request['payload']['action'] = action
        _request['payload']['inference_time'] = approximate_inference_time

        # Relay the action in a non-blocking way to the server
        # so that it can start doing an env.step on it in ~ parallel
        # Note - this can throw a Timeout
        _response = self._remote_request(_request)
        local_observation = _response['payload']['observation']
        unpack_observations(local_observation)
        local_reward = _response['payload']['all_rewards']
        local_done = _response['payload']['done']
        local_info = _response['payload']['info']

        # We use the last_env_step_time as an approximate measure of the inference time
        self.last_env_step_time = time.time()

        return [local_observation, local_reward, local_done, local_info]

    def submit(self):
        _request = {}
        _request['type'] = messages.AIRLIFT_RL.ENV_SUBMIT
        _request['payload'] = {}
        _response = self._remote_request(_request)

        ######################################################################
        # Print Local Stats
        ######################################################################
        print("=" * 100)
        print("=" * 100)
        print("## Client Performance Stats")
        print("=" * 100)
        for _key in self.stats:
            if _key.endswith("_mean"):
                metric_name = _key.replace("_mean", "")
                mean_key = "{}_mean".format(metric_name)
                min_key = "{}_min".format(metric_name)
                max_key = "{}_max".format(metric_name)
                print("\t - {}\t => min: {} || mean: {} || max: {}".format(
                    metric_name,
                    self.stats[min_key],
                    self.stats[mean_key],
                    self.stats[max_key]))
        print("=" * 100)
        return _response['payload']

    # evaluator will print stats, this function is just an additional
    # hack so the doeval function doesn't crash
    def print_stats(self):
        print("Evaluation Finished")


if __name__ == "__main__":
    evaluator = RemoteClient()
    solution = RandomAgent()
    solution_seed = 123
    episode = 0
    observation = True
    while observation:
        observation, infos, status = evaluator.env_create()
        dones = None
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

        while True:
            action = solution.policies(observation, dones, infos=infos)

            observation, all_rewards, dones, infos = evaluator.env_step(action)
            if all(dones.values()):
                print("Episode Done")
                break

        solution_seed += 1

    print("Evaluation Complete...")
    print(evaluator.submit())
