#!/usr/bin/env python
from __future__ import print_function

import os
import traceback
from networkx.readwrite import json_graph
import networkx

import msgpack
import msgpack_numpy as m
import pickle
import redis

import airlift

from airlift.evaluators import messages
from airlift.evaluators.local_evaluator import LocalEvaluationService

use_signals_in_timeout = True
if os.name == 'nt':
    """
    Windows doesnt support signals, hence
    timeout_decorators usually fall apart.
    Hence forcing them to not using signals
    whenever using the timeout decorator.
    """
    use_signals_in_timeout = False

m.patch()

########################################################
# CONSTANTS
########################################################

# Don't proceed to next Test if the previous one didn't reach this mean completion percentage
TEST_MIN_PERCENTAGE_COMPLETE_MEAN = float(os.getenv("TEST_MIN_PERCENTAGE_COMPLETE_MEAN", 0.25))


SUPPORTED_CLIENT_VERSIONS = \
    [
        airlift.__version__
    ]


def pack_observations(obs):
    for i, agent_idx in enumerate(obs):
        if i == 0:
            # adjust observation graph to json
            for graph_item in obs[agent_idx]['globalstate']['route_map'].keys():
                if isinstance(obs[agent_idx]['globalstate']['route_map'][graph_item], networkx.DiGraph):
                    obs[agent_idx]['globalstate']['route_map'][graph_item] = \
                        json_graph.node_link_data(obs[agent_idx]['globalstate']['route_map'][graph_item])
        else:
            # Remove global state to save space (only keep it for the 1st agent)
            obs[agent_idx]['globalstate'] = None


def unpack_actions(actions):
    pass


class RemoteEvaluationService:
    """
    The remote evaluation service designed to be used for the client for the airlift environment.
    """

    def __init__(self,
                 remote_host = '127.0.0.1',
                 remote_port = 6379,
                 remote_db = 0,
                 remote_password = None,
                 local_eval = None,
                 service_id = -1,
                 verbose = False,
                 report = False,
                 pickle = False):

        self.use_pickle = pickle
        self.local_eval = local_eval
        self.report = report
        self.service_id = service_id
        # Communication Protocol Related vars
        self.namespace = "airlift-rl"
        self.command_channel = "{}::{}::commands".format(
            self.namespace,
            self.service_id
        )
        self.error_channel = "{}::{}::errors".format(
            self.namespace,
            self.service_id
        )
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.remote_db = remote_db

        self.termination_cause = "Placeholder"
        self.verbose = verbose
        # Message Broker related vars
        self.remote_password = remote_password
        self.instantiate_redis_connection_pool()
        self.state_env_timed_out = False
        self.evaluation_done = False

        #evaluation specific vars

        self.evaluation_state = {
            "state": "PENDING",
            "progress": 0.0,
            "simulation_count": 0,
            "score": {
                "score": 0.0,
                "score_secondary": 0.0
            },
            "meta": {
                "normalized_reward": 0.0
            }
        }
        self.stats = {}
        self.previous_command = {
            "type": None
        }

    def instantiate_redis_connection_pool(self):
        """
        Instantiates a Redis connection pool which can be used to
        communicate with the message broker
        """
        if self.verbose or self.report:
            print("Attempting to connect to redis server at {}:{}/{}".format(
                self.remote_host,
                self.remote_port,
                self.remote_db))

        self.redis_pool = redis.ConnectionPool(
            host=self.remote_host,
            port=self.remote_port,
            db=self.remote_db,
            password=self.remote_password
        )
        self.redis_conn = redis.Redis(connection_pool=self.redis_pool)

    def get_redis_connection(self):
        """
        Obtains a new redis connection from a previously instantiated
        redis connection pool
        """
        return self.redis_conn

    def _error_template(self, payload):
        """
        Simple helper function to pass a payload as a part of a
        airlift comms error template.
        """
        _response = {}
        _response['type'] = messages.AIRLIFT_RL.ERROR
        _response['payload'] = payload
        return _response

    def get_next_command(self):

        def _get_next_command(command_channel, _redis):
            """
            A low level wrapper for obtaining the next command from a
            pre-agreed command channel.
            At the momment, the communication protocol uses lpush for pushing
            in commands, and brpop for reading out commands.
            """
            command = _redis.brpop(command_channel)[1]
            return command

        if True:
            _redis = self.get_redis_connection()
            command = _get_next_command(self.command_channel, _redis)
            if self.verbose or self.report:
                print("Command Service: ", command)

        if self.use_pickle:
            command = pickle.loads(command)
        else:
            command = msgpack.unpackb(
                command,
                object_hook=m.decode,
                strict_map_key=False,  # msgpack 1.0
                raw=False#"utf8"  # msgpack 1.0
            )
        if self.verbose:
            print("Received Request : ", command)

        return command

    def send_response(self, _command_response, command, suppress_logs=False):
        _redis = self.get_redis_connection()
        command_response_channel = command['response_channel']

        if self.verbose and not suppress_logs:
            print("Responding with : ", _command_response)

        if self.use_pickle:
            sResponse = pickle.dumps(_command_response)
        else:
            sResponse = msgpack.packb(
                _command_response,
                default=m.encode,
                use_bin_type=True)
        _redis.rpush(command_response_channel, sResponse)

    def send_error(self, error_dict, suppress_logs=False):
        """ For out-of-band errors like timeouts,
            where we do not have a command, so we have no response channel!
        """
        _redis = self.get_redis_connection()
        print("Sending error : ", error_dict)

        if self.use_pickle:
            sResponse = pickle.dumps(error_dict)
        else:
            sResponse = msgpack.packb(
                error_dict,
                default=m.encode,
                use_bin_type=True)

        _redis.rpush(self.error_channel, sResponse)

    def handle_ping(self, command):
        """
        Handles PING command from the client.
        """
        service_version = airlift.__version__
        if "version" in command["payload"].keys():
            client_version = command["payload"]["version"]
        else:
            # 2.1.4 -> when the version mismatch check was added
            client_version = "2.1.4"

        _command_response = {}
        _command_response['type'] = messages.AIRLIFT_RL.PONG
        _command_response['payload'] = {}
        if client_version not in SUPPORTED_CLIENT_VERSIONS:
            _command_response['type'] = messages.AIRLIFT_RL.ERROR
            _command_response['payload']['message'] = \
                "Client-Server Version Mismatch => " + \
                "[ Client Version : {} ] ".format(client_version) + \
                "[ Server Version : {} ] ".format(service_version)
            self.send_response(_command_response, command)
            raise Exception(_command_response['payload']['message'])

        print("Received ping from client")

        self.send_response(_command_response, command)

    def handle_env_create(self, command):
        """
        Handles a ENV_CREATE command from the client
        """

        observation, info, status = self.local_eval.env_create()
        if (observation != False):
            pack_observations(observation)

        #print(" -- [DEBUG] [env_create] return obs = " + str(observation) + "(END)")
        """
        All test env evaluations are complete
        """
        _command_response = {}
        _command_response['type'] = messages.AIRLIFT_RL.ENV_CREATE_RESPONSE
        _command_response['payload'] = {}
        _command_response['payload']['observation'] = observation
        _command_response['payload']['info'] = info
        _command_response['payload']['status'] = status
        print("STATUS IS: " + str(status))

        self.send_response(_command_response, command)

        return observation, info, status

    def handle_env_step(self, command):
        """
        Handles a ENV_STEP command from the client
        TODO: Add a high level summary of everything thats happening here.
        """

        if self.state_env_timed_out or self.evaluation_done:
            print("Ignoring step command after timeout.")
            return

        _payload = command['payload']

        if self.local_eval.get_done_status():
            raise Exception(
                "Client attempted to perform an action on an Env which \
                has done['__all__']==True")

        action = _payload['action']

        _observation, all_rewards, done, info = self.local_eval.env_step(action)
        pack_observations(_observation)

        _command_response = {}
        _command_response['type'] = messages.AIRLIFT_RL.ENV_STEP_RESPONSE
        _command_response['payload'] = {}
        _command_response['payload']['observation'] = _observation
        _command_response['payload']['all_rewards'] = all_rewards
        _command_response['payload']['done'] = done
        _command_response['payload']['info'] = info

        self.send_response(_command_response, command)

        # Is the episode over?
        if self.local_eval.get_done_status():
            self.simulation_done = True

    def handle_env_submit(self, command):
        """
        Handles a ENV_SUBMIT command from the client
        TODO: Add a high level summary of everything thats happening here.
        """
        _payload = command['payload']

        mean_score, sum_score, mean_normalized_score, sum_normalized_score, mean_cargo_percentage_missed = \
            self.local_eval.compute_mean_scores()

        _command_response = {}
        _command_response['type'] = messages.AIRLIFT_RL.ENV_SUBMIT_RESPONSE
        _payload = {}
        _payload['overall_score'] = sum_normalized_score
        _payload['mean_fraction_complete'] = mean_cargo_percentage_missed
        _command_response['payload'] = _payload
        self.send_response(_command_response, command)

        print("#" * 100)
        print("EVALUATION COMPLETE !!")
        print("#" * 100)
        print("# Overall Score: {} (Sum Normalized Reward)".format(sum_normalized_score))
        print("# Mean Percentage Complete : {}".format(mean_cargo_percentage_missed))
        # print("# Mean Reward : {}".format(mean_score))
        print("# Mean Normalized Score : {}".format(mean_normalized_score))
        print("#" * 100)
        print("#" * 100)
        exit(0)

    def report_error(self, error_message, command_response_channel):
        """
        A helper function used to report error back to the client
        """
        _redis = self.get_redis_connection()
        _command_response = {}
        _command_response['type'] = messages.AIRLIFT_RL.ERROR
        _command_response['payload'] = error_message

        if self.use_pickle:
            bytes_error = pickle.dumps(_command_response)
        else:
            bytes_error = msgpack.packb(
                _command_response,
                default=m.encode,
                use_bin_type=True)

        _redis.rpush(command_response_channel, bytes_error)

        self.evaluation_state["state"] = "ERROR"
        self.evaluation_state["error"] = error_message
        self.evaluation_state["meta"]["termination_cause"] = "An error occured."

    def run(self):
        """
        Main runner function which waits for commands from the client
        and acts accordingly.
        """
        print("Listening at : ", self.command_channel)
        MESSAGE_QUEUE_LATENCY = []

        while True:
            command = self.get_next_command()

            self.timeout_counter = 0

            try:
                if command['type'] == messages.AIRLIFT_RL.PING:
                    """
                        INITIAL HANDSHAKE : Respond with PONG
                    """
                    self.handle_ping(command)

                elif command['type'] == messages.AIRLIFT_RL.ENV_CREATE:
                    """
                        ENV_CREATE

                        Respond with an internal _env object
                    """
                    self.handle_env_create(command)
                elif command['type'] == messages.AIRLIFT_RL.ENV_STEP:
                    """
                        ENV_STEP

                        Request : Action dict
                        Respond with updated [observation,reward,done,info] after step
                    """
                    self.handle_env_step(command)
                elif command['type'] == messages.AIRLIFT_RL.ENV_SUBMIT:
                    """
                        ENV_SUBMIT

                        Submit the final cumulative reward
                    """

                    #print("Overall Message Queue Latency : ", np.array(MESSAGE_QUEUE_LATENCY).mean())

                    self.handle_env_submit(command)

                else:
                    _error = self._error_template(
                        "UNKNOWN_REQUEST:{}".format(
                            str(command)))
                    if self.verbose:
                        print("Responding with : ", _error)
                    if "response_channel" in command:
                        self.report_error(
                            _error,
                            command['response_channel'])
                    return _error
                ###########################################
                # We keep a record of the previous command
                # to be able to have different behaviors
                # between different "command transitions"
                #
                # An example use case, is when we want to
                # have a different timeout for the
                # first step in every environment
                # to account for some initial planning time
                self.previous_command = command
            except Exception as e:
                print("Error : ", str(e))
                print(traceback.format_exc())
                if ("response_channel" in command):
                    self.report_error(
                        self._error_template(str(e)),
                        command['response_channel'])
                return self._error_template(str(e))



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Submit the results')

    parser.add_argument('--service_id',
                        dest='service_id',
                        default='AIRLIFT_RL_SERVICE_ID',
                        required=False)

    parser.add_argument('--test_folder',
                        dest='test_folder',
                        default="/home/airlift/test_folder",
                        help="Folder containing the files for the test envs",
                        required=False)

    parser.add_argument('--mergeDir',
                        dest='mergeDir',
                        default=None,
                        help="Folder to store merged envs, actions, episodes.",
                        required=False)

    parser.add_argument('--pickle',
                        default=False,
                        action="store_true",
                        help="use pickle instead of msgpack",
                        required=False)

    parser.add_argument('--shuffle',
                        default=False,
                        action="store_true",
                        help="Shuffle the environments",
                        required=False)

    parser.add_argument('--disableTimeouts',
                        default=False,
                        action="store_true",
                        help="Disable all timeouts.",
                        required=False)

    parser.add_argument('--missingOnly',
                        default=False,
                        action="store_true",
                        help="only request the envs/actions which are missing",
                        required=False)

    parser.add_argument('--remote_port',
                        default="6379",
                        help="redis port",
                        required=False)

    parser.add_argument('--verbose',
                        default=False,
                        action="store_true",
                        help="verbose debug messages",
                        required=False)

    parser.add_argument('--output_dir',
                        dest='output_dir',
                        default="/home/airlift/results/",
                        required=False)

    parser.add_argument('--submission_id',
                        dest='submission_id',
                        default=-1,
                        required=False)

    parser.add_argument('--fail_threshold',
                        dest='fail_threshold',
                        default=0.3,
                        required=False)

    parser.add_argument('--first_step_timeout',
                        dest='first_step_timeout',
                        default=600,
                        required=False)

    parser.add_argument('--step_timeout',
                        dest='step_timeout',
                        default=60,
                        required=False)

    parser.add_argument('--render',
                        default=False,
                        action="store_true",
                        required=False)

    args = parser.parse_args()

    test_folder = args.test_folder

    local_eval = LocalEvaluationService(test_env_folder=args.test_folder,
                                        output_dir =args.output_dir,
                                        submission_id=args.submission_id,
                                        fail_threshold=args.fail_threshold,
                                        first_step_timeout_in_seconds=args.first_step_timeout,
                                        step_timeout_in_seconds=args.step_timeout,
                                        render=args.render
                                        )
    grader = RemoteEvaluationService(
        remote_port=args.remote_port,
        verbose=args.verbose,
        local_eval=local_eval
    )

    result = grader.run()
    if result['type'] == messages.AIRLIFT_RL.ENV_SUBMIT_RESPONSE:
        cumulative_results = result['payload']
    elif result['type'] == messages.AIRLIFT_RL.ERROR:
        error = result['payload']
        raise Exception("Evaluation Failed : {}".format(str(error)))
    else:
        # Evaluation failed
        print("Evaluation Failed : ", result['payload'])


