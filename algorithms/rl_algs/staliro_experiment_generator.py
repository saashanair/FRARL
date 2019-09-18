# start matlab engine and adjust paths
import os
import time
import queue
import random
import numpy as np
import matlab.engine as me
import multiprocessing as mp

mapping_staliro_exp = {}
DEBUG = False

def get_matlab_engine():

    eng = me.start_matlab("-nodesktop -nodisplay")
    try:
        eng.warning('off')
    except:
        eng.warning('off')
    pwd = os.getcwd()
    substr = "frarl"
    basepath = pwd[:pwd.find(substr) + len(substr)]
    eng.addpath(eng.genpath(r'{}/staliro_imports'.format(basepath)))
    eng.addpath(eng.genpath(r'{}/staliro_imports/trunk'.format(basepath)))
    eng.addpath(eng.genpath(r'{}/staliro_imports/matlab_bgl'.format(basepath)))
    eng.addpath(eng.genpath(r'{}/staliro_imports/Core_py2matlab'.format(basepath)))

    return eng

def job(staliro_seed, q):
    print("Start S-Taliro with seed {}".format(staliro_seed))
    eng = get_matlab_engine()
    try:
        staliro_obstacle_x_value, staliro_robustness_value = eng.car_sim_falsification(staliro_seed, nargout=2)
        q.put([np.array(staliro_obstacle_x_value), np.array(staliro_robustness_value)])
    except Exception as e:
        q.put(e)

def call_staliro_repeat_parallel(update):
    iter = 0
    max_iter = 10
    staliro_traces = []
    staliro_robustness_values = []
    while iter < max_iter:
        print("Not falsified, rerun {}/{}".format(iter+1, max_iter))
        staliro_traces, staliro_robustness_values = call_staliro_parallel(update+iter)
        if len(staliro_robustness_values) != 0:
            break
        iter += 1
    if  len(staliro_robustness_values) == 0:
        raise StopIteration("<Warning> Reached maximal Staliro runs! Still not falsified!")
    else:
        if DEBUG:
            print("<S-Taliro parallel repeat> staliro_traces.shape={}".format(staliro_traces.shape))
            print("<S-Taliro parallel repeat> staliro_robustness_values.shape={}".format(staliro_robustness_values.shape))
    np.save("staliro_results_{}.npy".format(update), staliro_traces)
    return staliro_traces, staliro_robustness_values


def call_staliro_parallel(update):
    """
    Call S-Taliro parallely in num_processes matlab engines and return results with lowest robustness
    """
    print("Calling S-Taliro at {} iteration".format(update))
    num_processes = 10
    timeout = 60
    # staliro_seeds = [update*10 + x for x in range(num_processes)]
    tmp = random.randint(1, 39)
    staliro_seeds = [tmp*10 + x for x in range(num_processes)]    
    staliro_traces = np.array([])
    staliro_robustnesses = np.array([])

    # create subprocesses
    q = mp.Manager().Queue(maxsize=num_processes)
    ps = [mp.Process(target=job, args=(staliro_seed, q)) for staliro_seed in staliro_seeds]

    # start subprocesses
    if DEBUG:
        print("<S-Taliro parallel>: Starting processes...")
    for p in ps:
        p.daemon = True
        p.start()
    time.sleep(20)
    # wait for all subprocesses to finish; timeout=15 to avoid process hanging
    if DEBUG:
        print("<S-Taliro parallel>: Finishing processes...")
    for p in ps:
        p.join(timeout=timeout)
    if DEBUG:
        print("<S-Taliro parallel>: Timeout!")
    # kill hanging processes
    for i, p in enumerate(ps):
        if p.is_alive():
            if DEBUG:
                print("<S-Taliro parallel>: Shutting down process {}".format(i+1))
            p.terminate()
            p.join(timeout=5)
    if DEBUG:
        print("<S-Taliro parallel>: Getting results {}...".format(q.qsize()))
    # get all results

    for j, p in enumerate(ps):
        try:
            if DEBUG:
                print("<S-Taliro parallel>: {}/{}".format(j+1, len(ps)))
            staliro_results = q.get_nowait()            
        except:
            continue
        # if matlab not crashed, get staliro results
        if isinstance(staliro_results, me.EngineError):
            print("<S-Taliro parallel>: Matlab Engine Error!")
        elif staliro_results[0].size != 0:
            trace = staliro_results[0]
            robustness = np.reshape(staliro_results[1], (1, 1)) if staliro_results[1].size == 1 else staliro_results[1]
            num_falsified = robustness.size
            if DEBUG:
                print("<S-Taliro parallel>: Num_falsified {}...".format(num_falsified))
                print("<S-Taliro parallel>: staliro shape {}".format(trace.shape))
                print("<S-Taliro parallel>: staliro shape {}".format(robustness.shape))
            # for i in range(num_falsified):
            staliro_traces = np.concatenate((staliro_traces, trace), axis=1) if staliro_traces.size else trace
            staliro_robustnesses = np.concatenate((staliro_robustnesses, robustness), axis=1) if staliro_robustnesses.size else robustness
                # staliro_traces.append(staliro_obstacle_x_value[:, i])
                # staliro_robustness_values.append(staliro_robustness_value[:, i])
            if DEBUG:
                print("<S-Taliro parallel>: staliro_traces shape {}".format(staliro_traces.shape))

    # get result with minimal robustness value
    if len(staliro_traces) == 0: # all matlab processes crashed!
        raise me.EngineError("<S-Taliro parallel>: All matlab processes crashed!")
    else:
        return staliro_traces, staliro_robustnesses

def call_staliro_multiple(update):
    """
    Call S-Taliro multiple times until falsified or max_iter reached
    """
    eng = get_matlab_engine()

    staliro_seed = update
    print("Calling S-Taliro at {} iteration".format(staliro_seed))
    staliro_robustness_value = 100.
    max_iter = 10
    iter = 0
    staliro_traces = []
    staliro_robustness_values = []

    while staliro_robustness_value > 0. and iter < max_iter:
        print("Calling S-Taliro {}/{} times".format(iter+1, max_iter))        
        staliro_obstacle_x_value, staliro_robustness_value = eng.car_sim_falsification(staliro_seed, nargout=2)
        staliro_traces.append(staliro_obstacle_x_value)
        staliro_robustness_values.append(staliro_robustness_value)
        staliro_seed += 1
        iter += 1

    if staliro_robustness_value > 0.:
        index = np.argmin(staliro_robustness_values)
        staliro_obstacle_x_value = staliro_traces[index]
        staliro_robustness_value = staliro_robustness_values[index]
        print("<Warning> Reached maximal Staliro runs! Taking minimal robustness sample r = {}...".format(\
            staliro_robustness_value))

    staliro_obstacle_x_value = np.asarray(staliro_obstacle_x_value)
    print("Finished S-Taliro...:")
    np.save("staliro_results_{}.npy".format(update), staliro_obstacle_x_value)
    return staliro_obstacle_x_value, staliro_robustness_value

def register(name):
    def _thunk(func):
        mapping_staliro_exp[name] = func
        return func
    return _thunk

@register("no_staliro")
def no_staliro(update, obs_x_queue):
    return None, None

@register("fixed_staliro")
def fixed_staliro(update, obs_x_queue):
    # if update > 1:
    if update % 10 == 1 and update > 1:
        obs_x_queue.clear()        
        # TODO: parallize staliro (running staliro takes a long time, setting opt.n_workers = sth>1 gives an error)
        staliro_obstacle_x_value, staliro_robustness_value = call_staliro_repeat_parallel(update)        
        obs_x_queue.append(staliro_obstacle_x_value[:, 0])
        return obs_x_queue, staliro_robustness_value
    return None, None

@register("randomized_staliro")
def randomized_staliro(update, obs_x_queue):
    if update % 10 == 1 and update > 1: #update % 10 == 1 and update > 1
        # TODO: parallize staliro (running staliro takes a long time, setting opt.n_workers = sth>1 gives an error)
        staliro_obstacle_x_value, staliro_robustness_value = call_staliro_repeat_parallel(update)        
        obs_x_queue.append(staliro_obstacle_x_value)
        return obs_x_queue, staliro_robustness_value
    return None, None

@register("weighted_queue")
def weighted_queue(update, obs_x_queue):
    # The difference between weighted_queue and randomized_staliro
    # is only in the way that obs_x value is picked in reset
    return randomized_staliro(update, obs_x_queue)

@register("success_counter")
def success_counter(data, obs_x_queue):
    """
    :param data: [0]: number of runs that reach the maximal reward in the last 20 runs; [1] update
    :param obs_x_queue:
    :return:
    """
    # TODO: specify x for success_counter
    if data[0] == 20:
        obs_x_queue.clear()
        staliro_obstacle_x_value, staliro_robustness_value = call_staliro_repeat_parallel(data[1])
        obs_x_queue.append(staliro_obstacle_x_value)
        return obs_x_queue, staliro_robustness_value
    return None, None

@register("variable_start")
def variable_start(data, obs_x_queue):
    """

    :param data: [0]: if number of collisions >= no. of offroad in the last 10 runs; [1] update
    :param obs_x_queue:
    :return:
    """
    print("VARIABLE START: ", data)
    # TODO: Figure out a way to get details about the cause of end of an episode,
    #  in terms of off-road, collision or success
    # Where is information being added infos (runner.py, line 50), other that l, r and t, can we add more information to it?
    if type(data) is list and data[0] and int(data[1]) % 10 == 1:
        obs_x_queue.clear()
        staliro_obstacle_x_value, staliro_robustness_value = call_staliro_repeat_parallel(data[1])
        obs_x_queue.append(staliro_obstacle_x_value)
        return obs_x_queue, staliro_robustness_value
    return None, None

@register("debug_staliro")
def debug_staliro(update, obs_x_queue):
    if update % 10 == 1:
        obs_x_queue.clear()
        # TODO: parallize staliro (running staliro takes a long time, setting opt.n_workers = sth>1 gives an error)
        staliro_path = "/home/xiao/projects/mlmeetsformalmethods/Implementation/openai/logs/CarSim-Python-v1-ppo2_rlf/debug"
        staliro_obstacle_x_value = np.load(os.path.join(staliro_path, "staliro_results_171.npy"))
        staliro_robustness_value = -1. # dummy value
        obs_x_queue.append(staliro_obstacle_x_value)
        return obs_x_queue, staliro_robustness_value
    return None, None

@register("multiple_staliro")
def multiple_staliro(update, obs_x_queue):
    # if update > 1:
    if update % 10 == 1 and update > 100: # call staliro after 101 updates
        staliro_obstacle_x_value, staliro_robustness_value = call_staliro_repeat_parallel(update)        
        for i in range(staliro_obstacle_x_value.shape[1]):
            obs_x_queue.append(staliro_obstacle_x_value[:, i])
        print("<multiple staliro>: staliro queue length {}".format(len(obs_x_queue)))
        return obs_x_queue, staliro_robustness_value
    return None, None

def get_staliro_experiment_builder(name):
    if name in mapping_staliro_exp:
        return mapping_staliro_exp[name]
"""
from collections import deque
a = deque([10, 20, 30, 40, 50], maxlen=5)
print(get_exp_builder("no_staliro")(10, a))
print(get_exp_builder("randomized_staliro")(11, a))
print(get_exp_builder("fixed_staliro")(21, a))
"""         
