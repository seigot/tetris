#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import subprocess
from argparse import ArgumentParser

def get_option(game_level, game_time, manual, use_sample, random_seed, resultlogjson, user_name):
    argparser = ArgumentParser()
    argparser.add_argument('-l', '--game_level', type=int,
                           default=game_level,
                           help='Specify game level')
    argparser.add_argument('-t', '--game_time', type=int,
                           default=game_time,
                           help='Specify game time(s)')
    argparser.add_argument('-m', '--manual',
                           default=manual,
                           help='Specify if manual control')
    argparser.add_argument('-s', '--use_sample',
                           default=use_sample,
                           help='Specify if use sample')
    argparser.add_argument('-r', '--random_seed', type=int,
                           default=random_seed,
                           help='Specify random seed if necessary') 
    argparser.add_argument('-f', '--resultlogjson', type=str,
                           default=resultlogjson,
                           help='Specigy result log file path if necessary')
    argparser.add_argument('-u', '--user_name', type=str,
                           default=user_name,
                           help='Specigy user name if necessary')
    return argparser.parse_args()

def get_python_cmd():
    ret = subprocess.run("python --version", shell=True, \
                         stderr=subprocess.PIPE, encoding="utf-8")
    print(ret)
    if "Python 2" in ret.stderr:
        return "python3"
    return "python"

def start():
    ## default value
    GAME_LEVEL = 1
    GAME_TIME = 180
    IS_MANUAL_CONTROLL = "n"
    IS_SAMPLE_CONTROLL = "n"
    INPUT_RANDOM_SEED = -1
    RESULT_LOG_JSON = "result.json"
    USER_NAME = "user_name"

    ## update value if args are given
    args = get_option(GAME_LEVEL,
                      GAME_TIME,
                      IS_MANUAL_CONTROLL,
                      IS_SAMPLE_CONTROLL,
                      INPUT_RANDOM_SEED,
                      RESULT_LOG_JSON,
                      USER_NAME)
    if args.game_level >= 0:
        GAME_LEVEL = args.game_level
    if args.game_time >= 0:
        GAME_TIME = args.game_time
    if args.manual in ("y", "g"):
        IS_MANUAL_CONTROLL = args.manual
    if args.use_sample == "y":
        IS_SAMPLE_CONTROLL = args.use_sample
    if args.random_seed >= 0:
        INPUT_RANDOM_SEED = args.random_seed
    if len(args.resultlogjson) != 0:
        RESULT_LOG_JSON = args.resultlogjson
    if len(args.user_name) != 0:
        USER_NAME = args.user_name

    ## set field parameter for level 1
    DROP_SPEED = 1000          # drop speed (s)
    RANDOM_SEED = 0            # random seed for field
    OBSTACLE_HEIGHT = 0        # obstacle height (blocks)
    OBSTACLE_PROBABILITY = 0   # obstacle probability (percent)

    ## update field parameter level
    if GAME_LEVEL == 0:   # level0
        GAME_TIME = -1
    elif GAME_LEVEL == 1: # level1
        RANDOM_SEED = 0
    elif GAME_LEVEL == 2: # level2
        RANDOM_SEED = -1
    elif GAME_LEVEL == 3: # level3
        RANDOM_SEED = -1
        OBSTACLE_HEIGHT = 10
        OBSTACLE_PROBABILITY = 40
    else:
        print('invalid level: ' + str(GAME_LEVEL), file=sys.stderr)
        sys.exit(1)

    ## update random seed
    if INPUT_RANDOM_SEED >= 0:
        RANDOM_SEED = INPUT_RANDOM_SEED

    ## print
    print('game_level: ' + str(GAME_LEVEL))
    print('game_time: ' + str(GAME_TIME))
    print('RANDOM_SEED: ' + str(RANDOM_SEED))
    print('IS_MANUAL_CONTROLL :' + str(IS_MANUAL_CONTROLL))
    print('IS_SAMPLE_CONTROLL :' + str(IS_SAMPLE_CONTROLL))
    print('OBSTACLE_HEIGHT: ' + str(OBSTACLE_HEIGHT))
    print('OBSTACLE_PROBABILITY: ' + str(OBSTACLE_PROBABILITY))

    ## start game
    PYTHON_CMD = get_python_cmd()
    cmd = PYTHON_CMD + ' ' + 'game_manager/game_manager.py' \
        + ' ' + '--game_time' + ' ' + str(GAME_TIME) \
        + ' ' + '--seed' + ' ' + str(RANDOM_SEED) \
        + ' ' + '--obstacle_height' + ' ' + str(OBSTACLE_HEIGHT) \
        + ' ' + '--obstacle_probability' + ' ' + str(OBSTACLE_PROBABILITY) \
        + ' ' + '--drop_speed' + ' ' + str(DROP_SPEED) \
        + ' ' + '--manual' + ' ' + str(IS_MANUAL_CONTROLL) \
        + ' ' + '--use_sample' + ' ' + str(IS_SAMPLE_CONTROLL) \
        + ' ' + '--resultlogjson' + ' ' + str(RESULT_LOG_JSON)

    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print('error: subprocess failed.', file=sys.stderr)
        sys.exit(1)
    #p = subprocess.Popen(cmd, shell=True)
    #try:
    #    p.wait()
    #except KeyboardInterrupt:
    #    print("KeyboardInterrupt, call p.terminate()")
    #    p.terminate()

if __name__ == '__main__':
    start()
