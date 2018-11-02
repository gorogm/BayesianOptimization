"""
Baysian hyperparameter optimization [https://github.com/fmfn/BayesianOptimization]
for Mean Absoulte Error objective
on default features for https://www.kaggle.com/c/allstate-claims-severity
"""

__author__ = "Vladimir Iglovikov"

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from bayes_opt import BayesianOptimization
from tqdm import tqdm
import subprocess

def xgb_evaluate(reward_first_step_idle,
                 reward_sooner_later_ratio,
                 reward_collectedPowerup,
                 reward_move_to_enemy,
                 reward_move_to_pickup):

    paramFile = "/tmp/hyperparams.txt"

    file = open(paramFile, "w")
    file.write("reward_first_step_idle=" + str(reward_first_step_idle) + "\r\n")
    file.write("reward_sooner_later_ratio=" + str(reward_sooner_later_ratio) + "\r\n")
    file.write("reward_collectedPowerup=" + str(reward_collectedPowerup) + "\r\n")
    file.write("reward_move_to_enemy=" + str(reward_move_to_enemy) + "\r\n")
    file.write("reward_move_to_pickup=" + str(reward_move_to_pickup) + "\r\n")
    file.write("silent=1" + "\r\n")
    file.close()

    result = subprocess.run(['python3', '/opt/work/pommermanmunchen/playground/berlin_benchmark.py'], stdout=subprocess.PIPE)
    print(result.stderr)
    #print(result.stdout)
    with open('/tmp/hypertune_result.txt', 'r') as content_file:
        winRatio = float(content_file.read())

    return winRatio

if __name__ == '__main__':

    num_iter = 10
    init_points = 10

    """
    float reward_first_step_idle = 0.001f;
    float reward_sooner_later_ratio = 0.98f;
    float reward_collectedPowerup = 0.5f;
    float reward_move_to_enemy = 100.0f;
    float reward_move_to_pickup = 1000.0f;
    """

    xgbBO = BayesianOptimization(xgb_evaluate, {'reward_first_step_idle': (0.01, 0.0001),
                                                'reward_sooner_later_ratio': (0.90, 0.99),
                                                'reward_collectedPowerup': (0.9, 0.1),
                                                'reward_move_to_enemy': (50, 150),
                                                'reward_move_to_pickup': (500, 1500),
                                                })

    xgbBO.maximize(init_points=init_points, n_iter=num_iter)

