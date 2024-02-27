import json
import matplotlib.pyplot as plt
from visualdl import LogWriter
from imgaug import ia

def str2json(str_json):
    return json.loads(str_json.replace('\'', '\"'))

def str_list2json_list(str_json_list):
    json_list = []
    for str_json in str_json_list:
        json_list.append(str2json(str_json))
    return json_list

def get_epoch_eval_best_metrics(monitor, operator, log_path):
    with open(log_path, 'r') as f:
        logs = f.readlines()

        log = str2json(logs[0])
        best = log[monitor]
        best_index = 0
        for i, log in enumerate(logs[1:]):
            log = str2json(log)
            if operator == 'gt':
                if best > log[monitor]:
                    best = log[monitor]
                    best_index = i+1
            elif operator == 'lt':
                if best < log[monitor]:
                    best = log[monitor]
                    best_index = i+1
            else:
                assert operator in ['gt', 'lt'], 'Operator no accept.'

        return str2json(logs[best_index])

def show_epoch_metrics(log_paths, monitors):

    for i, log_path in enumerate(log_paths):
        with open(log_path, 'r') as f:
            logs = f.readlines()
            logs = str_list2json_list(logs)

            with LogWriter(logdir="visual/log{}".format(i)) as writer:
                for log in logs:
                    for monitor in monitors:
                        writer.add_scalar(tag=monitor, step=log['epoch'], value=log[monitor])

def show_PR(log_paths):
    for i, log_path in enumerate(log_paths):
        with open(log_path, 'r') as f:
            logs = f.readlines()
            logs = str_list2json_list(logs)

            with LogWriter(logdir="visual/log{}/precision".format(i)) as writer:
                for log in logs:
                    writer.add_scalar(tag="pr", step=log['epoch'], value=log["ap"])
            with LogWriter(logdir="visual/log{}/recall".format(i)) as writer:
                for log in logs:
                    writer.add_scalar(tag="pr", step=log['epoch'], value=log["ar"])


if __name__ == '__main__':
    # visualdl --logdir ./tools/analysis/visual --port 8080
    # visualdl --logdir ./tools/analysis/visual/log0 --port 8080

    # monitor = 'ap'
    # operator = 'lt'
    # log_path = 'epoch_eval_log(4).txt'
    # result = get_epoch_eval_best_metrics(monitor, operator, log_path)

    # log_paths = ['epoch_eval_log(1).txt', 'epoch_eval_log(2).txt', 'epoch_eval_log(3).txt', 'epoch_eval_log(4).txt']
    # monitors = ['precision', 'recall', 'f1', 'map', 'head_acc']
    # show_epoch_metrics(log_paths, monitors)
    # show_PR(log_paths)
    pass