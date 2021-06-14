import numpy as np


def get_converted_target_list(y_true_list, id_targets, ood_targets):

    id_y_true_list = list()

    for y_true in y_true_list:
        if y_true in id_targets:
            id_y_true = 1
        elif y_true in ood_targets:
            id_y_true = -1
        else:
            raise Exception("No targets")

        id_y_true_list.append(id_y_true)

    return np.array(id_y_true_list)
