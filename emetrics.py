import numpy as np
from lifelines.utils import concordance_index
from sklearn.metrics import precision_recall_curve, auc


def get_cindex(Y, P):
    return concordance_index(Y, P)


def get_mse(Y, P):
    Y = np.array(Y)
    P = np.array(P)
    return np.average((Y - P) ** 2)



def r_squared_error(y_obs, y_pred):
    np.set_printoptions(precision=4)
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))
    return mult / (y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    np.set_printoptions(precision=4)
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / (sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    np.set_printoptions(precision=4)
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / (down))


def get_rm2(ys_orig, ys_line):
    np.set_printoptions(precision=4)

    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))



def get_aupr(Y, P, threshold):
    Y = np.array(Y)
    P = np.array(P)
    Y = np.where(Y >= threshold, 1, 0)
    P = np.where(P >= threshold, 1, 0)
    precision, recall, _ = precision_recall_curve(Y, P)
    aupr = auc(recall, precision)
    return aupr


def get_pearson(Y, P):
    Y = np.array(Y)
    P = np.array(P)
    return np.corrcoef(Y, P)[0, 1]