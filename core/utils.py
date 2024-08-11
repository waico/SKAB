import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .metrics import chp_score


def load_skab():
    path_to_data = "../data/"
    # benchmark files checking
    all_files = []

    for root, dirs, files in os.walk(path_to_data):
        for file in files:
            if file.endswith(".csv"):
                all_files.append(os.path.join(root, file))

    # datasets with anomalies loading
    list_of_df = [
        pd.read_csv(file, sep=";", index_col="datetime", parse_dates=True)
        for file in all_files
        if "anomaly-free" not in file
    ]
    # anomaly-free df loading
    anomaly_free_df = pd.read_csv(
        [file for file in all_files if "anomaly-free" in file][0],
        sep=";",
        index_col="datetime",
        parse_dates=True,
    )

    return list_of_df, anomaly_free_df


def preprocess_skab(list_of_df):
    Xy_traintest_list: list[list] = []
    for df in list_of_df:
        Xy_traintest_list.append(
            train_test_split(
                df.drop(["anomaly", "changepoint"], axis=1),
                df[["anomaly", "changepoint"]],
                train_size=400,
                shuffle=False,
                random_state=0,
            )
        )
    return Xy_traintest_list


def load_preprocess_skab():
    list_of_df, _ = load_skab()
    Xy_traintest_list = preprocess_skab(list_of_df)
    return Xy_traintest_list


# Generated training sequences for use in the model.
def create_sequences(values, time_steps):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)


def plot_results(*true_pred_pairs: tuple[pd.Series, pd.Series]):
    n = len(true_pred_pairs)
    fig, axs = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    if not isinstance(axs, (list | np.ndarray)):
        axs = [axs]
    for ax, (true, pred) in zip(axs, true_pred_pairs):
        ax.plot(true, label="True", marker="o", markersize=5)
        ax.plot(pred, label="Predicted", marker="x", markersize=5)
        ax.set_title(f"{true.name} detection")
        ax.legend()
    fig.show()


def print_results(
    y_true,
    y_pred,
    score_kwargs: list[dict],
):
    for kwargs in score_kwargs:
        print(kwargs)
        chp_score(y_true, y_pred, **kwargs)
        print()
