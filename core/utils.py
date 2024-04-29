import os

import pandas as pd


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
