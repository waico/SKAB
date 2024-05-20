"""
This module is part of library (tsad)[https://github.com/waico/tsad]
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def filter_detecting_boundaries(detecting_boundaries):
    """
    [[t1,t2],[],[t1,t2]] -> [[t1,t2],[t1,t2]]
    [[],[]] -> []
    """
    _detecting_boundaries = []
    for couple in detecting_boundaries.copy():
        if len(couple) != 0:
            _detecting_boundaries.append(couple)
    detecting_boundaries = _detecting_boundaries
    return detecting_boundaries


def single_detecting_boundaries(
    true_series,
    true_list_ts,
    prediction,
    portion,
    window_width,
    anomaly_window_destination,
    intersection_mode,
):
    """
    Extract detecting_boundaries from series or list of timestamps
    """

    if (true_series is not None) and (true_list_ts is not None):
        raise Exception("Choose the ONE type")
    elif true_series is not None:
        true_timestamps = true_series[true_series == 1].index
    elif true_list_ts is not None:
        if len(true_list_ts) == 0:
            return [[]]
        else:
            true_timestamps = true_list_ts
    else:
        raise Exception("Choose the type")
    #
    detecting_boundaries = []
    td = (
        pd.Timedelta(window_width)
        if window_width is not None
        else pd.Timedelta(
            (prediction.index[-1] - prediction.index[0])
            / (len(true_timestamps) + 1)
            * portion
        )
    )
    for val in true_timestamps:
        if anomaly_window_destination == "lefter":
            detecting_boundaries.append([val - td, val])
        elif anomaly_window_destination == "righter":
            detecting_boundaries.append([val, val + td])
        elif anomaly_window_destination == "center":
            detecting_boundaries.append([val - td / 2, val + td / 2])
        else:
            raise RuntimeError("choose anomaly_window_destination")

    # block for resolving intersection problem:
    # important to watch right boundary to be never included to avoid windows intersection
    if len(detecting_boundaries) == 0:
        return detecting_boundaries

    new_detecting_boundaries = detecting_boundaries.copy()
    intersection_count = 0
    for i in range(len(new_detecting_boundaries) - 1):
        if (
            new_detecting_boundaries[i][1]
            >= new_detecting_boundaries[i + 1][0]
        ):
            # transform print to list of intersections
            # print(f'Intersection of scoring windows {new_detecting_boundaries[i][1], new_detecting_boundaries[i+1][0]}')
            intersection_count += 1
            if intersection_mode == "cut left window":
                new_detecting_boundaries[i][1] = new_detecting_boundaries[
                    i + 1
                ][0]
            elif intersection_mode == "cut right window":
                new_detecting_boundaries[i + 1][0] = new_detecting_boundaries[
                    i
                ][1]
            elif intersection_mode == "cut both":
                _a = new_detecting_boundaries[i][1]
                new_detecting_boundaries[i][1] = new_detecting_boundaries[
                    i + 1
                ][0]
                new_detecting_boundaries[i + 1][0] = _a
            else:
                raise Exception("choose the intersection_mode")
    # print(f'There are {intersection_count} intersections of scoring windows')
    detecting_boundaries = new_detecting_boundaries.copy()
    return detecting_boundaries


def check_errors(my_list):
    """
    Check format of input true data

    Parameters
    ----------
    my_list - uniform format of true (See evaluate.evaluate)

    Returns
    ----------
    mx : depth of list, or variant of processing
    """
    assert isinstance(my_list, list)
    mx = 1
    #     ravel = []
    level_list = {}

    def check_error(my_list):
        return not (
            (all(isinstance(my_el, list) for my_el in my_list))
            or (all(isinstance(my_el, pd.Series) for my_el in my_list))
            or (all(isinstance(my_el, pd.Timestamp) for my_el in my_list))
        )

    def recurse(my_list, level=1):
        nonlocal mx
        nonlocal level_list

        if check_error(my_list):
            raise Exception(
                f"Non uniform data format in level {level}: {my_list}"
            )

        if level not in level_list.keys():
            level_list[level] = []  # for checking format

        for my_el in my_list:
            level_list[level].append(my_el)
            if isinstance(my_el, list):
                mx = max([mx, level + 1])
                recurse(my_el, level + 1)

    recurse(my_list)
    for level in level_list:
        if check_error(level_list[level]):
            raise Exception(
                f"Non uniform data format in level {level}: {my_list}"
            )

    if 3 in level_list:
        for el in level_list[2]:
            if not ((len(el) == 2) or (len(el) == 0)):
                raise Exception(
                    f"Non uniform data format in level {2}: {my_list}"
                )
    return mx


def extract_cp_confusion_matrix(
    detecting_boundaries, prediction, point=0, binary=False
):
    """
    prediction: pd.Series

    point=None for binary case
    Returns
    ----------
    dict: TPs: dict of numer window of [t1,t_cp,t2]
    FPs: list of timestamps
    FNs: list of numer window
    """
    _detecting_boundaries = []
    for couple in detecting_boundaries.copy():
        if len(couple) != 0:
            _detecting_boundaries.append(couple)
    detecting_boundaries = _detecting_boundaries

    times_pred = prediction[prediction.dropna() == 1].sort_index().index

    my_dict = {}
    my_dict["TPs"] = {}
    my_dict["FPs"] = []
    my_dict["FNs"] = []

    if len(detecting_boundaries) != 0:
        my_dict["FPs"].append(
            times_pred[times_pred < detecting_boundaries[0][0]]
        )  # left
        for i in range(len(detecting_boundaries)):
            times_pred_window = times_pred[
                (times_pred >= detecting_boundaries[i][0])
                & (times_pred <= detecting_boundaries[i][1])
            ]
            times_prediction_in_window = prediction[
                detecting_boundaries[i][0] : detecting_boundaries[i][1]
            ].index
            if len(times_pred_window) == 0:
                if not binary:
                    my_dict["FNs"].append(i)
                else:
                    my_dict["FNs"].append(times_prediction_in_window)
            else:
                my_dict["TPs"][i] = [
                    detecting_boundaries[i][0],
                    times_pred_window[point]
                    if not binary
                    else times_pred_window,  # attention
                    detecting_boundaries[i][1],
                ]
                if binary:
                    my_dict["FNs"].append(
                        times_prediction_in_window[
                            ~times_prediction_in_window.isin(times_pred_window)
                        ]
                    )
            if len(detecting_boundaries) > i + 1:
                my_dict["FPs"].append(
                    times_pred[
                        (times_pred > detecting_boundaries[i][1])
                        & (times_pred < detecting_boundaries[i + 1][0])
                    ]
                )

        my_dict["FPs"].append(
            times_pred[times_pred > detecting_boundaries[i][1]]
        )  # right
    else:
        my_dict["FPs"].append(times_pred)

    if len(my_dict["FPs"]) > 1:
        my_dict["FPs"] = np.concatenate(my_dict["FPs"])
    elif len(my_dict["FPs"]) == 1:
        my_dict["FPs"] = my_dict["FPs"][0]
    if len(my_dict["FPs"]) == 0:  # not elif on purpose
        my_dict["FPs"] = []

    if binary:
        if len(my_dict["FNs"]) > 1:
            my_dict["FNs"] = np.concatenate(my_dict["FNs"])
        elif len(my_dict["FNs"]) == 1:
            my_dict["FNs"] = my_dict["FNs"][0]
        if len(my_dict["FNs"]) == 0:  # not elif on purpose
            my_dict["FNs"] = []
    return my_dict


def confusion_matrix(true, prediction):
    true_ = true == 1
    prediction_ = prediction == 1
    TP = (true_ & prediction_).sum()
    TN = (~true_ & ~prediction_).sum()
    FP = (~true_ & prediction_).sum()
    FN = (true_ & ~prediction_).sum()
    return TP, TN, FP, FN


def single_average_delay(
    detecting_boundaries,
    prediction,
    anomaly_window_destination,
    clear_anomalies_mode,
):
    """
    anomaly_window_destination: 'lefter', 'righter', 'center'. Default='right'
    """
    detecting_boundaries = filter_detecting_boundaries(detecting_boundaries)
    point = 0 if clear_anomalies_mode else -1
    dict_cp_confusion = extract_cp_confusion_matrix(
        detecting_boundaries, prediction, point=point
    )

    missing = 0
    detectHistory = []
    all_true_anom = 0
    FP = 0

    FP += len(dict_cp_confusion["FPs"])
    missing += len(dict_cp_confusion["FNs"])
    all_true_anom += len(dict_cp_confusion["TPs"]) + len(
        dict_cp_confusion["FNs"]
    )

    if anomaly_window_destination == "lefter":

        def average_time(output_cp_cm_tp):
            return output_cp_cm_tp[2] - output_cp_cm_tp[1]
    elif anomaly_window_destination == "righter":

        def average_time(output_cp_cm_tp):
            return output_cp_cm_tp[1] - output_cp_cm_tp[0]
    elif anomaly_window_destination == "center":

        def average_time(output_cp_cm_tp):
            return output_cp_cm_tp[1] - (
                output_cp_cm_tp[0]
                + (output_cp_cm_tp[2] - output_cp_cm_tp[0]) / 2
            )
    else:
        raise Exception("Choose anomaly_window_destination")

    for fp_case_window in dict_cp_confusion["TPs"]:
        detectHistory.append(
            average_time(dict_cp_confusion["TPs"][fp_case_window])
        )
    return missing, detectHistory, FP, all_true_anom


def my_scale(
    fp_case_window=None,
    A_tp=1,
    A_fp=0,
    koef=1,
    detalization=1000,
    clear_anomalies_mode=True,
    plot_figure=False,
):
    """
    ts - segment on which the window is applied
    """
    x = np.linspace(-np.pi / 2, np.pi / 2, detalization)
    x = x if clear_anomalies_mode else x[::-1]
    y = (
        (A_tp - A_fp)
        / 2
        * -1
        * np.tanh(koef * x)
        / (np.tanh(np.pi * koef / 2))
        + (A_tp - A_fp) / 2
        + A_fp
    )
    if not plot_figure and fp_case_window is not None:
        event = int(
            (fp_case_window[1] - fp_case_window[0])
            / (fp_case_window[-1] - fp_case_window[0])
            * detalization
        )
        if event >= len(x):
            event = len(x) - 1
        score = y[event]
        return score
    else:
        return y


def single_evaluate_nab(
    detecting_boundaries,
    prediction,
    table_of_coef=None,
    clear_anomalies_mode=True,
    scale_func="improved",
    scale_koef=1,
):
    """

    detecting_boundaries: list of list of two float values
                The list of lists of left and right boundary indices
                for scoring results of labeling if empty. Can be [[]], or [[],[t1,t2],[]]
    table_of_coef: pandas array (3x4) of float values
                Table of coefficients for NAB score function
                indices: 'Standard','LowFP','LowFN'
                columns:'A_tp','A_fp','A_tn','A_fn'

    scale_func {default}, improved
    недостатки scale_func default  -
    1 - зависит от относительного шага, а это значит, что если
    слишком много точек в scoring window то перепад будет слишком
    жестким в середение.
    2-   то самая левая точка не равно  Atp, а права не равна Afp
    (особенно если пррименять расплывающую множитель)

    clear_anomalies_mode тогда слева от границы Atp срправа Afp,
    иначе fault mode, когда слева от границы Afp срправа Atp
    """
    if scale_func == "improved":
        scale_func = my_scale
    else:
        raise Exception("choose the scale_func")

    # filter
    detecting_boundaries = filter_detecting_boundaries(detecting_boundaries)

    if table_of_coef is None:
        table_of_coef = pd.DataFrame(
            [
                [1.0, -0.11, 1.0, -1.0],
                [1.0, -0.22, 1.0, -1.0],
                [1.0, -0.11, 1.0, -2.0],
            ]
        )
        table_of_coef.index = pd.Index(["Standard", "LowFP", "LowFN"])
        table_of_coef.index.name = "Metric"
        table_of_coef.columns = ["A_tp", "A_fp", "A_tn", "A_fn"]

    # GO
    point = 0 if clear_anomalies_mode else -1
    dict_cp_confusion = extract_cp_confusion_matrix(
        detecting_boundaries, prediction, point=point
    )

    Scores, Scores_perfect, Scores_null = [], [], []
    for profile in ["Standard", "LowFP", "LowFN"]:
        A_tp = table_of_coef["A_tp"][profile]
        A_fp = table_of_coef["A_fp"][profile]
        A_fn = table_of_coef["A_fn"][profile]

        score = 0
        score += A_fp * len(dict_cp_confusion["FPs"])
        score += A_fn * len(dict_cp_confusion["FNs"])
        for fp_case_window in dict_cp_confusion["TPs"]:
            set_times = dict_cp_confusion["TPs"][fp_case_window]
            score += scale_func(set_times, A_tp, A_fp, koef=scale_koef)

        Scores.append(score)
        Scores_perfect.append(len(detecting_boundaries) * A_tp)
        Scores_null.append(len(detecting_boundaries) * A_fn)

    return np.array(
        [np.array(Scores), np.array(Scores_null), np.array(Scores_perfect)]
    )


def chp_score(
    true,
    prediction,
    metric="nab",
    window_width=None,
    portion=0.1,
    anomaly_window_destination="lefter",
    clear_anomalies_mode=True,
    intersection_mode="cut right window",
    table_of_coef=None,
    scale_func="improved",
    scale_koef=1,
    plot_figure=False,
    verbose=True,
):
    """
    Parameters
    ----------
    true: variants:
        or: if one dataset : pd.Series with binary int labels (1 is
        anomaly, 0 is not anomaly);

        or: if one dataset : list of pd.Timestamp of true labels, or []
        if haven't labels ;

        or: if one dataset : list of list of t1,t2: left and right
        detection, boundaries of pd.Timestamp or [[]] if haven't labels

        or: if many datasets: list (len of number of datasets) of pd.Series
        with binary int labels;

        or: if many datasets: list of list of pd.Timestamp of true labels, or
        true = [ts,[]] if haven't labels for specific dataset;

        or: if many datasets: list of list of list of t1,t2: left and right
        detection boundaries of pd.Timestamp;
        If we haven't true labels for specific dataset then we must insert
        empty list of labels: true = [[[]],[[t1,t2],[t1,t2]]].

        __True labels of anomalies or changepoints.
        It is important to have appropriate labels (CP or
        anomaly) for corresponding metric (See later "metric")

    prediction: variants:
        or: if one dataset : pd.Series with binary int labels
        (1 is anomaly, 0 is not anomaly);

        or: if many datasets: list (len of number of datasets)
        of pd.Series with binary int labels.

        __Predicted labels of anomalies or changepoints.
        It is important to have appropriate labels (CP or
        anomaly) for corresponding metric (See later "metric")

    metric: {'nab', 'binary', 'average_time', 'confusion_matrix'}.
        Default='nab'
        Affects to output (see later: Returns)
        Changepoint problem: {'nab', 'average_time'}.
        Standard AD problem: {'binary', 'confusion_matrix'}.
        'nab' is Numenta Anomaly Benchmark metric

        'average_time' is both average delay or time to failure
        depend on situation.

        'binary': FAR, MAR, F1.

        'confusion_matrix' standard confusion_matrix for any point.

    window_width: 'str' for pd.Timedelta
        Width of detection window. Default=None.

    portion : float, default=0.1
        The portion is needed if window_width = None.
        The width of the detection window in this case is equal
        to a portion of the width of the length of prediction divided
        by the number of real CPs in this dataset. Default=0.1.

    anomaly_window_destination: {'lefter', 'righter', 'center'}. Default='right'
        The parameter of the location of the detection window relative to the anomaly.
        'lefter'  : the detection window will be on the left side of the anomaly
        'righter' : the detection window will be on the right side of the anomaly
        'center'  : the scoring window will be positioned relative to the center of anom.

    clear_anomalies_mode : boolean, default=True.
        True : then the `left value of a Scoring function is Atp and the
        `right is Afp. Only the `first value inside the detection window is taken.
        False: then the `right value of a Scoring function is Atp and the
        `left is Afp. Only the `last value inside the detection window is taken.

    intersection_mode: {'cut left window', 'cut right window', 'both'}.
        Default='cut right window'
        The parameter will be used if the detection windows overlap for
        true changepoints, which is generally undesirable and requires a
        different approach than simply cropping the scoring window using
        this parameter.
        'cut left window' : will cut the overlapping part of the left window
        'cut right window': will cut the intersecting part of the right window
        'both'            : will crop the intersecting portion of both the left
        and right windows

    verbose:  boolean, default=True.
        If True, then output useful information

    plot_figure : boolean, default=False.
        If True, then drawing the score fuctions, detection windows and predictions
        It is used for example, for calibration the scale_koef.

    table_of_coef (metric='nab'): pd.DataFrame of specific form. See bellow.
        Application profiles of NAB metric.If Default is None:
        table_of_coef = pd.DataFrame([[1.0,-0.11,1.0,-1.0],
                                      [1.0,-0.22,1.0,-1.0],
                                      [1.0,-0.11,1.0,-2.0]])
        table_of_coef.index = ['Standard','LowFP','LowFN']
        table_of_coef.index.name = "Metric"
        table_of_coef.columns = ['A_tp','A_fp','A_tn','A_fn']

    scale_func (metric='nab'): "default" of "improved". Default="improved".
        Scoring function in NAB metric.
        'default'  : standard NAB scoring function
        'improved' : Our function for resolving disadvantages
        of standard NAB scoring function

    scale_koef : float > 0. Default=1.0.
        Smoothing factor. The smaller it is,
        the smoother the scoring function is.

    Returns
    ----------
    metrics : value of metrics, depend on metric
        'nab': tuple
            - Standard profile, float
            - Low FP profile, float
            - Low FN profile
        'average_time': tuple
            - Average time (average delay, or time to failure)
            - Missing changepoints, int
            - FPs, int
            - Number of true changepoints, int
        'binary': tuple
            - F1 metric, float
            - False alarm rate, %, float
            - Missing Alarm Rate, %, float
        'binary': tuple
            - TPs, int
            - TNs, int
            - FPs, int
            - FNS, int

    """

    assert isinstance(true, pd.Series) or isinstance(true, list)
    # checking prediction
    if isinstance(prediction, pd.Series):
        true = [true]
        prediction = [prediction]
    elif isinstance(prediction, list):
        if not all(isinstance(my_el, pd.Series) for my_el in prediction):
            raise Exception("Incorrect format for prediction")
    else:
        raise Exception("Incorrect format for prediction")

    # checking dataset length: Number of dataset unequal
    assert len(true) == len(prediction)

    # final check
    input_variant = check_errors(true)

    def check_sort(my_list, input_variant):
        for dataset in my_list:
            if input_variant == 2:
                assert all(np.sort(dataset) == np.array(dataset))
            elif input_variant == 3:
                assert all(
                    np.sort(np.concatenate(dataset)) == np.concatenate(dataset)
                )
            elif input_variant == 1:
                assert all(
                    dataset.index.values == dataset.sort_index().index.values
                )

    check_sort(true, input_variant)
    check_sort(prediction, 1)

    # part 2. To detected boundaries
    if (
        ((metric == "nab") or (metric == "average_time"))
        and (window_width is None)
        and (input_variant != 3)
    ):
        print(
            f"Since you didn't choose window_width and portion, portion will be default ({portion})"
        )

    if input_variant == 1:
        detecting_boundaries = [
            single_detecting_boundaries(
                true_series=true[i],
                true_list_ts=None,
                prediction=prediction[i],
                window_width=window_width,
                portion=portion,
                anomaly_window_destination=anomaly_window_destination,
                intersection_mode=intersection_mode,
            )
            for i in range(len(true))
        ]

    elif input_variant == 2:
        detecting_boundaries = [
            single_detecting_boundaries(
                true_series=None,
                true_list_ts=true[i],
                prediction=prediction[i],
                window_width=window_width,
                portion=portion,
                anomaly_window_destination=anomaly_window_destination,
                intersection_mode=intersection_mode,
            )
            for i in range(len(true))
        ]

    elif input_variant == 3:
        detecting_boundaries = true.copy()
        # Next anti fool system [[[t1,t2]],[]] -> [[[t1,t2]],[[]]]
        for i in range(len(detecting_boundaries)):
            if len(detecting_boundaries[i]) == 0:
                detecting_boundaries[i] = [[]]
    else:
        raise Exception("Unknown format for true data")

    # part 3. To compute metric
    if plot_figure:
        num_datasets = len(true)
        if ((metric == "binary") or (metric == "confusion_matrix")) and (
            input_variant == 1
        ):
            f = plt.figure(figsize=(16, 5 * num_datasets))
            grid = gridspec.GridSpec(num_datasets, 1)
            for i in range(num_datasets):
                globals()["ax" + str(i)] = f.add_subplot(grid[i])
                prediction[i].plot(
                    ax=globals()["ax" + str(i)], label="pred", marker="o"
                )
                true[i].plot(  # type: ignore
                    ax=globals()["ax" + str(i)], label="true", marker="o"
                )
                globals()["ax" + str(i)].legend()
            plt.show()
        else:
            f = plt.figure(figsize=(16, 5 * num_datasets))
            grid = gridspec.GridSpec(num_datasets, 1)
            detalization = 100
            for i in range(num_datasets):
                globals()["ax" + str(i)] = f.add_subplot(grid[i])
                print_legend_boundary = True

                def plot_cp(couple, anomaly_window_destination, ax, label):
                    if anomaly_window_destination == "lefter":
                        ax.axvline(couple[1], c="r", label=label)
                    elif anomaly_window_destination == "righter":
                        ax.axvline(couple[0], c="r", label=label)
                    elif anomaly_window_destination == "center":
                        ax.axvline(
                            couple[0] + ((couple[1] - couple[0]) / 2),
                            c="r",
                            label=label,
                        )

                for couple in detecting_boundaries[i]:
                    if len(couple) > 0:
                        globals()["ax" + str(i)].axvspan(
                            couple[0],
                            couple[1],
                            alpha=0.5,
                            color="green",
                            label="detection \nboundary"
                            if print_legend_boundary
                            else None,
                        )
                        nab = pd.Series(
                            my_scale(
                                plot_figure=True, detalization=detalization
                            ),
                            index=pd.date_range(
                                couple[0], couple[1], periods=detalization
                            ),
                        )
                        nab.plot(
                            ax=globals()["ax" + str(i)],
                            linewidth=0.4,
                            color="brown",
                            label="nab scoring func"
                            if print_legend_boundary
                            else None,
                        )
                        plot_cp(
                            couple,
                            anomaly_window_destination,
                            globals()["ax" + str(i)],
                            label="Changepoint"
                            if print_legend_boundary
                            else None,
                        )
                        print_legend_boundary = False
                    else:
                        pass
                prediction[i].plot(
                    ax=globals()["ax" + str(i)], label="pred", marker="o"
                )
                globals()["ax" + str(i)].legend()
            plt.show()

    if metric == "nab":
        matrix = np.zeros((3, 3))
        for i in range(len(prediction)):
            matrix_ = single_evaluate_nab(
                detecting_boundaries[i],
                prediction[i],
                table_of_coef=table_of_coef,
                clear_anomalies_mode=clear_anomalies_mode,
                scale_func=scale_func,
                scale_koef=scale_koef,
                # plot_figure=plot_figure,
            )
            matrix = matrix + matrix_

        results = {}
        desc = ["Standard", "LowFP", "LowFN"]
        for t, profile_name in enumerate(desc):
            results[profile_name] = round(
                100
                * (matrix[0, t] - matrix[1, t])
                / (matrix[2, t] - matrix[1, t]),
                2,
            )
            if verbose:
                print(profile_name, " - ", results[profile_name])
        return results

    elif metric == "average_time":
        missing, detectHistory, FP, all_true_anom = 0, [], 0, 0
        for i in range(len(prediction)):
            missing_, detectHistory_, FP_, all_true_anom_ = (
                single_average_delay(
                    detecting_boundaries[i],
                    prediction[i],
                    anomaly_window_destination=anomaly_window_destination,
                    clear_anomalies_mode=clear_anomalies_mode,
                )
            )
            missing, detectHistory, FP, all_true_anom = (
                missing + missing_,
                detectHistory + detectHistory_,
                FP + FP_,
                all_true_anom + all_true_anom_,
            )
        add = np.mean(detectHistory)
        if verbose:
            print("Amount of true anomalies", all_true_anom)
            print(f"A number of missed CPs = {missing}")
            print(f"A number of FPs = {int(FP)}")
            print("Average time", add)
        return add, missing, int(FP), all_true_anom

    elif (metric == "binary") or (metric == "confusion_matrix"):
        if all(isinstance(my_el, pd.Series) for my_el in true):
            TP, TN, FP, FN = 0, 0, 0, 0
            for i in range(len(prediction)):
                TP_, TN_, FP_, FN_ = confusion_matrix(true[i], prediction[i])
                TP, TN, FP, FN = TP + TP_, TN + TN_, FP + FP_, FN + FN_
        else:
            print(
                "For this metric it is better if you use pd.Series format for true \nwith common index of true and prediction"
            )
            TP, TN, FP, FN = 0, 0, 0, 0
            for i in range(len(prediction)):
                dict_cp_confusion = extract_cp_confusion_matrix(
                    detecting_boundaries[i], prediction[i], binary=True
                )
                TP += np.sum(
                    [
                        len(dict_cp_confusion["TPs"][window][1])
                        for window in dict_cp_confusion["TPs"]
                    ]
                )
                FP += len(dict_cp_confusion["FPs"])
                FN += len(dict_cp_confusion["FNs"])
                TN += len(prediction[i]) - TP - FP - FN

        if metric == "binary":
            f1 = round(TP / (TP + (FN + FP) / 2), 2)
            far = round(FP / (FP + TN) * 100, 2)
            mar = round(FN / (FN + TP) * 100, 2)
            if verbose:
                print(f"False Alarm Rate {far} %")
                print(f"Missing Alarm Rate {mar} %")
                print(f"F1 metric {f1}")
            return f1, far, mar

        elif metric == "confusion_matrix":
            if verbose:
                print("TP", TP)
                print("TN", TN)
                print("FP", FP)
                print("FN", FN)
            return TP, TN, FP, FN
    else:
        raise Exception("Choose the performance metric")
