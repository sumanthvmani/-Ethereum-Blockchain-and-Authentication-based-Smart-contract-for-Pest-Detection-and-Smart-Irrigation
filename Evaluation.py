import numpy as np
import math


def evaluat_error(sp, act) -> object:
    r = np.squeeze(act)
    x = np.squeeze(sp)
    points = np.zeros((len(x)))
    abs_r = np.zeros((len(x)))
    abs_x = np.zeros((len(x)))
    abs_r_x = np.zeros((len(x)))
    abs_x_r = np.zeros((len(x)))
    abs_r_x__r = np.zeros((len(x)))
    for j in range(1, len(x)):
        points[j] = np.mean(abs(x[j] - x[j - 1]))
    for i in range(len(r)):
        abs_r[i] = np.mean(abs(r[i]))
    for i in range(len(r)):
        abs_x[i] = np.mean(abs(x[i]))
    for i in range(len(r)):
        abs_r_x[i] = np.mean(abs(r[i] - x[i]))
    for i in range(len(r)):
        abs_x_r[i] = np.mean(abs(x[i] - r[i]))
    for i in range(len(r)):
        abs_r_x__r[i] = np.mean(abs((r[i] - x[i]) / r[i]))
    md = (100 / len(x)) * sum(abs_r_x__r)
    smape = (1 / len(x)) * sum(abs_r_x / ((abs_r + abs_x) / 2))
    mase = sum(abs_r_x) / ((1 / (len(x) - 1)) * sum(points))
    mae = sum(abs_r_x) / len(r)
    rmse = (sum(abs_x_r ** 2) / len(r)) ** 0.5
    onenorm = sum(abs_r_x)
    twonorm = (sum(abs_r_x ** 2) ** 0.5)
    infinitynorm = np.max(abs_r_x)
    EVAL_ERR = [md, smape, mase, mae, rmse, onenorm, twonorm, infinitynorm]
    return EVAL_ERR


def evaluation(sp, act):
    Tp = np.zeros((len(act), 1))
    Fp = np.zeros((len(act), 1))
    Tn = np.zeros((len(act), 1))
    Fn = np.zeros((len(act), 1))
    for i in range(len(act)):
        p = sp[i]
        a = act[i]
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for j in range(len(p)):
            if a[j] == 1 and p[j] == 1:
                tp = tp + 1
            elif a[j] == 0 and p[j] == 0:
                tn = tn + 1
            elif a[j] == 0 and p[j] == 1:
                fp = fp + 1
            elif a[j] == 1 and p[j] == 0:
                fn = fn + 1
        Tp[i] = tp
        Fp[i] = fp
        Tn[i] = tn
        Fn[i] = fn

    tp = np.squeeze(sum(Tp))
    fp = np.squeeze(sum(Fp))
    tn = np.squeeze(sum(Tn))
    fn = np.squeeze(sum(Fn))

    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
    sensitivity = tp / (tp + fn) * 100
    specificity = tn / (tn + fp) * 100
    precision = tp / (tp + fp) * 100
    FPR = fp / (fp + tn) * 100
    FNR = fn / (tp + fn) * 100
    NPV = tn / (tn + fp) * 100
    FDR = fp / (tp + fp) * 100
    F1_score = (2 * tp) / (2 * tp + fp + fn) * 100
    MCC = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    EVAL = [tp, tn, fp, fn, accuracy, sensitivity, specificity, precision, FPR, FNR, NPV, FDR, F1_score, MCC]
    return EVAL


def evaluation_n(sp, act):
    Tp = np.zeros((len(act), 1))
    Fp = np.zeros((len(act), 1))
    Tn = np.zeros((len(act), 1))
    Fn = np.zeros((len(act), 1))
    for i in range(len(act)):
        p = sp[i]
        a = act[i]
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for j in range(len(p)):
            if a[j] == 1 and p[j] == 1:
                tp = tp + 1
            elif a[j] == 0 and p[j] == 0:
                tn = tn + 1
            elif a[j] == 0 and p[j] == 1:
                fp = fp + 1
            elif a[j] == 1 and p[j] == 0:
                fn = fn + 1
        Tp[i] = tp
        Fp[i] = fp
        Tn[i] = tn
        Fn[i] = fn

    tp = sum(Tp)
    fp = sum(Fp)
    tn = sum(Tn)
    fn = sum(Fn)

    accuracy = ((tp + tn) / (tp + tn + fp + fn)) * 100
    sensitivity = (tp / (tp + fn)) * 100
    specificity = (tn / (tn + fp)) * 100
    precision = (tp / (tp + fp)) * 100
    FPR = (fp / (fp + tn)) * 100
    FNR = (fn / (tp + fn)) * 100
    NPV = (tn / (tn + fp)) * 100
    FDR = (fp / (tp + fp)) * 100
    F1_score = ((2 * tp) / (2 * tp + fp + fn)) * 100
    MCC = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    PT = np.math.sqrt(FPR) / (np.math.sqrt(sensitivity) + np.math.sqrt(FPR))  # Prevalence Threshold (PT)
    CSI = (tp / (tp + fn + fp)) * 100  # Threat Score (TS) or Critical Success Index (CSI)
    FM = np.math.sqrt(specificity * precision)  # Fowlkes–Mallows Index (FM)
    BM = sensitivity + specificity - 1  # Informedness or Bookmaker Informedness (BM)
    MK = precision + NPV - 1  # Markedness (MK) or DeltaP (Δp)
    EVAL = [tp, tn, fp, fn, accuracy, sensitivity, specificity, precision, FPR, FNR, NPV, FDR, F1_score, MCC, PT, CSI,
            FM, BM, MK]
    return EVAL


def evaluation_1(sp, act):
    Tp = np.zeros((len(act), 1))
    Fp = np.zeros((len(act), 1))
    Tn = np.zeros((len(act), 1))
    Fn = np.zeros((len(act), 1))
    for i in range(len(act)):
        p = sp[i]
        a = act[i]
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for j in range(len(p)):
            if a[j] == 1 and p[j] == 1:
                tp = tp + 1
            elif a[j] == 0 and p[j] == 0:
                tn = tn + 1
            elif a[j] == 0 and p[j] == 1:
                fp = fp + 1
            elif a[j] == 1 and p[j] == 0:
                fn = fn + 1
        Tp[i] = tp
        Fp[i] = fp
        Tn[i] = tn
        Fn[i] = fn

    tp = np.squeeze(sum(Tp))
    fp = np.squeeze(sum(Fp))
    tn = np.squeeze(sum(Tn))
    fn = np.squeeze(sum(Fn))

    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
    sensitivity = tp / (tp + fn) * 100
    specificity = tn / (tn + fp) * 100
    precision = tp / (tp + fp) * 100
    FPR = fp / (fp + tn) * 100
    FNR = fn / (tp + fn) * 100
    NPV = tn / (tn + fp) * 100
    FDR = fp / (tp + fp) * 100
    FOR = fn / (fn + tn) * 100
    F1_score = (2 * tp) / (2 * tp + fp + fn) * 100
    MCC = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    EVAL = [tp, tn, fp, fn, accuracy, sensitivity, FNR, specificity, FPR, precision, FDR, NPV, FOR, F1_score, MCC]
    return EVAL
