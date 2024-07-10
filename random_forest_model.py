#!/usr/bin/python
# -*- coding: UTF-8 -*-txt')
# written by Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

# <editor-fold desc="**0** load the package">
import os
import argparse
from datetime import datetime, timezone

import pytz
import platform

import numpy as np
import pandas as pd
from itertools import combinations

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
# </editor-fold>


def rfModel(rf_classifier, X_train, y_train, X_test, y_test):
    # training
    rf_classifier.fit(X_train, y_train)

    # testing
    y_pred = rf_classifier.predict(X_test)
    y_pred_probability = rf_classifier.predict_proba(X_test)

    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
    print(f"testing: sration: {STATION}, F1:{f1:.4f}")

    # save mdoel
    joblib.dump(rf_classifier, f"{OUTPUT_DIR}{STATION}rf.pkl")

    Visualize_confusion_matrix(y_train, rf_classifier.predict(X_train), "training")

    return y_pred, y_pred_probability


def record_Modelresults(timeStamps_test, y_test, Pro, y_pred, y_pred_probability):

    f = open(f"{OUTPUT_DIR}{STATION}RFloss_out_all.txt", 'a')
    title = f"step, station, time, observed_label, observed_probability, predicted_label, predicted_0, predicted_1"
    f.write(str(title) + "\n")
    f.close()

    f = open(f"{OUTPUT_DIR}{STATION}RFloss_out_all.txt", 'a')
    title = f"step, station, time, observed_label, observed_probability, predicted_label, predicted_0, predicted_1"
    f.write(str(title) + "\n")
    f.close()

    for i in range(len(y_test)):

        f = open(f"{OUTPUT_DIR}{STATION}RFloss_out_all.txt", 'a')
        time = datetime.fromtimestamp(timeStamps_test[i], tz=pytz.utc)
        time = time.strftime('%Y-%m-%d %H:%M:%S')

        record = f"{i},{STATION},{time},{y_test[i]},{Pro[i]:.3},{y_pred[i]},{y_pred_probability[i][0]:.3f},{y_pred_probability[i][1]:.3f}"
        f.write(str(record) + "\n")

        if y_test[i]==1 or y_pred[i]==1:
            f = open(f"{OUTPUT_DIR}{STATION}RFloss_out.txt", 'a')
            record = f"{i},{STATION},{time},{y_test[i]},{Pro[i]:.3},{y_pred[i]}"
            f.write(str(record) + "\n")


def Visualize_confusion_matrix(all_targets, all_outputs, training_or_testing):
    plt.rcParams.update({'font.size': 8})#, 'font.family': "Arial"})
    cm_raw = confusion_matrix(all_targets, all_outputs)
    cm_df_raw = pd.DataFrame(cm_raw, index=["0:None DF", "1:DF"], columns=["0:None DF", "1:DF"])

    cm_normalize = confusion_matrix(all_targets, all_outputs, normalize='true')
    cm_df_normalize = pd.DataFrame(cm_normalize, index=["0:None DF", "1:DF"], columns=["0:None DF", "1:DF"])

    f1 = f1_score(all_targets, all_outputs, average='binary', zero_division=0)

    fig = plt.figure(figsize=(4.5, 4.5))
    ax1 = fig.add_subplot(1, 1, 1)
    sns.heatmap(cm_df_raw, xticklabels=1, yticklabels=1, annot=True, fmt='.1f', square=True, cmap="Blues", cbar=False)

    plt.text(x=0.35, y=0.62, s=f"{cm_df_normalize.iloc[0, 0]:.4f}", color="black")
    plt.text(x=1.35, y=0.62, s=f"{cm_df_normalize.iloc[0, 1]:.4f}", color="black")

    plt.text(x=0.35, y=1.62, s=f"{cm_df_normalize.iloc[1, 0]:.4f}", color="black")
    plt.text(x=1.35, y=1.62, s=f"{cm_df_normalize.iloc[1, 1]:.4f}", color="black")

    plt.ylabel("Actual Class", weight='bold')
    plt.xlabel(f"Predicted Class" + "\n" + f"{training_or_testing}, {STATION}, F1={f1:.4}", weight='bold')

    plt.tight_layout()
    if training_or_testing == "training":
        plt.savefig(f"{OUTPUT_DIR}{STATION}_trainingF1_{f1:.4f}.png", dpi=600)
    elif training_or_testing == "testing":
        plt.savefig(f"{OUTPUT_DIR}{STATION}_testingF1_{f1:.4f}.png", dpi=600)
    plt.close(fig)


def createNoneOverlapTimeSeq(allTimeID, usedTimeID, minDFduration=30, maxDFduration=180):
    overlapLength, i = 1, 0
    while overlapLength > 0:
        randomStartID = np.random.choice(allTimeID, size=1)[0]  # random start time
        randomEndID = randomStartID + np.random.randint(minDFduration, maxDFduration + 1)  # random DF length
        overlapLength = np.intersect1d(usedTimeID, np.arange(randomStartID, randomEndID)).size

        i += 1
        if i == 100:
            print("can not find enough space for s1&s2")
            break
    return randomStartID, randomEndID


def runRF_BL_time(input_station):
    global DATA_DIR
    # feature goodness of fit and alpha
    DATA_DIR = f"/home/qizhou/1projects/dataForML/out60filterFor1BLpaper/processed/"

    df1 = pd.read_csv(f"{DATA_DIR}all1component/2017-2020{input_station}_EHZ_BL.txt",
                      header=0, low_memory=False, usecols=[4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17])#15 is IQR
    df1 = df1.fillna(0)
    df1 = df1.replace([np.inf, -np.inf], 100)

    df2 = pd.read_csv(f"{DATA_DIR}#2017-2020{input_station}_observedLabelsYES.txt",
                      header=0, low_memory=False, usecols=[1, 2, 4])

    df = pd.concat([df1, df2], axis=1, ignore_index=True)
    columnsName = np.concatenate([df1.columns.values, df2.columns.values])
    df.columns = columnsName

    id1 = 509760
    train_df = df.iloc[:id1, :]
    X_train, y_train, timeStamps_train, pro_train = \
        train_df.iloc[:, :-3], train_df.iloc[:, -2], \
        train_df.iloc[:, -3], train_df.iloc[:, -1]
    X_train, y_train, timeStamps_train, pro_train = \
        X_train.astype(float), y_train.astype(float), \
        timeStamps_train.astype(float), pro_train.astype(float)

    test_df = df.iloc[id1:, :]
    X_test, y_test, timeStamps_test, pro_test = \
        test_df.iloc[:, :-3], test_df.iloc[:, -2], \
        test_df.iloc[:, -3], test_df.iloc[:, -1]
    X_test, y_test, timeStamps_test, pro_test = \
        X_test.astype(float), y_test.astype(float), \
        timeStamps_test.astype(float), pro_test.astype(float)

    inputFeatures_names_train = columnsName[:-3]

    classifier = RandomForestClassifier(n_estimators=800)
    y_pred, y_pred_probability = rfModel(classifier, X_train, y_train, X_test, y_test)

    Visualize_confusion_matrix(all_targets=y_test, all_outputs=y_pred, training_or_testing="testing")
    record_Modelresults(np.array(timeStamps_test), np.array(y_test), np.array(pro_test),
                        y_pred, y_pred_probability)

    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
    print(f"testing: sration: {STATION}, F1:{f1:.4f}")


def runRF_BL_segment(input_station):
    global DATA_DIR
    DATA_DIR = f"/home/qizhou/1projects/dataForML/out60filterFor1BLpaper/processed/"

    # <editor-fold desc="load data">
    df1 = pd.read_csv(f"{DATA_DIR}all1component/2017-2020{input_station}_EHZ_BL.txt",
                      header=0, low_memory=False, usecols=[4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17]) #15 is IQR
    df1 = df1.fillna(0)
    df1 = df1.replace([np.inf, -np.inf], 100)

    df2 = pd.read_csv(f"{DATA_DIR}#2017-2020{input_station}_observedLabelsYES.txt",
                      header=0, low_memory=False, usecols=[1, 2, 4])

    df = pd.concat([df1, df2], axis=1, ignore_index=True)
    columnsName = np.concatenate([df1.columns.values, df2.columns.values])
    df.columns = columnsName
    # </editor-fold>


    # <editor-fold desc="train model">
    id1 = 509760
    #train_df = df.iloc[:id1, :]
    #X_train, y_train, timeStamps_train, pro_train = \
        #train_df.iloc[:, :-3], train_df.iloc[:, -2], \
        #train_df.iloc[:, -3], train_df.iloc[:, -1]
    #X_train, y_train, timeStamps_train, pro_train = \
        #X_train.astype(float), y_train.astype(float), \
        #timeStamps_train.astype(float), pro_train.astype(float)

    #classifier = RandomForestClassifier(n_estimators=800)
    #classifier.fit(X_train, y_train) # train
    #joblib.dump(rf_classifier, f"{OUTPUT_DIR}{STATION}rf.pkl") # save model

    #inputFeatures_names_train = columnsName[:-3] # input features name
    #feature_imp_visualize(classifier, inputFeatures_names_train, rf_or_bl)
    #Visualize_confusion_matrix(y_train, rf_classifier.predict(X_train), "training")

    # </editor-fold>

    classifier = joblib.load(f"/home/qizhou/3paper/1BL/Step2RfDetector/RFmodel/1st2020OneByOne/{STATION}rf.pkl")

    # <editor-fold desc="test model">
    dfTP = pd.read_csv("/home/qizhou/3paper/1BL/Step2RfDetector/2020events.txt", header=None)
    test_dfALL = df.iloc[id1:, :]  # all 2020 data
    allTimeStamps = np.array(test_dfALL.iloc[:, 12])

    for stepDurationThreshold in range(1, 61):
        allTimeID = np.arange(0, allTimeStamps.shape[0])
        usedTimeID = np.empty((0, 1))

        y_testManually = np.empty((0, 1))
        y_predManually = np.empty((0, 1))


        for step in range(len(dfTP) * 101):

            if step <= 11:  # TP period
                s1 = datetime.strptime(dfTP.iloc[step, 0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
                s2 = datetime.strptime(dfTP.iloc[step, 1], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()

                s1 = np.where(allTimeStamps == s1)[0][0]
                s2 = np.where(allTimeStamps == s2)[0][0]

                y_testManually = np.append(y_testManually, [1])  # Positive
            else:  # background noise period
                s1, s2 = createNoneOverlapTimeSeq(allTimeID, usedTimeID)
                y_testManually = np.append(y_testManually, [0])  # Negative

            if np.intersect1d(usedTimeID, np.arange(s1, s2)).size > 0 and step >= 12:
                print("overlap", step, s1, s2)

            usedTimeID = np.append(usedTimeID, np.arange(s1, s2 + 1))
            allTimeID = np.setdiff1d(allTimeID, np.arange(s1, s2 + 1))

            test_df = test_dfALL.iloc[s1:s2 + 1, :]
            X_test, y_test, timeStamps_test, pro_test = \
                test_df.iloc[:, :-3], test_df.iloc[:, -2], \
                test_df.iloc[:, -3], test_df.iloc[:, -1]
            X_test, y_test, timeStamps_test, pro_test = \
                X_test.astype(float), y_test.astype(float), \
                timeStamps_test.astype(float), pro_test.astype(float)

            y_pred = classifier.predict(X_test)
            y_pred_probability = classifier.predict_proba(X_test)

            if np.sum(y_pred) >= stepDurationThreshold:  # thresold to control how many positive
                y_predManually = np.append(y_predManually, [1])
                record_Modelresults(np.array(timeStamps_test),
                                    np.array(y_test), np.array(pro_test),
                                    y_pred, y_pred_probability)
            else:
                y_predManually = np.append(y_predManually, [0])

        cm = confusion_matrix(y_testManually, y_predManually)
        print(stepDurationThreshold, cm)
        TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        f1 = f1_score(y_testManually, y_predManually, average='binary', zero_division=0)
        FPR, FNR = FP / (FP + TP), FN / (FN + TP)

        f = open(f"{OUTPUT_DIR}/#durationThreshold.txt", 'a')
        record = f"{stepDurationThreshold}, {TN}, {FP}, {FN}, {TP}, {f1}, {FPR}, {FNR}"
        f.write(str(record) + "\n")
        f.close()


        if stepDurationThreshold == 15:
            Visualize_confusion_matrix(all_targets=y_testManually, all_outputs=y_predManually, training_or_testing="testing")
        else:
            pass



def main(input_station:str):
    global STATION, OUTPUT_DIR, DATA_DIR

    STATION = input_station
    DATA_DIR = f"It is defined in the function runRF_BL"

    OUTPUT_DIR = f"/home/qizhou/3paper/1BL/Step2RfDetector/RFmodel/1st2020OneByOne/"
    runRF_BL_time(input_station=STATION)

    OUTPUT_DIR = f"/home/qizhou/3paper/1BL/Step2RfDetector/RFmodel/2nd2020SegmentBySegment/"
    runRF_BL_segment(input_station=STATIONB)


    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple distributed job')
    parser.add_argument("--input_station", default="ILL12", type=str, help="station source")

    args = parser.parse_args()

    main(args.input_station)
