#!/usr/bin/python
# -*- coding: UTF-8 -*-txt')
import os
import numpy as np
import pandas as pd
import pytz
from datetime import datetime

from scipy.optimize import curve_fit
from scipy.fftpack import hilbert
from scipy.stats import iqr, ks_2samp, chi2_contingency, mannwhitneyu
from sklearn.metrics import r2_score

from obspy import read, Stream, UTCDateTime

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter

plt.rcParams.update( {'font.size':7, 'axes.formatter.limits': (-2, 2) } )

global DIR, SAC_DIR, NAME, TIME, STATION, CHANNEL, SOURCE, SPS, WINDOW, obsOptimalTIME, RULER
DIR = "/home/qizhou/1projects/0events_catalog/RawWaveforms/EventPeriod/"
WINDOW = 60


def calBL_feature(data, ruler=0):
    '''
    Parameters
    ----------
    data: np array data after dtrend+dmean, or raw data
    ruler: shit the data from 0 to ruler 100, default = 100

    Returns: 1*16 np.array
    -------
    '''
    data = np.abs(data)
    # BL theoretical value
    BL_frequency = [0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046]
    dataSelected = data[data >= 100]
    print(dataSelected.shape)

    # <editor-fold desc="iq, max, min">
    iq = float("{:.2f}".format(iqr(dataSelected)))
    max_amp = float("{:.2f}".format(np.max(dataSelected - ruler)))
    min_amp = float("{:.2f}".format(np.min(dataSelected - ruler)))
    # </editor-fold>

    # <editor-fold desc="digit frequency">
    amp_data = pd.DataFrame(dataSelected)
    amp_data = amp_data.astype(str)

    d = (amp_data.iloc[:, 0]).str[0: 1]
    d = list(d)

    digit_count = np.empty((0, 9))
    for digit in range(1, 10):
        first_digit = d.count(str(digit))
        digit_count = np.append(digit_count, first_digit)

    digit_frequency = digit_count / np.sum(digit_count)
    digit_frequency = [float('{:.3f}'.format(i)) for i in digit_frequency]
    # </editor-fold>

    # <editor-fold desc="get goodness, ks, chi-squared, alpha">
    frequency = np.empty((0, 9))
    for a in range(0, 9):
        first_digit_frequency = pow((digit_frequency[a] - BL_frequency[a]), 2) / BL_frequency[a]
        frequency = np.append(frequency, first_digit_frequency)
    goodness = (1 - pow(sum(frequency), 0.5)) * 100
    goodness = float("{:.3f}".format(goodness))

    statistic, pvalue = ks_2samp(BL_frequency, digit_frequency, alternative='two-sided', method='exact')
    ks = float("{:.3f}".format(pvalue))  # pvalue

    # digit_frequency = [x / sum(digit_frequency) for x in digit_frequency] # for chi squared
    #statistic, pvalue = chisquare(f_obs=digit_frequency, f_exp=BL_frequency)
    #chi = float("{:.3f}".format(pvalue)) # pvalue

    statistic, pvalue = mannwhitneyu(BL_frequency, digit_frequency, alternative='two-sided', method='exact')
    MannWhitneU = float("{:.3f}".format(pvalue))  # pvalue

    if ks >= 0.95 and MannWhitneU >= 0.95:
        follow = 1 # follow BL
    else:
        follow = 0 # do not follow BL

    sum_d = []
    y_min = np.min(dataSelected)
    if y_min == 0: # in csse of "divide by zero encountered in scalar divide"
        y_min = 1
    else:
        pass
    for s in range(0, len(dataSelected)):
        i = np.log(dataSelected[s] / y_min)
        sum_d.append(i)
    alpha = 1 + len(dataSelected) / np.sum(sum_d)
    alpha = float("{:.4f}".format(alpha))
    # </editor-fold>

    output = np.array([max_amp, min_amp, iq, goodness, alpha, ks, MannWhitneU, follow], dtype=float)
    output = np.append(digit_frequency, output)
    return output


def plotAll(event_start_id, event_end_id, max_goodness_id,
            amp, followBL, goodness, alpha,
            optimalGoodnessArray, obsOptimalTIME,
            obsStartGoodnessArray, observedStartTIME,
            obsEndGoodnessArray, observedEndTIME):
    fig = plt.figure(figsize=(5.5, 4))

    # <editor-fold desc="1a amp">
    amp = np.nan_to_num(amp, nan=0)

    ax1 = fig.add_subplot(4, 2, 1)
    plt.plot(amp, color="black", lw=1,
             label=f"{STATION}-{CHANNEL}" + "\n" +
                   "demean, detrend, filter 1-45Hz")

    plt.axvline(x=event_start_id * 60 * SPS, color="grey", lw=1, ls="--")
    plt.axvline(x=event_end_id * 60 * SPS, color="grey", lw=1, ls="--")
    plt.axvline(x=max_goodness_id * 60 * SPS, color="#CD5C5C", lw=1, ls="--")

    plt.text(x=0, y=np.nanmax(amp) * 0.7, s="(a)", weight="bold")
    plt.legend(loc='upper right', fontsize=5)
    plt.grid(axis='y', ls="--", lw=0.5)

    ax1.axes.xaxis.set_ticklabels([])
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(SPS * 3600 * 2))
    ax1.ticklabel_format(axis='y', style='sci')
    plt.ylabel("Amplitude" + "\n" + "(counts)", fontweight="bold")
    # </editor-fold>

    p = False
    if p == True:
        # <editor-fold desc="1b, ks">
        ax2 = fig.add_subplot(5, 2, 3)
        ks = [0 if i < 0.95 else i for i in ks]

        plt.scatter(np.arange(0, len(ks)), ks, facecolors="black", edgecolors="none", s=7, alpha=0.5)

        plt.grid(axis='y', ls="--", lw=0.5)
        plt.ylabel("Kolmogorov" + "\n" + "Smirnov test", fontweight="bold")
        plt.ylim(-0.3, 1.3)
        plt.text(x=0, y=1, s="(b)", weight="bold")

        # event start and end time
        plt.axvline(x=event_start_id, color="grey", lw=1, ls="--")
        plt.axvline(x=event_end_id, color="grey", lw=1, ls="--")
        # max goodness of fit
        plt.axvline(x=max_goodness_id, color="#CD5C5C", lw=1, ls="--")

        ax2.axes.xaxis.set_ticklabels([])
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))
        # x-axis is 2h interval
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(60 * 2))
        # </editor-fold>

        # <editor-fold desc="1c, MannWhitneyU">
        ax3 = fig.add_subplot(5, 2, 5)
        MannWhitneyU = [0 if i < 0.95 else i for i in MannWhitneyU]

        plt.scatter(np.arange(0, len(MannWhitneyU)), MannWhitneyU, facecolors="black", edgecolors="none", s=7,
                    alpha=0.5)
        plt.grid(axis='y', ls="--", lw=0.5)
        plt.ylabel("Mann Whitney" + "\n" + "U rank test", fontweight="bold")
        plt.ylim(-0.3, 1.3)
        plt.text(x=0, y=1, s="(c)", weight="bold")

        # event start and end time
        plt.axvline(x=event_start_id, color="grey", lw=1, ls="--")
        plt.axvline(x=event_end_id, color="grey", lw=1, ls="--")
        # max goodness of fit
        plt.axvline(x=max_goodness_id, color="#CD5C5C", lw=1, ls="--")

        ax3.axes.xaxis.set_ticklabels([])
        ax3.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax3.xaxis.set_major_locator(ticker.MultipleLocator(60 * 2))
        # </editor-fold>

        # <editor-fold desc="1b, ks">

    # <editor-fold desc="1a followBL">
    ax2 = fig.add_subplot(4, 2, 3)
    followBL = np.nan_to_num(followBL, nan=0)
    plt.scatter(np.arange(0, len(followBL)), followBL, facecolors="black", edgecolors="none", s=7, alpha=0.5)

    plt.grid(axis='y', ls="--", lw=0.5)
    plt.ylabel("Hypothesis" + "\n" "test", fontweight="bold")
    plt.ylim(-0.3, 1.3)
    plt.text(x=0, y=1, s="(b)", weight="bold")
    plt.legend(["1: follow BL" + "\n"
                + "0: not follow BL"], loc='center right', fontsize=5)

    # event start and end time
    plt.axvline(x=event_start_id, color="grey", lw=1, ls="--")
    plt.axvline(x=event_end_id, color="grey", lw=1, ls="--")
    # max goodness of fit
    plt.axvline(x=max_goodness_id, color="#CD5C5C", lw=1, ls="--")

    ax2.axes.xaxis.set_ticklabels([])
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # x-axis is 2h interval
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(60 * 2))
    # </editor-fold>

    # <editor-fold desc="1d, alpha">
    ax7 = fig.add_subplot(4, 2, 5)
    alpha = np.nan_to_num(alpha, nan=5)
    plt.scatter(np.arange(0, len(alpha)), alpha, facecolors="black", edgecolors="none", s=7, alpha=0.5)
    plt.grid(axis='y', ls="--", lw=0.5)
    plt.ylabel("Power law" + "\n" + r"exponent"+ r"$\mathbf{\mathit{\alpha}}$", fontweight="bold")
    plt.ylim(1, 6)
    plt.text(x=0, y=4, s="(d)", weight="bold")
    plt.yticks([1, 3, 5], ["1", "3", "5"])
    #plt.legend([f"shift up {RULER}"], loc='upper right', fontsize=5)


    # event start and end time
    plt.axvline(x=event_start_id, color="grey", lw=1, ls="--")
    plt.axvline(x=event_end_id, color="grey", lw=1, ls="--")
    # max goodness of fit
    plt.axvline(x=max_goodness_id, color="#CD5C5C", lw=1, ls="--")

    ax7.axes.xaxis.set_ticklabels([])
    ax7.xaxis.set_major_locator(ticker.MultipleLocator(60 * 2))
    # </editor-fold>

    # <editor-fold desc="1d, goodness of fit">
    ax4 = fig.add_subplot(4, 2, 7)
    goodness = np.nan_to_num(goodness, nan=-50)

    plt.scatter(np.arange(0, len(goodness)), goodness, facecolors="black", edgecolors="none", s=7, alpha=0.5)

    plt.ylabel(r"Goodness of fit"+ "\n" + r"$\mathbf{\mathit{\phi}}$ (%)", fontweight="bold")
    plt.xlabel(f"UTC+0 Time {SOURCE}" + "\n" + f"(from {TIME} by minute)", fontweight="bold")
    plt.text(x=0, y=np.nanmax(goodness) * 0.7, s="(e)", weight="bold")

    # event start and end time
    plt.axvline(x=event_start_id, color="grey", lw=1, ls="--")
    plt.axvline(x=event_end_id, color="grey", lw=1, ls="--")
    # max goodness of fit
    plt.axvline(x=max_goodness_id, color="#CD5C5C", lw=1, ls="--")

    plt.grid(axis='y', ls="--", lw=0.5)

    ax4.xaxis.set_major_locator(ticker.MultipleLocator(60 * 2))
    # </editor-fold>



    # <editor-fold desc="1e, max goodness of fit">
    ax5 = fig.add_subplot(1, 2, 2)

    digit = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    data0 = [0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046]
    obsOptimal = optimalGoodnessArray.astype(float)
    obsStart = obsStartGoodnessArray.astype(float)
    obseEnd = obsEndGoodnessArray.astype(float)


    plt.plot(digit, data0, color="black", marker="o", markeredgecolor="black", ls="--", markersize=4, lw=0.8,
             label="BL theoretical")

    plt.plot(digit, obsStart, color="grey", marker="o", markeredgecolor="black", ls="--", markersize=3, lw=0.8,
             label="Labeled starting" + "\n" +
                   observedStartTIME + "\n" +
                   r"$\phi$=" + f"{goodness[event_start_id]:.2f}%")

    plt.plot(digit, obsOptimal, color="#CD5C5C", marker="o", markeredgecolor="black", ls="--", markersize=4, lw=0.8,
             label="Observed" + "\n" +
                   obsOptimalTIME + "\n" +
                   r"$\phi$=" + f"{goodness[max_goodness_id]:.2f}%")

    plt.plot(digit, obseEnd, color="grey", marker="s", markeredgecolor="black", ls="--", markersize=3, lw=0.8,
             label="Labeled ending" + "\n" +
                   observedEndTIME + "\n" +
                   r"$\phi$=" + f"{goodness[event_end_id]:.2f}%")

    plt.grid(axis='x', lw=0.5, ls="--")

    for step in range(1, 9):
        d = digit[step - 1:step + 1]
        y = data0[step - 1:step + 1]
        z = obsOptimal[step - 1:step + 1]

        fun_1 = np.polyfit(d, y, deg=1)
        fun_2 = np.polyfit(d, z, deg=1)

        f = lambda x: x * fun_1[0] + fun_1[1]
        g = lambda x: x * fun_2[0] + fun_2[1]
        x = np.linspace(step, step + 1, 400)

        fx = f(x)
        gx = g(x)

        plt.fill_between(x, fx, gx, color="#CD5C5C", alpha=0.5)

    plt.xlabel(r"First Digit", fontweight="bold")
    plt.ylabel("Digit Frequency", fontweight="bold")

    plt.xlim(0.5, 9.5)
    plt.legend(loc='upper right')
    plt.text(x=1, y=0.05, s="(e)", weight="bold")

    ax5.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # </editor-fold>

    plt.tight_layout()
    plt.savefig(f"{DIR}BLwaveforms/{NAME}.png", dpi=600)
    #plt.show()


def PSD(st, event_start_id, event_end_id, max_goodness_id):
    fig = plt.figure(figsize=(5.5, 2))
    ax2 = fig.add_subplot(1, 1, 1)
    st.spectrogram(per_lap=0.5, wlen=60, log=False, dbscale=True, mult=True, title="", axes=ax2)

    plt.ylim(0, 50)
    plt.axvline(x=event_start_id * 60, color="grey", lw=1, ls="--")
    plt.axvline(x=event_end_id * 60, color="grey", lw=1, ls="--")
    plt.axvline(x=max_goodness_id * 60, color="white", lw=2, ls="--")

    cbar = plt.colorbar(mappable=ax2.images[-1], ax=ax2, orientation='vertical')
    cbar.set_label('Power (dB)')

    ax2.xaxis.set_major_locator(ticker.MultipleLocator(120 * 60))  # unit is saecond
    ax2.set_ylabel("Frequency (Hz)", fontweight="bold")

    plt.xlabel(f"UTC+0 Time" + "\n" +
               f"(from {TIME} by minute)", fontweight="bold")


    plt.tight_layout()
    plt.savefig(f"{DIR}psd/{NAME}_psd.png", dpi=600)
    #plt.show()
    # </editor-fold>


def func_in(x, a, b, c):  # increase part
    return a * np.exp(b * x) + c


def func_de(x, a, b, c):  # decrease part
    return a * np.exp(-b * x) + c


def fit_ExpCurve(y, p0):
    '''
    :param y: time series iq for fitting, data array
    :param p0: initial parameters for fitting
    :return: cofficients of Exp function, a, b, c, r2
    '''
    # p0 = [5000, 0.1, -4000] #for 2013 2014
    # p0 = [100, 0.1, -1000] #for 2017-2020
    try:
        popt, pcov = curve_fit(func_in, np.arange(0, len(y), 1), y, p0=p0, maxfev=5000)
        a, b, c = popt[0], popt[1], popt[2]

        fit_y = func_in(np.arange(0, len(y), 1), a, b, c)
        r2 = r2_score(y, fit_y)
    except Exception as error:
        a, b, c, r2 = 0, 0, 0, 0
        print(error)

    return a, b, c, r2


def plot_fit_ExpCurve(iq_array, id, y, a, b, c, time, name):
    '''
    :param iq_array: time series iq array, defaut is 1s one point
    :param id and TIME: the #id and start TIME# of Used data
    :param y: used data for fiting
    :param a b c: cofficients of fitting curve
    :return:
    '''
    fig = plt.figure(figsize=(5, 4))
    ax1 = fig.add_subplot(1, 1, 1)

    x = np.arange(0, iq_array.size, 1)
    xUsed = np.arange(id, id + y.size, 1)

    plt.plot(x, iq_array, color="black", lw=1,
             label="Observed IQR")

    fit_y = func_in(xUsed - id, a, b, c)
    plt.plot(xUsed, fit_y, color="red", lw=1,
             label="Fitted Exp. curve")

    plt.scatter(xUsed, y, facecolors="#519E3E", edgecolors="none", alpha=0.5,
                label="Data for fitting")

    aStr = "{:.2e}".format(a)
    bStr = "{:.2e}".format(b)
    cStr = "{:.2e}".format(c)
    equation_text = r'$S(t)={a} \cdot e^{{{b} \cdot t}} + {c}$'

    plt.text(x=0, y=max(fit_y) * 0.8, s=equation_text)
    plt.text(x=0, y=max(fit_y) * 0.7, s=f"$a$={aStr}, $b$={bStr}, $c$={cStr}")
    plt.text(x=0, y=max(fit_y) * 0.6, s=rf"$R^2={r2:.3f}$")

    ax1.xaxis.set_major_locator(ticker.MultipleLocator(60))

    plt.xlabel(f"UTC+0 Time $t$" + "\n" + f"(Interval by seconds from {time})")
    plt.ylabel("Interquartile range IQR $S(t)$")
    plt.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(f"{DIR}{name}.png", dpi=600)
    #plt.show()


df = pd.read_csv("/home/qizhou/1projects/0events_catalog/#EventInfor/#eventfrom2013to2020ManuallySelected.txt", header=0)

for i in range(len(df)):
    # <editor-fold desc="set event start/end">
    dataS, dataE = UTCDateTime(df.iloc[i, 0])-3600*3, UTCDateTime(df.iloc[i, 1])+3600*3
    eventS, eventE = UTCDateTime(df.iloc[i, 0]), UTCDateTime(df.iloc[i, 1])

    juldayS, juldayE = dataS.julday, dataE.julday

    jDay = [juldayS, juldayE]
    jDay = set(jDay)
    # </editor-fold>


    # <editor-fold desc="select seismic data">
    st = Stream()
    for step in jDay:
        step = str(step).zfill(3)

        if dataS.year in [2013, 2014]:  # 2013-2014 data
            SAC_DIR = f"/home/qizhou/0data/{dataS.year}/IGB02/BHZ/"
            miniSeed_name = f"GM.IGB02.BHZ.{dataS.year}.{step}"
            GAP = 6
            RULER = 0
        elif dataS.year not in [2013, 2014]:  # 2017-2020 data
            SAC_DIR = f"/home/qizhou/0data/{dataS.year}/ILL12/EHZ/"
            miniSeed_name = f"9S.ILL12.EHZ.{dataS.year}.{step}"
            GAP = 4
            RULER = 0
        st += read(f"{SAC_DIR}{miniSeed_name}")

    st.merge(method=1, fill_value=0)
    st.trim(starttime=dataS, endtime=dataE, nearest_sample=False)
    st.detrend('linear')
    st.detrend('demean')
    st.filter("bandpass", freqmin=1, freqmax=45)
    st._cleanup()
    # </editor-fold>


    # <editor-fold desc="cal BL features">
    SPS = st[0].stats.sampling_rate
    bl_array = np.empty((0, 18))
    seismic_array = np.empty((0, int(WINDOW * SPS)))

    for step in range(int((dataE-dataS)/WINDOW)):
        t1 = dataS + (step) * WINDOW
        t2 = dataS + (step + 1) * WINDOW

        tr = st.copy()
        try:
            tr.trim(starttime=t1, endtime=t2, nearest_sample=False)
            seismic_data = tr[0].data[:-1]
            bl = calBL_feature(data=seismic_data, ruler=RULER)
            record = np.append(str(t2), bl)

        except Exception as e:
            print(f"no data at {step}, {t1}, {e}")
            bl = np.full(17, np.nan)
            record = np.append(str(t2), bl)
            seismic_data = np.full(int(WINDOW * SPS), np.nan)

        bl_array = np.vstack((bl_array, record))
        seismic_array = np.vstack((seismic_array, seismic_data))
    # </editor-fold>


    # <editor-fold desc="ready data">
    amp = seismic_array.reshape(-1)
    ks = bl_array[:, 15].astype(float)
    MannWhitneyU = bl_array[:, 16].astype(float)
    followBL = bl_array[:, 17].astype(float)
    goodness = bl_array[:, 13].astype(float)
    alpha = bl_array[:, 14].astype(float)
    IQR = bl_array[:, 12].astype(float)


    max_goodness_id = int((UTCDateTime(df.iloc[i, 4]) - dataS)/60)-1 #manually observed time
    optimalGoodnessArray = bl_array[max_goodness_id, 1:10].astype(float)
    obsOptimalTIME = datetime.strptime(bl_array[max_goodness_id, 0], '%Y-%m-%dT%H:%M:%S.%fZ')
    obsOptimalTIME = obsOptimalTIME.strftime('%Y-%m-%d %H:%M:%S')

    event_start_id = int( (eventS-dataS) / 60)-1
    obsStartGoodnessArray = bl_array[event_start_id, 1:10].astype(float)
    observedStartTIME = datetime.strptime(bl_array[event_start_id, 0], '%Y-%m-%dT%H:%M:%S.%fZ')
    observedStartTIME = observedStartTIME.strftime('%Y-%m-%d %H:%M:%S')

    event_end_id =   int( (eventE-dataS) / 60)-1
    obsEndGoodnessArray = bl_array[event_end_id, 1:10].astype(float)
    observedEndTIME = datetime.strptime(bl_array[event_end_id, 0], '%Y-%m-%dT%H:%M:%S.%fZ')
    observedEndTIME = observedEndTIME.strftime('%Y-%m-%d %H:%M:%S')


    STATION = st[0].stats.station
    CHANNEL = st[0].stats.channel
    TIME = dataS.strftime('%Y-%m-%d %H:%M:%S')
    SOURCE = df.iloc[i, 2]
    NAME = f"{i}_{TIME}"
    # </editor-fold>


    # <editor-fold desc="record data">
    for step in range(bl_array.shape[0]):
        record = bl_array[step, :]
        with open(f"{DIR}BLwaveforms/bl.txt", 'a') as file:
            np.savetxt(file, record.reshape(1, -1), header='', delimiter=',', comments='', fmt='%s')

    record = bl_array[max_goodness_id, :]
    with open(f"{DIR}BLwaveforms/maxGoodnessPeriod.txt", 'a') as file:
        np.savetxt(file, record.reshape(1, -1), header='', delimiter=',', comments='', fmt='%s')

    record = bl_array[event_start_id, :]
    with open(f"{DIR}BLwaveforms/eventStartPeriod.txt", 'a') as file:
        np.savetxt(file, record.reshape(1, -1), header='', delimiter=',', comments='', fmt='%s')

    np.save(f"{DIR}npy/{NAME}.npy",
            seismic_array[max_goodness_id])
    # </editor-fold>


    # <editor-fold desc="plot data">
    plotAll(event_start_id, event_end_id, max_goodness_id,
            amp, followBL, goodness, alpha,
            optimalGoodnessArray, obsOptimalTIME,
            obsStartGoodnessArray, observedStartTIME,
            obsEndGoodnessArray, observedEndTIME)

    #PSD(st, event_start_id, event_end_id, max_goodness_id)
    # </editor-fold>


    # <editor-fold desc="calculate IQR by 1s window">
    obsOptimalTIME = UTCDateTime(obsOptimalTIME)#(df2.iloc[i, 0])
    iq_array = np.empty((0, 1))
    howMany = 6 * 60  # How many seconds before obsOptimalTIME?

    tr = st.copy()
    dataSelectStart = obsOptimalTIME - howMany
    tr.trim(starttime=dataSelectStart, endtime=obsOptimalTIME, nearest_sample=False)

    for step in range(howMany):
        t1 = dataSelectStart + (step)
        t2 = dataSelectStart + (step + 1)

        tr1 = tr.copy()
        try:
            tr1.trim(starttime=t1, endtime=t2, nearest_sample=False)
            seismic_data = tr1[0].data
            m = np.abs(seismic_data[seismic_data != 0])
            iq_array = np.append(iq_array, iqr(m))
        except:
            print(f"error {t1}")
    # </editor-fold>


    # <editor-fold desc="fit the Exp curve">
    if dataS.year in [2013, 2014]:  # 2013-2014 data
        p0 = [5000, 0.1, -4000]
    elif dataS.year not in [2013, 2014]:  # 2017-2020 data
        p0 = [100, 0.1, -1000]

    coefficientsArray = np.empty((0, 4))
    max_iq_arrayID = np.where(iq_array == np.max(iq_array[-60:]))[0][0]

    for step in range(1, 6):  # 5 min before the max goodness period (1-min period)
        id = (5 - step) * 60  # i=1, 1min before max goodness period, select data from 240 to 360
        y = iq_array[id:max_iq_arrayID + 1]

        a, b, c, r2 = fit_ExpCurve(y, p0)
        coefficients = np.array([a, b, c, r2])
        coefficientsArray = np.vstack((coefficientsArray, coefficients))

        name = f"figExp/{step}_{eventS.strftime('%Y-%m-%d %H:%M:%S')}"
        plot_fit_ExpCurve(iq_array, id, y, a, b, c, dataSelectStart.strftime('%Y-%m-%d %H:%M:%S'), name)
        
        
    maxFittedStep = np.argmax(coefficientsArray[:, 3]) + 1
    id = (5 - maxFittedStep) * 60  # i=1, 1min before max goodness period, select data from 240 to 360
    y = iq_array[id:max_iq_arrayID + 1]
    a, b, c, r2 = fit_ExpCurve(y, p0)

    name = f"figExpOptimal/{i}_{eventS.strftime('%Y-%m-%d %H:%M:%S')}"
    plot_fit_ExpCurve(iq_array, id, y, a, b, c, dataSelectStart.strftime('%Y-%m-%d %H:%M:%S'), name)
    # </editor-fold>

    # <editor-fold desc="record data">
    dataUsedPeriod = np.array([dataSelectStart.strftime('%Y-%m-%d %H:%M:%S'),
                               (dataSelectStart + max_iq_arrayID).strftime('%Y-%m-%d %H:%M:%S')])
    record = np.array([a, b, c, r2])
    dataUsedPeriod = np.append(dataUsedPeriod, record)
    with open(f"{DIR}figExpOptimal/OptimalFitted.txt", 'a') as file:
        np.savetxt(file, dataUsedPeriod.reshape(1, -1), header='', delimiter=',', comments='', fmt='%s')
    # </editor-fold>