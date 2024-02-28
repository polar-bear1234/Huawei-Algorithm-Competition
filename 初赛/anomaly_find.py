import numpy as np
from sklearn.cluster import Birch
import pandas as pd
import time
import copy
import warnings
from sklearn.preprocessing import normalize
import plotly.graph_objects as go
import seaborn as sns
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cufflinks as cf
from sklearn import preprocessing
import torch.utils.data as data_utils
from pylab import *
# from utils_anomaly.utils_anomaly import *
# from utils_anomaly.usad import *
from scipy.stats import chi2, kstest, probplot, ks_2samp
from statsmodels.tsa.api import VAR, ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.base.datetools import dates_from_str
import math
from sklearn.metrics import f1_score, precision_score

from utils_anomaly.usad import UsadModel, testing, device
from utils_anomaly.utils import to_device

warnings.filterwarnings("ignore")
smoothing_window = 12


'''异常检测'''


# ============================================================= 1. 基于统计的方法
class MyAR:
    __slots__ = ('samples', 'n', 'p')

    def __init__(self, samples):
        self.samples = samples.iloc[:, 1]
        self.n = samples.shape[0]
        self.p = 1

    def logdiff(self):
        self.samples = np.log(self.samples).diff().dropna()  # 一阶对数差分

    def gene_indep_variable(self):
        n, p = self.n, self.p
        sample = self.samples
        X = np.zeros((p + 1, n - p))
        for i in range(n - p):
            X[:, i] = np.concatenate((np.ones(1), sample[i:i + p].values[::-1]), axis=0)
        return X

    def M_statistic(self, i):
        # return ARIMA(self.samples, (i,0,0)).fit().bic
        model1 = ARIMA(self.samples, order=(i, 0, 0)).fit()
        model2 = ARIMA(self.samples, order=(i - 1, 0, 0)).fit()
        sigma1 = 1 / (self.n - 2 * i - 1) * sum(model1.resid ** 2)
        sigma2 = 1 / (self.n - 2 * (i - 1) - 1) * sum(model2.resid ** 2)
        return -(self.n - i - 2.5) * np.log(sigma1 / sigma2)

    def estimate(self):
        # 定阶
        p = []
        for i in range(1, 9):
            try:
                p.append(self.M_statistic(i))
            except:
                p.append(-1e10)
        self.p = p.index(max(p)) + 1
        y = self.samples.values[self.p:]
        X = self.gene_indep_variable()
        model = ARIMA(self.samples, order=(self.p, 0, 0)).fit()
        new_B = np.array(model.params[:(self.p + 1)])
        new_sigma2 = sum((y - np.dot(new_B, X)) ** 2) / self.n
        residual = model.resid
        return new_sigma2, new_B  # , residual


class ARscoreTest:
    __slots__ = (
    'whole_samples', 'date', 'samples', 'whole_samples_n', 'B', 'p', 'sigma2', 'y', 'X', 'I22', 'significance',
    'ms_critical', 'vw_critical', 'ms_scores', 'vw_scores')

    def __init__(self, samples, sigma2, coef, significance=0.05):
        self.whole_samples = samples
        self.date = samples.iloc[:, 0]
        self.samples = samples.iloc[:, 1]
        self.whole_samples_n = samples.shape[0]
        self.B = coef
        self.p = len(coef) - 1
        self.sigma2 = sigma2
        self.y = self.samples.values[self.p:]
        self.X = self.gene_indep_variable()
        self.I22 = self.calc_I22()
        self.significance = significance
        self.ms_critical = chi2.isf(self.significance / (self.whole_samples_n - self.p), 1)
        self.vw_critical = chi2.isf(self.significance / (self.whole_samples_n - self.p), 1)
        self.ms_scores = None
        self.vw_scores = None

    def gene_indep_variable(self):
        n, p = self.whole_samples_n, self.p
        sample = self.samples
        X = np.zeros((p + 1, n - p))
        for i in range(n - p):
            X[:, i] = np.concatenate((np.ones(1), sample[i:i + p].values[::-1]), axis=0)
        return X

    def calc_I22(self):
        X = self.X
        n, p, sigma2 = self.whole_samples_n, self.p, self.sigma2
        I22 = np.zeros((p + 1 + 1, p + 1 + 1))
        I22[:(p + 1), :(p + 1)] = np.dot(X, X.T)
        I22[(p + 1), (p + 1)] = n / (2 * sigma2 ** 2)
        return I22

    def mean_shift(self):
        B, X = self.B, self.X
        n, p, sigma2 = self.whole_samples_n, self.p, self.sigma2
        critical = self.ms_critical
        y = self.samples.values[p:]
        score = [0] * p
        for i in range(n - p):
            Xi = X[:, i]
            I12 = np.zeros((1, p + 1 + 1))
            I12[:, :(p + 1)] = Xi.reshape(1, p + 1) / sigma2
            Lr = (y[i] - np.dot(B, Xi)) / sigma2
            score.append(Lr ** 2 / (1 / sigma2 - np.dot(I12, np.dot(inv(self.I22), I12.T))))
        scores = pd.DataFrame(score, columns=['scores'])
        scores = pd.concat((self.whole_samples, scores), axis=1, ignore_index=False)
        self.ms_scores = scores
        outliers_scores = scores[scores.scores >= critical]
        return self.ms_scores, outliers_scores  # , correct_samples

    def variance_weight(self):
        B, X = self.B, self.X
        n, p, sigma2 = self.whole_samples_n, self.p, self.sigma2
        critical = self.vw_critical
        y = self.y
        score = [0] * p
        for i in range(n - p):
            Xi = X[:, i]
            score.append(n / (2 * n - 2) * (1 - (y[i] - np.dot(B, Xi)) ** 2 / sigma2) ** 2)
        scores = pd.DataFrame(score, columns=['scores'])
        scores = pd.concat((self.whole_samples, scores), axis=1, ignore_index=False)
        self.vw_scores = scores
        outliers_scores = scores[scores.scores >= critical]
        return self.vw_scores, outliers_scores  # , correct_samples


def stati(df, label2):
    df0 = df.iloc[:, 1:]
    df0['timestamp'] = pd.to_datetime(df0['timestamp'])
    df0 = df0.sort_values(by=['timestamp'], ascending=True)
    df_train = df0.iloc[:43, :]
    df_train.iloc[:, 1:] = np.log(df_train.iloc[:, 1:] + 1).diff()
    df_train = df_train.iloc[1:, :]
    ARmodel = MyAR(df_train.loc[:, ['timestamp', 'ctn_memory']])
    new_sigma2, new_B = ARmodel.estimate()
    df_test = df.iloc[:, 1:]
    df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])
    df_test = df_test.sort_values(by=['timestamp'], ascending=True)
    result = df_test.copy()
    result.iloc[:, 1:] = np.zeros(df_test.iloc[:, 1:].shape)
    df_test.loc[:, 'ctn_memory'] = np.log(df_test.loc[:, 'ctn_memory'] + 1).diff()
    df_test = df_test.iloc[1:, :]
    S = ARscoreTest(df_test.loc[:, ['timestamp', 'ctn_memory']], new_sigma2, new_B, 20)
    [vw_scores, vw_outliers_scores] = S.variance_weight()
    result.loc[result['timestamp'].isin(vw_outliers_scores['timestamp']), 'ctn_memory'] = 1
    target = label2.label.values.tolist()
    pred = result.loc[:, 'ctn_memory'].values.tolist()
    Pre = precision_score(target, pred)
    F1_score = f1_score(target, pred)
    return Pre, F1_score


def statistics():
    Precision, F1_score = stati(df, label2)
    print("准确率为--------------------------{}".format(Precision))
    print("f1_score为-----------------------{}".format(F1_score))


# =================================================================== 2. 机器学习方法

def birch_ad_with_smoothing(latency_df, threshold):
    """
    birch方法，用于后续根因定位
    """
    anomalies = []
    for svc, latency in latency_df.iteritems():
        if svc != 'timestamp' and 'Unnamed' not in svc and 'rabbitmq' not in svc and 'db' not in svc:
            latency = latency.rolling(window=smoothing_window, min_periods=1).mean()
            x = np.array(latency)
            x = np.where(np.isnan(x), 0, x)
            normalized_x = preprocessing.normalize([x])
            X = normalized_x.reshape(-1, 1)
            brc = Birch(branching_factor=50, n_clusters=None, threshold=threshold, compute_labels=True)
            brc.fit(X)
            brc.predict(X)
            labels = brc.labels_
            n_clusters = np.unique(labels).size
            if n_clusters > 1:
                anomalies.append(svc)
    print("经算法检测，异常的指标为----------------------------------------：", "\n", anomalies)
    return anomalies


def birch(x, branching_factor, threshold):
    """
    birch方法，用于异常检测
    """
    brc = Birch(branching_factor=branching_factor,
                n_clusters=2,
                threshold=threshold,
                compute_labels=True)
    brc.fit(x)
    pred = brc.predict(x)
    return pred


def machine_learning():
    smoothing_window = 6
    val = data['@value']
    val = val.rolling(window=smoothing_window, min_periods=1).mean()
    x = np.array(val)
    normal_x = normalize([x])
    norm_x = normal_x.reshape(-1, 1)
    threshold = 0.00000001
    branching_factor = 50
    y_pred = birch(norm_x, branching_factor=branching_factor, threshold=threshold)
    F1_score = f1_score(label['target'].values.tolist(), y_pred)
    Precision = precision_score(label['target'].values.tolist(), y_pred)
    print("准确率为--------------------------{}".format(Precision))
    print("f1_score为-----------------------{}".format(F1_score))


# ============================================3. 基于深度学习的方法

def USADtest(df, normal, BATCH_SIZE, N_EPOCHS, hidden_size, yuzhi, shuchu):
    window_size = 1
    normal = normal.drop(["timestamp"], axis=1)
    min_max_scaler = preprocessing.MinMaxScaler()
    x = normal.values
    x_scaled = min_max_scaler.fit_transform(x)
    normal = pd.DataFrame(x_scaled)
    attack = df.drop(["timestamp"], axis=1)
    y = attack.values
    y_scaled = min_max_scaler.transform(y)
    attack = pd.DataFrame(y_scaled)
    windows_normal = normal.values[np.arange(window_size)[None, :] + np.arange(normal.shape[0] - window_size)[:, None]]
    windows_attack = attack.values[np.arange(window_size)[None, :] + np.arange(attack.shape[0] - window_size)[:, None]]
    w_size = windows_normal.shape[1] * windows_normal.shape[2]
    z_size = windows_normal.shape[1] * hidden_size
    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_attack).float().view(([windows_attack.shape[0], w_size]))), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model = UsadModel(w_size, z_size)
    model = to_device(model, device)
    torch.save({
        'encoder': model.encoder.state_dict(),
        'decoder1': model.decoder1.state_dict(),
        'decoder2': model.decoder2.state_dict()}, "model.pth")
    checkpoint = torch.load("model.pth")
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder1.load_state_dict(checkpoint['decoder1'])
    model.decoder2.load_state_dict(checkpoint['decoder2'])
    results = testing(model, test_loader)
    result_ = results[-1].flatten().detach().cpu().numpy()
    predict_label = []
    for i in range(len(result_)):
        if list(result_)[i] > 1:
            predict_label.append(1)
        else:
            predict_label.append(0)
    predict_label.append(0)
    biaozhu = []
    for i in range(len(predict_label)):
        if predict_label[i] >= yuzhi:
            biaozhu.append(i)
    result_df = df
    result_df['label'] = predict_label
    return biaozhu, predict_label


def deep_learning():
    BATCH_SIZE = 1900
    N_EPOCHS = 100
    hidden_size = 100
    yuzhi = 1
    shuchu = "node_sockstat_TCP_tw"
    biaozhu, predict_label = USADtest(
        anomaly, normal, BATCH_SIZE, N_EPOCHS, hidden_size, yuzhi, shuchu
    )
    target = pd.read_csv('anomaly_data/label_node_sockstat_TCP_tw.csv')
    target = target.target.values.tolist()
    pred = predict_label
    Precision = precision_score(target, pred)
    F1_score = f1_score(target, pred)
    print("准确率为--------------------------{}".format(Precision))
    print("f1_score为-----------------------{}".format(F1_score))


# ##################################### 图像可视化
def plot(data):
    anomaly_result = data[data['anomaly'] == 1]
    print(anomaly_result)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=anomaly_result['@timestamp'], y=anomaly_result['@value'], opacity=0.75, mode='markers',
                             name='anomaly', marker=dict(color='red', size=8)))
    fig.add_trace(go.Scatter(x=data['@timestamp'], y=data['@value'], opacity=0.75, mode='lines', name='value',
                             line=dict(color='#34A5DA', width=5)))
    fig.update_layout(autosize=False, width=1000, height=600)
    fig.show()
    return fig


def detection(num):
    dt_number = {
        1: statistics,
        2: machine_learning,
        3: deep_learning
    }
    detect = dt_number.get(num)
    if detect:
        detect()


def anomaly_detection():
    while 1:
        print('/******************************************\n'
              '可选择异常检测算法如下：\n'
              '1、基于统计的异常检测算法\n'
              '2、基于机器学习的异常检测算法\n'
              '3、基于深度学习的异常检测算法\n'
              '0、退出程序\n'
              '**********************************************/')
        print('请输入想要使用的异常检测算法(1、2、3、0)：\n')
        num = int(input())
        if num < 0 or num > 3:
            print('参数{}输入错误！'.format(num))
            return
        if num == 0:
            print('程序退出！')
            return
        detection(num)
        print('\n')


if __name__ == "__main__":

    df = pd.read_csv('anomaly_data/1_user.csv')
    label = pd.read_csv('anomaly_data/target.csv')
    label2 = pd.read_csv('anomaly_data/label_ctn_memory.csv')
    normal = pd.read_csv('anomaly_data/1_normaly.csv')
    anomaly = pd.read_csv('anomaly_data/1_anomaly.csv')
    df.fillna(0, inplace=True)
    data = df[['timestamp', 'node_memory']].rename(columns={'timestamp': '@timestamp', 'node_memory': '@value'})

    """&&&&&&&&&&--------运行主程序--------&&&&&&&&&&&&&"""
    anomaly_detection()




