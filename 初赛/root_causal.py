import requests
import pandas as pd
import networkx as nx
import gc
import numpy as np
import time
import argparse
import warnings

warnings.filterwarnings("ignore")

metric_step = '60s'
N = 0.81


def mpg(prom_url, faults_name, start_time, end_time):
    DG = nx.DiGraph()
    df = pd.DataFrame(columns=['source', 'destination'])
    response = requests.get(prom_url,
                            params={
                                'query': 'sum(istio_tcp_received_bytes_total{destination_workload!=\'unknown\', source_workload!=\'unknown\'}) by (source_workload, destination_workload)',
                                'start': start_time,
                                'end': end_time, 'step':metric_step})
    results = response.json()['data']['result']
    for result in results:
        metric = result['metric']
        source = metric['source_workload']
        destination = metric['destination_workload']
        df = df.append({'source': source, 'destination': destination}, ignore_index=True)
        DG.add_edge(source, destination)
        DG.nodes[source]['type'] = 'service'
        DG.nodes[destination]['type'] = 'service'
    response = requests.get(prom_url,
                            params={
                                 'query': 'sum(istio_requests_total{destination_workload_namespace=\'sock-shop\',destination_workload!=\'unknown\', source_workload!=\'unknown\'}) by (source_workload, destination_workload)',
                                 'start': start_time,
                                 'end': end_time, 'step': metric_step})
    results = response.json()['data']['result']
    for result in results:
        metric = result['metric']
        source = metric['source_workload']
        destination = metric['destination_workload']
        df = df.append({'source': source, 'destination': destination}, ignore_index=True)
        DG.add_edge(source, destination)
        DG.nodes[source]['type'] = 'service'
        DG.nodes[destination]['type'] = 'service'
    response = requests.get(prom_url,
                            params={
                                'query': 'sum(container_cpu_usage_seconds_total{namespace="sock-shop", name!~\'POD|istio-proxy\'}) by (instance, container)',
                                'start': start_time,
                                'end': end_time, 'step': metric_step
    })
    results = response.json()['data']['result']
    for result in results:
        metric = result['metric']
        if 'container' in metric:
            source = metric['container']
            destination = metric['instance']
            df = df.append({'source': source, 'destination': destination}, ignore_index=True)
            DG.add_edge(source, destination)
            DG.nodes[source]['type'] = 'service'
            DG.nodes[destination]['type'] = 'host'
    filename = 'Dataset1/' + faults_name + '_mpg.csv'
    df.to_csv(filename)
    print("创建图文件------------------------------保存成功：", filename)
    return DG


def node_weight(svc, anomaly_graph, baseline_df, faults_name):
    # Get the average weight of the in_edges
    in_edges_weight_avg = 0.0
    num = 0
    for u, v, data in anomaly_graph.in_edges(svc, data=True):
        num = num + 1
        in_edges_weight_avg = in_edges_weight_avg + data['weight']
    if num > 0:
        in_edges_weight_avg = in_edges_weight_avg / num
    filename = 'Dataset1/' + faults_name + '_' + svc + '.csv'
    df = pd.read_csv(filename)
    node_cols = ['node_cpu', 'node_network', 'node_memory']
    max_corr = 0.01
    metric = 'node_cpu'
    for col in node_cols:
        temp = abs(baseline_df[svc].corr(df[col]))
        if temp > max_corr:
            max_corr = temp
            metric = col
    data = in_edges_weight_avg * max_corr
    return data, metric


def svc_personalization(svc, anomaly_graph, baseline_df, faults_name):
    filename = 'Dataset1/' + faults_name + '_' + svc + '.csv'
    df = pd.read_csv(filename)
    ctn_cols = ['ctn_cpu', 'ctn_network', 'ctn_memory']
    max_corr = 0.01
    metric = 'ctn_cpu'
    for col in ctn_cols:
        temp = abs(baseline_df[svc].corr(df[col]))     
        if temp > max_corr:
            max_corr = temp
            metric = col
    edges_weight_avg = 0.0
    num = 0
    for u, v, data in anomaly_graph.in_edges(svc, data=True):
        num = num + 1
        edges_weight_avg = edges_weight_avg + data['weight']
    for u, v, data in anomaly_graph.out_edges(svc, data=True):
        if anomaly_graph.nodes[v]['type'] == 'service':
            num = num + 1
            edges_weight_avg = edges_weight_avg + data['weight']
    edges_weight_avg = edges_weight_avg / num
    personalization = edges_weight_avg * max_corr
    return personalization, metric


def anomaly_subgraph(DG, anomalies, latency_df, faults_name, alpha):
    edges = []
    nodes = []
    baseline_df = pd.DataFrame()
    edge_df = {}
    for anomaly in anomalies:
        edge = anomaly.split('_')
        edges.append(tuple(edge))
        svc = edge[1]
        nodes.append(svc)
        baseline_df[svc] = latency_df[anomaly]
        edge_df[svc] = anomaly
    nodes = set(nodes)
    personalization = {}
    for node in DG.nodes():
        if node in nodes:
            personalization[node] = 0
    # Get the subgraph of anomaly
    anomaly_graph = nx.DiGraph()
    for node in nodes:
        # print("anomaly_subgraph中node为----------------------：", '\n', node)
        for u, v, data in DG.in_edges(node, data=True):
            edge = (u, v)
            if edge in edges:
                data = alpha
            else:
                normal_edge = u + '_' + v
                data = baseline_df[v].corr(latency_df[normal_edge])
            data = round(data, 3)
            anomaly_graph.add_edge(u,v, weight=data)
            anomaly_graph.nodes[u]['type'] = DG.nodes[u]['type']
            anomaly_graph.nodes[v]['type'] = DG.nodes[v]['type']
        for u, v, data in DG.out_edges(node, data=True):
            edge = (u, v)
            if edge in edges:
                data = alpha
            else:
                if DG.nodes[v]['type'] == 'host':
                    data, col = node_weight(u, anomaly_graph, baseline_df, faults_name)
                else:
                    normal_edge = u + '_' + v
                    data = baseline_df[u].corr(latency_df[normal_edge])
            data = round(data, 3)
            anomaly_graph.add_edge(u, v, weight=data)
            anomaly_graph.nodes[u]['type'] = DG.nodes[u]['type']
            anomaly_graph.nodes[v]['type'] = DG.nodes[v]['type']
    for node in nodes:
        max_corr, col = svc_personalization(node, anomaly_graph, baseline_df, faults_name)
        personalization[node] = max_corr / anomaly_graph.degree(node)
    anomaly_graph = anomaly_graph.reverse(copy=True)
    edges = list(anomaly_graph.edges(data=True))
    for u, v, d in edges:
        if anomaly_graph.nodes[node]['type'] == 'host':
            anomaly_graph.remove_edge(u, v)
            anomaly_graph.add_edge(v, u, weight=d['weight'])
    anomaly_score = nx.pagerank(anomaly_graph, alpha=0.85, personalization=personalization, max_iter=10000)
    anomaly_score = sorted(anomaly_score.items(), key=lambda x: x[1], reverse=True)
    return anomaly_score


def attributed_graph(faults_name):
    filename = 'Dataset1/' + faults_name + '_mpg.csv'
    df = pd.read_csv(filename)
    DG = nx.DiGraph()
    for index, row in df.iterrows():
        source = row['source']
        destination = row['destination']
        if 'rabbitmq' not in source and 'rabbitmq' not in destination and 'db' not in destination and 'db' not in source:
            DG.add_edge(source, destination)
    for node in DG.nodes():
        if 'k8s-node' in node:
            DG.nodes[node]['type'] = 'host'
        else:
            DG.nodes[node]['type'] = 'service'
    return DG


def rt_invocations(faults_name):
    latency_filename = 'Dataset1/' + faults_name + '_latency_source_50.csv'  # inbound
    latency_df_source = pd.read_csv(latency_filename)
    latency_df_source['unknown_front-end'] = 0
    latency_filename = 'Dataset1/' + faults_name + '_latency_destination_50.csv'  # outbound
    latency_df_destination = pd.read_csv(latency_filename)
    latency_df = latency_df_destination.append(latency_df_source)
    return latency_df


def parse_args():
    """Parse the args."""
    parser = argparse.ArgumentParser(
        description='Root cause analysis for microservices')
    parser.add_argument('--folder', type=str, required=False,
                        default='all',
                        help='folder name to store csv file')
    parser.add_argument('--length', type=int, required=False,
                        default=1,
                        help='length of time series')
    parser.add_argument('--url', type=str, required=False,
                        default='http://47.103.63.176:31200/api/v1/query_range',
                        help='url of prometheus query')
    return parser.parse_args()


def print_rank(anomaly_score, target):
    num = 10
    for idx, anomaly_target in enumerate(anomaly_score):
        if target in anomaly_target:
            num = idx + 1
            continue
    print(target, ' Top K: ', num)
    return num


def prk(df, k, n):
    result = pd.DataFrame(columns=[
        'folder', 'target', 'fault_type', 'num', 'anomaly_score_new', 'anomaly_nodes'
    ])
    for i in range(1, n+1):
        data = df[df['num'] == i]
        result = result.append(data)
    df_ = result[result.num <= k]
    pre = df_.shape[0]/result.shape[0]
    # print(df_.shape[0], result.shape[0])
    print("PR@{}得分为--------------------------{}：".format(k, pre))
    return pre


def prk2(df1, k, n):
    print(df1.target.unique())
    Pr_df = pd.DataFrame(columns=df1.target.unique())
    for i, col in enumerate(df1.target.unique()):
        df0 = df1[df1['target'] == col]
        result = pd.DataFrame(columns=[
            'folder', 'target', 'fault_type', 'num', 'anomaly_score_new', 'anomaly_nodes'
        ])
        for j in range(1, n+1):
            data = df0[df0['num'] == j]
            result = result.append(data)
        df_ = result[result.num <= k]
        if df_.shape[0] == 0 or result.shape[0] == 0:
            Pr_df.loc[0, col] = N
            continue
        pre = df_.shape[0]/result.shape[0]
        Pr_df.loc[0, col] = pre
    del df0, result, data, pre, df_, df1
    gc.collect()
    return Pr_df