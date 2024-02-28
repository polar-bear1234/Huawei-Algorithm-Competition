import requests
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

metric_step = '60s'
smoothing_window = 12

node_dict = {'k8s-master': '192.168.0.36:9100',
             'k8s-node': '192.168.0.37:9100',
             'k8s-node1': '192.168.0.39:9100',
             'k8s-node2': '192.168.0.38:9100'}


def latency_source_50(prom_url, start_time, end_time, faults_name):
    latency_df = pd.DataFrame()
    response = requests.get(prom_url,
                            params={'query': 'histogram_quantile(0.50, sum(irate(istio_request_duration_milliseconds_bucket{reporter=\"source\", destination_workload_namespace=\"sock-shop\"}[1m])) by (destination_workload, source_workload, le))',
                                    'start': start_time,
                                    'end': end_time,
                                    'step': metric_step})
    results = response.json()['data']['result']
    for result in results:
        dest_svc = result['metric']['destination_workload']
        src_svc = result['metric']['source_workload']
        name = src_svc + '_' + dest_svc
        values = result['values']
        values = list(zip(values))
        if 'timestamp' not in latency_df:
            timestamp = [val[0][0] for val in values]
            latency_df['timestamp'] = timestamp
            latency_df['timestamp'] = latency_df['timestamp'].astype('datetime64[s]')
        metric = [val[0][1] for val in values]
        latency_df[name] = pd.Series(metric)
        latency_df[name] = latency_df[name].astype('float64') * 1000
    response = requests.get(prom_url,
                            params={'query': 'sum(irate(istio_tcp_sent_bytes_total{reporter=\"source\", destination_workload!=\'unknown\', source_workload!=\'unknown\'}[1m])) by (destination_workload, source_workload) / 1000',
                                    'start': start_time,
                                    'end': end_time,
                                    'step': metric_step})
    results = response.json()['data']['result']
    for result in results:
        dest_svc = result['metric']['destination_workload']
        src_svc = result['metric']['source_workload']
        name = src_svc + '_' + dest_svc
        values = result['values']
        if 'timestamp' not in latency_df:
            timestamp = [val[0] for val in values]
            latency_df['timestamp'] = timestamp
            latency_df['timestamp'] = latency_df['timestamp'].astype('datetime64[s]')
        metric = [val[1] for val in values]
        latency_df[name] = pd.Series(metric)
        latency_df[name] = latency_df[name].astype('float64').rolling(window=smoothing_window, min_periods=1).mean()
    latency_df.fillna(0, inplace=True)                        # 加的
    filename = 'Dataset/' + faults_name + '_latency_source_50.csv'
    latency_df.set_index('timestamp')
    latency_df.to_csv(filename)
    print("第一个函数保存的文件为-------------------------：", filename)
    return latency_df


def latency_destination_50(prom_url, start_time, end_time, faults_name):
    latency_df = pd.DataFrame()
    response = requests.get(prom_url,
                            params={'query': 'histogram_quantile(0.50, sum(irate(istio_request_duration_milliseconds_bucket{reporter=\"destination\",destination_workload!=\'unknown\', source_workload!=\'unknown\', destination_workload_namespace=\"sock-shop\"}[1m])) by (destination_workload, source_workload, le))',
                                    'start': start_time,
                                    'end': end_time,
                                    'step': metric_step})
    results = response.json()['data']['result']
    for result in results:
        dest_svc = result['metric']['destination_workload']
        src_svc = result['metric']['source_workload']
        name = src_svc + '_' + dest_svc
        values = result['values']
        values = list(zip(*values))
        if 'timestamp' not in latency_df:
            timestamp = values[0]
            latency_df['timestamp'] = timestamp
            latency_df['timestamp'] = latency_df['timestamp'].astype('datetime64[s]')
        metric = values[1]
        latency_df[name] = pd.Series(metric)
        latency_df[name] = latency_df[name].astype('float64') * 1000
    response = requests.get(prom_url,
                            params={'query': 'sum(irate(istio_tcp_sent_bytes_total{reporter=\"destination\", destination_workload!=\'unknown\', source_workload!=\'unknown\'}[1m])) by (destination_workload, source_workload) / 1000',
                                    'start': start_time,
                                    'end': end_time,
                                    'step': metric_step})
    results = response.json()['data']['result']
    for result in results:
        dest_svc = result['metric']['destination_workload']
        src_svc = result['metric']['source_workload']
        name = src_svc + '_' + dest_svc
        values = result['values']
        values = list(zip(*values))
        if 'timestamp' not in latency_df:
            timestamp = values[0]
            latency_df['timestamp'] = timestamp
            latency_df['timestamp'] = latency_df['timestamp'].astype('datetime64[s]')
        metric = values[1]
        latency_df[name] = pd.Series(metric)
        latency_df[name] = latency_df[name].astype('float64').rolling(window=smoothing_window, min_periods=1).mean()
    latency_df.fillna(0, inplace=True)
    filename = 'Dataset/' + faults_name + '_latency_destination_50.csv'
    latency_df.set_index('timestamp')
    latency_df.to_csv(filename)
    print("第二个函数保存的文件为------------------------------：", filename)
    return latency_df


def svc_metrics(prom_url, start_time, end_time, faults_name):
    response = requests.get(prom_url,
                            params={
                                'query': 'sum(rate(container_cpu_usage_seconds_total{namespace="sock-shop", container!~\'POD|istio-proxy|\'}[1m])) by (pod, instance, container)',
                                'start': start_time,
                                'end': end_time,
                                'step': metric_step})
    results = response.json()['data']['result']
    for result in results:
        df = pd.DataFrame()
        svc = result['metric']['container']
        pod_name = result['metric']['pod']
        nodename = result['metric']['instance']
        values = result['values']
        values = list(zip(*values))
        if 'timestamp' not in df:
            timestamp = values[0]
            df['timestamp'] = timestamp
            df['timestamp'] = df['timestamp'].astype('datetime64[s]')
        metric = pd.Series(values[1])
        df['ctn_cpu'] = metric
        df['ctn_cpu'] = df['ctn_cpu'].astype('float64')
        df['ctn_network'] = ctn_network(prom_url, start_time, end_time, pod_name)
        df['ctn_network'] = df['ctn_network'].astype('float64')
        df['ctn_memory'] = ctn_memory(prom_url, start_time, end_time, pod_name)
        df['ctn_memory'] = df['ctn_memory'].astype('float64')
        instance = node_dict[nodename]
        df_node_cpu = node_cpu(prom_url, start_time, end_time, instance)
        df = pd.merge(df, df_node_cpu, how='left', on='timestamp')
        df_node_network = node_network(prom_url, start_time, end_time, instance)
        df = pd.merge(df, df_node_network, how='left', on='timestamp')
        df_node_memory = node_memory(prom_url, start_time, end_time, instance)
        df = pd.merge(df, df_node_memory, how='left', on='timestamp')
        filename = 'Dataset/' + faults_name + '_' + svc + '.csv'
        df.set_index('timestamp')
        df.to_csv(filename)
        print("svc保存的文件为--------------------------------保存成功：", filename)


def ctn_network(prom_url, start_time, end_time, pod):
    response = requests.get(prom_url,
                            params={
                                'query': 'sum(rate(container_network_transmit_packets_total{namespace="sock-shop", pod="%s"}[1m])) / 1000 * sum(rate(container_network_transmit_packets_total{namespace="sock-shop", pod="%s"}[1m])) / 1000' % (
                                pod, pod),
                                'start': start_time,
                                'end': end_time,
                                'step': metric_step})
    results = response.json()['data']['result']
    values = results[0]['values']
    values = list(zip(*values))
    metric = pd.Series(values[1])
    return metric


def ctn_memory(prom_url, start_time, end_time, pod):
    response = requests.get(prom_url,
                            params={
                                'query': 'sum(rate(container_memory_working_set_bytes{namespace="sock-shop", pod="%s"}[1m])) / 1000 ' % pod,
                                'start': start_time,
                                'end': end_time,
                                'step': metric_step})
    results = response.json()['data']['result']
    values = results[0]['values']
    values = list(zip(*values))
    metric = pd.Series(values[1])
    return metric


def node_network(prom_url, start_time, end_time, instance):
    response = requests.get(prom_url,
                            params={'query': 'rate(node_network_transmit_packets_total{instance="%s"}[1m]) / 1000' % instance,
                                    'start': start_time,
                                    'end': end_time,
                                    'step': metric_step})
    results = response.json()['data']['result']
    values = results[0]['values']
    values = list(zip(*values))
    df = pd.DataFrame()
    df['timestamp'] = values[0]
    df['timestamp'] = df['timestamp'].astype('datetime64[s]')
    df['node_network'] = pd.Series(values[1])
    df['node_network'] = df['node_network'].astype('float64')
    return df


def node_cpu(prom_url, start_time, end_time, instance):
    response = requests.get(prom_url,
                            params={
                                'query': 'sum(rate(node_cpu_seconds_total{mode != "idle",mode!= "iowait",mode!~"^(?:guest.*)$",instance="%s" }[1m]))/count(node_cpu_seconds_total{mode="system",instance="%s"})' % (instance, instance),
                                'start': start_time,
                                'end': end_time,
                                'step': metric_step})
    results = response.json()['data']['result']
    values = results[0]['values']
    values = list(zip(*values))
    df = pd.DataFrame()
    df['timestamp'] = values[0]
    df['timestamp'] = df['timestamp'].astype('datetime64[s]')
    df['node_cpu'] = pd.Series(values[1])
    df['node_cpu'] = df['node_cpu'].astype('float64')
    return df


def node_memory(prom_url, start_time, end_time, instance):
    response = requests.get(prom_url,
                            params={
                                'query': '1 - sum(node_memory_MemAvailable_bytes{instance="%s"}) / sum(node_memory_MemTotal_bytes{instance="%s"})' % (
                                instance, instance),
                                'start': start_time,
                                'end': end_time,
                                'step': metric_step})
    results = response.json()['data']['result']
    values = results[0]['values']
    values = list(zip(*values))
    df = pd.DataFrame()
    df['timestamp'] = values[0]
    df['timestamp'] = df['timestamp'].astype('datetime64[s]')
    df['node_memory'] = pd.Series(values[1])
    df['node_memory'] = df['node_memory'].astype('float64')
    return df