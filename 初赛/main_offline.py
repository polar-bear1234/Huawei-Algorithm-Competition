from anomaly_find import birch_ad_with_smoothing
from root_causal import*
import csv


alpha = 0.55
ad_threshold = 0.045

folders = ['2', '3', '4', '5']
faults_type = ['orders_carts', 'front-end_orders', 'orders_payment', 'orders_shipping']
targets = ['front-end', 'catalogue', 'orders', 'user', 'carts', 'payment', 'shipping']

for folder in folders:
    for fault_type in faults_type:
        for target in targets:
            if target == 'front-end' and fault_type != 'svc_latency':
                continue
            faults_name = folder
            latency_df = rt_invocations(faults_name)
            if (target == 'payment' or target == 'shipping') and fault_type != 'svc_latency':
                threshold = 0.02
            else:
                threshold = ad_threshold
            anomalies = birch_ad_with_smoothing(latency_df, threshold)

            # get the anomalous service
            anomaly_nodes = []
            for anomaly in anomalies:
                edge = anomaly.split('_')
                anomaly_nodes.append(edge[1])
            anomaly_nodes = set(anomaly_nodes)

            # construct attributed graph
            DG = attributed_graph(faults_name)
            anomaly_score = anomaly_subgraph(DG, anomalies, latency_df, faults_name, alpha)
            print(anomaly_score)
            anomaly_score_new = []
            for anomaly_target in anomaly_score:
                if anomaly_target[0] in targets:
                    anomaly_score_new.append(anomaly_target)
            num = print_rank(anomaly_score_new, target)
            print("num----------------------------------:", num)

            result = pd.DataFrame(anomaly_score_new, columns=['Micro_name', 'Root_causal_score'])
            result.sort_values(by=['Root_causal_score'], ascending=False, inplace=True)
            result['flag'] = 0
            result['describe'] = '运行成功'
            print("根因定位结果----------------------------：", "\n", result)

            if num < eval(folders[2]):
                filename = 'output_offline/MicroRCA_results.csv'
                with open(filename, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([folder, target, fault_type, num, anomaly_score_new[:num], anomaly_nodes])


output_result = pd.read_csv('output_offline/MicroRCA_results.csv')
prk1 = prk2(output_result, 2, 3)
# prk2 = prk2(output_result, 2, 3)
# prk3 = prk2(output_result, 3, 3)

print(prk1)
# print(prk2)
# print(prk3)

