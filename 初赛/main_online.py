from data_extract import latency_source_50
from data_extract import latency_destination_50
from data_extract import svc_metrics
from anomaly_find import birch_ad_with_smoothing
from root_causal import*
from fault_injection import main


'''Begainning---------------------------------'''

# ============================================注入故障
fault_time_list = main()
print("产生故障时间区间分别为----------------------:", "\n", fault_time_list)
# (注：故障注入停止之后才会进行下面程序)

args = parse_args()
folder = args.folder
len_second = args.length
prom_url = args.url
faults_name = folder

# 设置采集数据时间范围
end_time = time.time()
start_time = end_time - len_second

# end_time = 1658528160
# start_time = 1658488320

# Tuning parameters
alpha = 0.55
ad_threshold = 0.045


# ========================================采集时延数据
latency_df_source = latency_source_50(prom_url, start_time, end_time, faults_name)
latency_df_destination = latency_destination_50(prom_url, start_time, end_time, faults_name)
latency_df = latency_df_destination.append(latency_df_source)
latency_df.to_csv("Dataset/labency_df.csv", index=False)
print("soruce + destination-----shape为-------------：", latency_df.shape)


# ========================================微服务异常数据采集并保存
svc_metrics(prom_url, start_time, end_time, faults_name)


# ========================================绘制图
DG = mpg(prom_url, faults_name, start_time, end_time)


# ========================================异常检测
smoothing_window = 12
anomalies = birch_ad_with_smoothing(latency_df, ad_threshold)
anomaly_nodes = []
for anomaly in anomalies:
    edge = anomaly.split('_')
    anomaly_nodes.append(edge[1])
anomaly_nodes = set(anomaly_nodes)


# ========================================提取异常子图并且根因排序
anomaly_score = anomaly_subgraph(DG, anomalies, latency_df, faults_name, alpha)
anomaly_score_new = []
for anomaly_target in anomaly_score:
    node = anomaly_target[0]
    if DG.nodes[node]['type'] == 'service':
        anomaly_score_new.append(anomaly_target)
num = print_rank(anomaly_score_new, anomaly_score)
print("Topk---------------------------------:", "\n", num)

result = pd.DataFrame(anomaly_score_new, columns=['Micro_name', 'Root_causal_score'])
result.sort_values(by=['Root_causal_score'], ascending=False, inplace=True)
result['flag'] = 0
result['describe'] = '运行成功'
print("根因定位结果----------------------------：", "\n", result)



