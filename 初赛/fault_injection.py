"""
该脚本使用http请求chaos mesh平台实现自动化的故障注入
"""

import requests
import json
import time
import random
import pandas as pd
from datetime import datetime

sevice = 'http://47.103.63.176:31984'

program = ['front-end', 'catalogue', 'carts', 'orders', 'payment', 'shipping', 'user']
log = pd.DataFrame(columns=['类别','位置','状态','开始时间','结束时间'])

"""
cpu故障脚本 
"""
cpu_fj = {
    "kind": "StressChaos",
    "apiVersion": "chaos-mesh.org/v1alpha1",
    "metadata": {
        "namespace": "sock-shop",
        "name": "stress-cpu3"
    },
    "spec": {
        "selector": {
            "namespaces": [
                "sock-shop"
            ],
            "labelSelectors": {
                "name": "user"
            }
        },
        "mode": "all",
        "stressors": {
            "cpu": {
                "workers": 5,
                "load": 95
            }
        },
        "duration": "2m"
    }
}

"""
内存故障脚本
"""
mem_fj = {
    "kind": "StressChaos",
    "apiVersion": "chaos-mesh.org/v1alpha1",
    "metadata": {
        "namespace": "sock-shop",
        "name": "stress-mem2"
    },
    "spec": {
        "selector": {
            "namespaces": [
                "sock-shop"
            ],
            "labelSelectors": {
                "name": "front-end"
            }
        },
        "mode": "all",
        "stressors": {
            "memory": {
                "workers": 2,
                "size": "2G"
            }
        },
        "duration": "5m"
    }
}

"""
pod故障脚本
"""
pod_fj = {
    "kind": "PodChaos",
    "apiVersion": "chaos-mesh.org/v1alpha1",
    "metadata": {
        "namespace": "sock-shop",
        "name": "chaos-pod-kill"
    },
    "spec": {
        "selector": {
            "namespaces": [
                "sock-shop"
            ],
            "labelSelectors": {
                "name": "front-end"
            }
        },
        "mode": "all",
        "action": "pod-kill",
        "gracePeriod": 0
    }
}

"""
网络故障脚本
"""
net_fj = {
    "kind": "NetworkChaos",
    "apiVersion": "chaos-mesh.org/v1alpha1",
    "metadata": {
        "namespace": "sock-shop",
        "name": "fj-netdelay"
    },
    "spec": {
        "selector": {
            "namespaces": [
                "sock-shop"
            ],
            "labelSelectors": {
                "name": "catalogue"
            }
        },
        "mode": "all",
        "action": "delay",
        "duration": "5m",
        "delay": {
            "latency": "100ms",
            "correlation": "50",
            "jitter": "10ms"
        },
        "direction": "to"
    }
}


def fault_injection_exec(data):
    """
    执行故障注入
    :param data:
    :return:
    """
    url = sevice+'/api/experiments'
    headers = {
        'Content-Type': 'application/json'
    }
    response = http_exec("POST", url, headers, data)
    # print(response.text)
    return response


def fault_injection_delete(uid):
    """
    归档以及完成的故障注入
    :param uid:
    :return:
    """
    url = sevice+'/api/experiments/' + uid
    headers = {}
    response = http_exec("DELETE", url, headers, {})
    return response


def fault_injection_get(uid):
    """
    归档以及完成的故障注入
    :param uid:
    :return:
    """
    url = sevice+'/api/experiments/' + uid
    headers = {}
    response = http_exec("GET", url, headers, {})
    # print(response.text)
    return response


def fault_injection_list():
    """
    列出chaos mesh平台实验故障注入
    :return:
    """
    url = sevice+'/api/experiments'
    headers = {}
    response = http_exec("GET", url, headers, {})
    # print(response.text)
    return response


def http_exec(method, url, headers, data):
    """
    执行http请求
    :param method: 请求方法，get，post，delete
    :param url: 服务地址
    :param headers: 请求头
    :param data: body数据
    :return: response
    """
    # noinspection PyBroadException
    try:
        response = requests.request(method, url, headers=headers, data=data)
    except Exception:
        print("{}请求服务失败！".format(url))
        return None
    return response


def deal_finiashed_fault_injection():
    """
    删除所有的实验故障注入，否侧有重复的时候无法注入
    :return:
    """
    response = fault_injection_list()
    if response is None or response.status_code != 200:
        return -1
    if len(response.text) > 0:
        experiments = json.loads(response.text)
        for i in experiments:
            uid = i["uid"]
            state = i["status"]
            if state == 'finished':
                fault_injection_delete(uid)
    return 0


def falt_injection_exec(type):
    """
    执行故障注入，并判断故障执行的状态，执行完成返回
    :param type: 故障注入的类型
    :return: -1 失败，-2 故障执行后未获取到信息， 0 成功
    """
    if deal_finiashed_fault_injection() < 0:
        print("归档失败不进行故障注入！")
        return -1

    rep = fault_injection_exec(type)
    if rep is None or rep.status_code != 200:
        return -1

    rep1 = fault_injection_list()
    if rep1 is None or rep1.status_code != 200:
        return -1

    if len(rep1.text) < 0:
        return -1

    experiments = json.loads(rep1.text)
    uid = ''
    for i in experiments:
        name = i["name"]
        if name == json.loads(type)['metadata']['name']:
            uid = i['uid']

    if len(uid) <= 0:
        return -2

    # 去获取注入故障的信息，如果故障状态为完成则退出，这个地方不会死循环，故障注入如果超时也算完成
    timeout = 0
    while 1:
        # 状态查询失败10次则退出
        if timeout > 10:
            return -2

        rep2 = fault_injection_get(uid)
        if rep2 is None or rep2.status_code != 200:
            time.sleep(0.5)
            timeout += 1
            continue

        if len(rep2.text) < 0:
            time.sleep(0.5)
            timeout += 1
            continue

        status = json.loads(rep2.text)['status']
        if status == 'finished':
            return 0
        if len(status) > 0:
            print('请等待……故障状态：{}'.format(status))
        time.sleep(0.5)


def fault_injection_cpu():
    """
    cup 故障注入
    :return:
    """
    return falt_injection_exec(json.dumps(cpu_fj))


def fault_injection_mem():
    """
    内存故障注入
    :return:
    """
    return falt_injection_exec(json.dumps(mem_fj))


def fault_injection_pod():
    """
    pod故障注入
    :return:
    """
    return falt_injection_exec(json.dumps(pod_fj))


def fault_injection_net():
    """
    网络故障注入
    :return:
    """
    return falt_injection_exec(json.dumps(net_fj))


def fault_injection_switch(num):
    fj_numbers = {
        1: fault_injection_cpu,
        2: fault_injection_mem,
        3: fault_injection_pod,
        4: fault_injection_net
    }
    fault_injection = fj_numbers.get(num)
    if fault_injection:
        start_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        ret = fault_injection()
        end_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        state = '注入成功！'
        time_now = time.time()  # 改过
        if ret < 0:
            state = '注入失败！'
        print_fj_state(fault_injection.__name__, state, start_time, end_time)
        return time_now

    random_fj_number = list(fj_numbers.keys())
    random.shuffle(random_fj_number)

    random_fault_injection(random_fj_number)


def random_fault_injection(random_fj_list):
    fj_numbers = {
        1: fault_injection_cpu,
        2: fault_injection_mem,
        3: fault_injection_pod,
        4: fault_injection_net
    }
    fj_names = {
        1: cpu_fj,
        2: mem_fj,
        3: pod_fj,
        4: net_fj
    }
    for (i, j) in random_fj_list:
        fault_injection = fj_numbers.get(i)
        fault_name = fj_names.get(i)
        if fault_injection:
            # 重写fault的yaml文件
            fault_name['spec']['selector']['labelSelectors']['name'] = program[j-1]
            print('开始自动化注入{}'.format(fault_injection.__name__))
            print('注入位置：' + program[j - 1])
            start_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            ret = fault_injection()
            end_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            state = '注入成功！'
            if ret < 0:
                state = '注入失败！'
            print_fj_state(fault_injection.__name__, state, start_time, end_time)
            save_fj_state(fault_injection.__name__, state, start_time, end_time, program[j-1])
            print('注入位置：'+program[j-1])
            print('等待30分钟后继续注入故障…………')
        time.sleep(15*60)


def print_fj_state(name,state, start_time, end_time):
    print('/*************************************\n'
          '{}故障注入完成\n'
          '注入状态：{}\n'
          '故障注入开始时间：{}\n'
          '故障注入完成时间：{}\n'
          '**************************************/'.format(name, state, start_time, end_time))


def save_fj_state(name, state, start_time, end_time, position):
    global log
    log = log.append([{
        '类别': name,
        '位置': position,
        '状态': state,
        '开始时间': start_time,
        '结束时间': end_time
    }])
    log.to_csv('./log.csv', encoding="utf_8_sig", index=False)


def main():
    fault_time_list = []
    while 1:
        print('/*************************************\n'
              '故障注入类型如下：\n '
              '1、cpu故障注入\n '
              '2、内存故障注入\n '
              '3、pod故障注入\n '
              '4、网络故障注入\n '
              '5、4种故障随机顺序注入一次（参数自动生成）\n '
              '6、随机顺序随机位置注入故障\n '
              '0、退出程序\n'
              '**************************************/')
        print('请输入需要注入的故障类型：\n')
        num = int(input())
        if num < 0 or num > 6:
            print('参数{}输入错误！'.format(num))
            return fault_time_list
        if num == 0:
            print('程序退出！')
            return fault_time_list
        if num == 6:
            random_fj_list = [[a,b] for a in range(1,5) for b in range(1,8)]
            random.shuffle(random_fj_list)
            random_fault_injection(random_fj_list)

            return fault_time_list
        time1 = time.time()
        time2 = fault_injection_switch(num)
        fault_time_list.append([time1, time2])
        print('\n')


if __name__ == '__main__':
    main()
