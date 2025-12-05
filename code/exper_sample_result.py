import os
import pandas as pd
import re
import pickle
import numpy as np

csv_path = '/Users/admin/Documents/原文稿/jupyter/AAAI/experiment0720/data/total_data.csv'
expression_dict = {
        'Reduce proportion of idle vehicles': "sum((I[i][k] - sum(decision[f'cluster{i}_cluster{j}_{k}']for j in D)) for i in S for k in K)",
        'Reduce idle vehicles cost': "sum((k+1)*(I[i][k] - sum(decision[f'cluster{i}_cluster{j}_{k}']for j in D)) for i in S for k in K)",
        'Improve number of high-powered taxis in demand areas': "sum(decision[f'cluster{i}_cluster{j}_2'] for i in S for j in D)",
        'Improve future service Level of vehicles':"sum(k*decision[f'cluster{i}_cluster{j}_{k}'] for i in S for j in D for k in K)",
        'Reduce scheduled vehicle response time': "sum(decision[f'cluster{i}_cluster{j}_{k}'] *d[i][j]  for i in S for j in D for k in K)",
        'Reduce complaint rate of vehicles': "sum((k+1) * decision[f'cluster{i}_cluster{j}_{k}'] for i in S for j in D for k in K)",
        'Improve service Level of vehicles': "sum(decision[f'cluster{i}_cluster{j}_{k}'] *(k+1) for i in S for j in D for k in K)",
        'Reduce average travel price of vehicles': "sum(decision[f'avg_reward{j}_{k}'] for j in D for k in K)",
        'Improve order completion rate of vehicles': "sum((k+1) * decision[f'cluster{i}_cluster{j}_{k}'] for i in S for j in D for k in K) ",
        'Reduce average waiting time of vehicles': "sum(decision[f'cluster{i}_cluster{j}_{k}'] *d[i][j]  for i in S for j in D for k in K)",
        'Improve number of pre-allocated vehicles': "sum(decision[f'cluster{i}_cluster{j}_{k}'] for i in S for j in D for k in K)",
        'Improve average passenger capacity of vehicles': "sum((k+1)* decision[f'cluster{i}_cluster{j}_{k}'] for i in S for j in D for k in K)",
        'Improve number of users covered by vehicles': "sum((k+1)* decision[f'cluster{i}_cluster{j}_{k}'] for i in S for j in D for k in K)",
        'Improve user satisfaction of vehicles': "sum((k+1)* decision[f'cluster{i}_cluster{j}_{k}'] for i in S for j in D for k in K)",
        'Improve demand satisfaction rate': "sum((k+1)*decision[f'cluster{i}_cluster{j}_{k}'] for i in S for j in D for k in K)"
    }

index_opt_direction = {
    'Proportion of idle vehicles': -1,
    'Idle vehicles cost': -1,
    'Number of high-powered taxis in demand areas': 1,
    'Future service Level of vehicles': 1,
    'Future service Level of vehicles in demand areas':1,
    'Scheduled vehicle response time': -1,
    'Complaint rate of vehicles': 1,
    'Service Level of vehicles': 1,
    'Average travel price of vehicles': -1,
    'Order completion rate of vehicles': 1,
    'Average waiting time of vehicles':-1,
    'Number of pre-allocated vehicles':1,
    'Average passenger capacity of vehicles':1,
    'Number of users covered by vehicles':1,
    'User satisfaction of vehicles':1,
    'Demand satisfaction rate':1
}
global_dict = {'D': [4, 7, 12, 15, 37, 38, 44, 49],
               'K': [0,1,2]}
#'S': [0,1,2,3,5,6,8,9,10,11,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,39,40,41,42,43,45,46,47,48],


def generate_net_demand(area_num,demand_area):
    demand_avg = np.array([-1.34, -72.57, -427.42, -1.8, 11.87, -10.29, -2.72, 544.47, -1.67, -0.78, -0.41, -2.35, 392.5, -12.62,
                  -4.96,
                  70.81, -1.0, -4.46, -1.33, -9.61, -1.67, -3.6, -4.22, -1.14, -2.17, -3.28, -3.07, -4.08, -1.0, -0.94,
                  -9.84, -2.02,
                  -1.82, -1.04, -1.0, -1.79, -91.87, 36.66, 414.2, -0.46, -2.29, -0.84, -1.54, -5.08, 32.29, -1.35,
                  -1.44, -1.49,
                  -2.8, 18.66])
    demand_std = np.array([0.81, 48.52, 306.76, 1.64, 11.88, 5.19, 1.82, 259.58, 1.06, 0.65, 0.61, 1.55, 171.99, 5.53, 3.08,
                  42.4, 0.0, 3.09, 2.15, 4.98, 1.81, 4.77, 2.45, 2.19, 1.96, 9.07, 2.41, 2.86, 0.0, 0.56, 5.26, 1.9,
                  1.1, 0.6, 0.41, 1.67, 112.49, 112.69, 193.06, 0.76, 1.91, 0.37, 1.34, 3.42, 22.54, 0.96, 1.03, 2.11,
                  2.5, 12.25])
    taxi_storage = np.random.randint(8, 10, size=area_num)
    # reward_soc = np.random.normal([10, 20, 40], [2, 4, 8], (1, len(demand_area), num_soc))
    taxi_demand = np.random.normal(demand_avg, np.log(demand_std + 1), size=len(demand_avg))
    taxi_netdemand = taxi_demand - taxi_storage  # shape(cluster)
    net_demand_soc = []
    for i in range(area_num):
        if i in demand_area:
            taxi_netdemand[i] = max(taxi_netdemand[i], 0)
            p_soc = np.array([np.random.randint(0, 3), np.random.randint(2, 6), np.random.randint(5, 11)])
        else:
            taxi_netdemand[i] = min(taxi_netdemand[i], 0)
            p_soc = np.array([np.random.randint(0, 3), np.random.randint(2, 6), np.random.randint(3, 7)])
        p_soc = p_soc / sum(p_soc)
        net_demand_soc.append(taxi_netdemand[i, np.newaxis] * p_soc)
    # net_demand_soc (cluster,soc)
    net_demand_soc = np.stack(tuple(net_demand_soc), axis=0)
    return net_demand_soc


def match_decision(target_str):
    value_dict = {}
    x_result = re.findall("cluster\d+_cluster\d+_\d+ = \-?\d+\.\d+", target_str)
    for x in x_result:
        x = x.replace(" ", "")
        name_value = x.split('=')
        value_dict[name_value[0]] = float(name_value[1])
    r_result = re.findall("avg_reward\d+_\d+ = \-?\d+\.\d+", target_str)
    for u in r_result:
        u = u.replace(" ", "")
        name_value = u.split('=')
        value_dict[name_value[0]] = float(name_value[1])
    return value_dict


def cal_history_score(index, csv_path):
    data = pd.read_csv(csv_path)
    # data = data.rename(columns={f"{i}_I_{k}": f"u_{i}_{k}" for i in range(55) for k in range(3)})
    # data = data.rename(columns={f"{i}_exchange_{k}": f"x_{i}_exchange_{k}" for i in range(55) for k in range(2)})
    mean_data = data.mean(axis=0).to_dict()
    for key in mean_data:
        mean_data[key] = int(mean_data[key])
    global_dict['decision'] = mean_data
    score = eval(expression_dict[index], global_dict)
    return score


def cal_query_satisfaction(index, decision_dict):
    # value_dict={}
    global_dict['decision'] = decision_dict
    # value_dict["I"]=[9, 13, 16, 18, 31],
    # value_dict["C"]=[43, 36, 37, 39, 44, 39, 39, 39, 37, 39, 36, 36, 32, 42, 40, 35, 43, 37, 43, 37, 41, 40, 37, 32, 37,
    #                  31, 43, 32, 44, 42, 42, 31, 38, 40, 41, 39, 41, 40, 31, 39, 40, 35, 38, 39, 46, 37, 41, 38, 36, 41,
    #                  38, 31, 43, 38, 40]
    score = eval(expression_dict[index], global_dict)
    return score


if __name__ == '__main__':
    result_all_df = pd.DataFrame(
        columns=['sample','index', 'selected num', 'LEO-CPU time', 'Gurobi-CPU time','obj_gap1(%)','obj_gap2(%)'])
    for flag in ['insample', 'outsample']:
        original_file_folder = "exper_sample/" + flag
        for file in os.listdir(original_file_folder):
            if file.endswith(".txt"):
                # print(file)
                file_path = os.path.join(original_file_folder, file)
                f = open(file_path)  # 打开文件
                text = f.read()  # 将文件的内容读取出来赋值给t
                f.close()
                params = file.split('_')
                index = params[0][5:]
                position_1 = text.find('selected_num:')
                position_2 = text.find('selected decision')
                selected_num_text = text[position_1:position_2]
                selected_num = int(re.findall("\d+", selected_num_text)[0])
                p_LEO_time_text=text.find('LEO_optimize_time:')
                p_LEO_result_text = text.find('LEO_result:')
                LEO_time_text=text[p_LEO_time_text:p_LEO_result_text]
                LEO_time=float(re.findall("\d+\.?\d*", LEO_time_text)[0])
                p_gurobi_time_text=text.find('base_optimize_time:')
                p_gurobi_result_text = text.find('base_result:')
                gurobi_time_text=text[p_gurobi_time_text:p_gurobi_result_text]
                gurobi_time=float(re.findall("\d+\.?\d*", gurobi_time_text)[0])
                gap1_position = text.find('Optimization gap 1:')
                gap2_position = text.find('Optimization gap 2:')
                gap_end_position=text.find('LEO Model Variables and Values:')
                gap1_text = text[gap1_position:gap2_position]
                gap1_str = re.findall('\d+\.?\d*%', gap1_text)
                gap1 = float(gap1_str[0][:-1])
                gap2_text = text[gap2_position:gap_end_position]
                gap2_str = re.findall('\d+\.?\d*%', gap2_text)
                gap2 = float(gap2_str[0][:-1])
                result = [flag, index, selected_num, LEO_time,gurobi_time,gap1,gap2]
                result_all_df.loc[len(result_all_df)] = result
    result_all_df.to_csv('satisfaction_sample.csv', index=False, encoding="utf_8_sig")
