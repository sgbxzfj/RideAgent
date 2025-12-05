import numpy as np
import pandas as pd
import math
import random
import warnings
import os
from langchain_openai import ChatOpenAI
import gurobipy as gp
from gurobipy import GRB
import re 
import scipy as sp
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from operator import itemgetter
import traceback
from langchain_core.tools import tool
from langchain import PromptTemplate
import time
from langchain_core.messages import AIMessage, HumanMessage
import re
from collections import Counter
import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from jellyfish import jaro_winkler_similarity
import pickle
def extract_data_from_filename(filename):
    # 提取文件名中的中文内容和sample0
    parts = filename.split('_')
    index = parts[0]
    sample = parts[2]
    chinese_description = parts[1]
    return index, chinese_description, sample

def read_txt_files_and_extract_data(folder_path):
    # 读取文件夹中的所有txt文件并提取数据
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            index, chinese_description, sample = extract_data_from_filename(filename)
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                gurobi_code = extract_value_from_content(content, "temp_text:")
                
                data.append([index, chinese_description, sample, gurobi_code])
    return data

def extract_value_from_content(content, keyword):
    try:
        start = content.index(keyword) + len(keyword)
        end = content.index('\n', start)
        return content[start:end].strip()
    except ValueError:
        return None

def process_data(data):
    none_count = 0
    none_indices = []
    new_data = []

    for item in data:
        if item[-1] is None:
            none_count += 1
            none_indices.append(item[1])  # Assuming the "index后的汉字内容" is the first element
        else:
            new_data.append(item)  # Only add to new_data if the last item is not None

    print("Total None count:", none_count)
    print("Indices with None:", none_indices)
    from collections import Counter

    # 计算元素频次
    frequency = Counter(none_indices)
    print(frequency)
    return new_data, none_count, none_indices

def calculate_levenshtein_distance(code1, code2):
    distance = Levenshtein.distance(code1, code2)
    max_len = max(len(code1), len(code2))
    similarity = (max_len - distance) / max_len
    return similarity

def calculate_cosine_distance(expr1, expr2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([expr1, expr2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

def calculate_jaccard_distance(expr1, expr2):
    set1 = set(expr1.split())
    set2 = set(expr2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


def calculate_jaro_winkler_distance(expr1, expr2):
    return jaro_winkler_similarity(expr1, expr2)

def parse_expression(expression):
    # 去除空白并使用正则表达式提取带符号的系数和变量名
    tokens = re.findall(r'([+-]?\s*\d*\.?\d*\s*[a-z_]\w*)', expression.replace(' ', ''))

    # 分离系数和变量名，处理成 (变量名, 系数) 的形式
    parsed_tokens = []
    for token in tokens:
        # 匹配系数和变量名
        match = re.match(r'([+-]?)(\d*\.?\d*)\s*([a-z_]\w*)', token)
        if match:
            sign = match.group(1)
            coefficient = match.group(2) if match.group(2) else '1'  # 如果没有明确系数，默认为1
            variable = match.group(3)
            # 正确处理系数的符号和值
            full_coefficient = f"{sign}{coefficient}" if sign else coefficient
            parsed_tokens.append((variable, float(full_coefficient)))

    # 按变量名排序
    sorted_tokens = sorted(parsed_tokens, key=lambda x: x[0])

    # 重组表达式并添加操作符
    formatted_expression = ' + '.join(
        [f"{coeff} {var}" if coeff < 0 else f"{coeff} {var}" for var, coeff in sorted_tokens]
    ).strip()

    # 删除表达式开头的无意义的加号（如果存在）
    if formatted_expression.startswith('+'):
        formatted_expression = formatted_expression[1:].strip()

    return formatted_expression


def calculate_levenshtein_similarity(expr1, expr2):
    parsed_expr1 = parse_expression(expr1)
    parsed_expr2 = parse_expression(expr2)
    # print(parsed_expr1)
    # print(parsed_expr2)
    distance = Levenshtein.distance(parsed_expr1, parsed_expr2)
    max_len = max(len(parsed_expr1), len(parsed_expr2))
    similarity = (max_len - distance) / max_len
    return similarity

def calculate_cosine_similarity(expr1, expr2):
    parsed_expr1 = parse_expression(expr1)
    parsed_expr2 = parse_expression(expr2)
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([parsed_expr1, parsed_expr2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

def calculate_jaccard_similarity(expr1, expr2):
    parsed_expr1 = parse_expression(expr1)
    parsed_expr2 = parse_expression(expr2)
    set1 = set(parsed_expr1.split())
    set2 = set(parsed_expr2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

def calculate_jaro_winkler_similarity(expr1, expr2):
    parsed_expr1 = parse_expression(expr1)
    parsed_expr2 = parse_expression(expr2)
    return jaro_winkler_similarity(parsed_expr1, parsed_expr2)

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

def test_new_obj(formulation):
    import gurobipy as gp
    model = gp.Model()
    area_num = 50
    tree_num = 200
    booking_fee = 5
    theta = 0.2
    fee_pkm = 0.5
    inventory_avg = 9
    total_area = list(range(area_num))
    soc=[0,1,2]
    demand_area = [4, 7, 12, 15, 37, 38, 44, 49]
    supply_area = [0, 1, 2, 3, 5, 6, 8, 9, 10, 11, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                31, 32, 33, 34, 35, 36, 39, 40, 41, 42, 43, 45, 46, 47, 48]
    with open(r'/Users/admin/Documents/原文稿/jupyter/AAAI//experiment0720/distance_matrix.pickle', 'rb') as f:  #供给区到需求区的距离矩阵
        distance_matrix= pickle.load(f)
    w_hat = distance_matrix * fee_pkm  
    w = w_hat + booking_fee 
    demand_avg = np.array([-1.34, -72.57, -427.42, -1.8, 11.87, -10.29, -2.72, 544.47, -1.67, -0.78, -0.41, -2.35, 392.5, -12.62,
            -4.96,
            70.81, -1.0, -4.46, -1.33, -9.61, -1.67, -3.6, -4.22, -1.14, -2.17, -3.28, -3.07, -4.08, -1.0, -0.94,
            -9.84, -2.02,
            -1.82, -1.04, -1.0, -1.79, -91.87, 36.66, 414.2, -0.46, -2.29, -0.84, -1.54, -5.08, 32.29, -1.35,
            -1.44, -1.49,
            -2.8, 18.66])
    net_demand_soc = generate_net_demand(area_num,demand_area)
    # 添加变量
    x_decision_variables = model.addVars(supply_area,demand_area,soc,vtype=gp.GRB.INTEGER,lb=0,name=[[[f'cluster{s}_cluster{d}_{k}'for k in soc]for d in demand_area]for s in supply_area])  #预分配车辆数目的决策
    r_decision_variables = model.addVars(demand_area, soc, vtype=gp.GRB.CONTINUOUS,lb=0,name=[[f'avg_reward{d}_{k}'for k in soc]for d in demand_area])   # 平均每单客单价的决策,特征，r^
    model.update()

    # 设置目标函数
    exec(formulation, {'model': model, 'gp': gp, "GRB": gp.GRB, 'I': net_demand_soc, "b": booking_fee, "theta": theta, "fee_pkm": fee_pkm, "w_hat": w_hat, "w": w, "D":demand_area, "S":supply_area, "K":soc,"d":distance_matrix, "demand_avg":demand_avg, "inventory_avg":inventory_avg})
    model.update()

    # 保存原始目标函数
    new_obj = str(model.getObjective())
    model.dispose()
    return str(new_obj)

if __name__ == "__main__":
    
    # user_input = "reduce proportion of low-battery bicycles of available shared E-bikes at the site named 'Harvard Ave & E Pine St, E Pine St & 16th Ave, Summit Ave E & E Republican St, 2nd Ave & Pine St, REI / Yale Ave N & John St'"  # 输入的内容
    # index = '低电量单车占比'  # 要测试的指标，要是中文的，且和expression_dict中的key对应
    # In_sample = False  # 实验是否是in_sample的，True则从ground truth中抽样出index的gurobi语句和groundtruth_sample_num-1个随机指标
    # groundtruth_sample_num = 3

    # expression_dict = {
    #     'Proportion of idle vehicles': "model.setObjective(gp.quicksum((I[S.index(i)][K.index(k)] - gp.quicksum(model.getVarByName(f'cluster{i}_cluster{j}_{k}')for j in D)) for i in S for k in K), GRB.MINIMIZE)",
    #     'Idle vehicles cost': "model.setObjective(gp.quicksum((k+1)*(I[S.index(i)][K.index(k)] - gp.quicksum(model.getVarByName(f'cluster{i}_cluster{j}_{k}')for j in D)) for i in S for k in K), GRB.MINIMIZE)",
    #     'Number of high-powered taxis in demand areas': "model.setObjective(gp.quicksum(model.getVarByName(f'cluster{i}_cluster{j}_2') for i in S for j in D) , GRB.MAXIMIZE)",
    #     'Number of low-powered taxis in demand areas':"model.setObjective(gp.quicksum(k * model.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in S for j in D for k in K) , GRB.MAXIMIZE)",
    #     'Scheduled vehicle response time': "model.setObjective(gp.quicksum(model.getVarByName(f'cluster{i}_cluster{j}_{k}') * d[S.index(i)][D.index(j)]  for i in S for j in D for k in K), GRB.MINIMIZE)",
    #     'Complaint rate of vehicles': "model.setObjective(gp.quicksum((k+1) * model.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in S for j in D for k in K), GRB.MAXIMIZE)",
    #     'Service Level of vehicles': "model.setObjective(gp.quicksum((k+1) * model.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in S for j in D for k in K), GRB.MAXIMIZE)",
    #     'Average travel price of vehicles': "model.setObjective(gp.quicksum(model.getVarByName(f'avg_reward{j}_{k}') for j in D for k in K), GRB.MINIMIZE)",
    #     'Order completion rate of vehicles': "model.setObjective(gp.quicksum((k+1) * model.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in S for j in D for k in K), GRB.MAXIMIZE) ",
    #     'Average waiting time of vehicles': "model.setObjective(gp.quicksum(model.getVarByName(f'cluster{i}_cluster{j}_{k}') * d[S.index(i)][D.index(j)] for i in S for j in D for k in K), GRB.MINIMIZE)",
    #     'Number of pre-allocated vehicles': "model.setObjective(gp.quicksum(model.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in S for j in D for k in K), GRB.MAXIMIZE)",
    #     'Average passenger capacity of vehicles': "model.setObjective(gp.quicksum((k+1) * model.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in S for j in D for k in K), GRB.MAXIMIZE)",
    #     'Number of users covered by vehicles': "model.setObjective(gp.quicksum((k+1) * model.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in S for j in D for k in K), GRB.MAXIMIZE)",
    #     'User satisfaction of vehicles': "model.setObjective(gp.quicksum((k+1) * model.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in S for j in D for k in K), GRB.MAXIMIZE)",
    #     'Demand satisfaction rate': "model.setObjective(gp.quicksum((k+1) * model.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in S for j in D for k in K), GRB.MAXIMIZE)"
    # }
    expression_dict = {
        'Dispatching efficiency of vehicles': "model.setObjective(gp.quicksum(model.getVarByName(f'avg_reward{j}_{k}') * model.getVarByName(f'cluster{i}_cluster{j}_{k}') - w[S.index(i)][D.index(j)] * model.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in S for j in D for k in K), GRB.MAXIMIZE)",
        'Supply-demand matching degree of vehicles': "model.setObjective(gp.quicksum((gp.quicksum(I[S.index(i)][K.index(k)] - gp.quicksum(model.getVarByName(f'cluster{i}_cluster{j}_{k}')for j in D) for k in K ))**2 for i in S) + gp.quicksum((demand_avg[D.index(j)] - inventory_avg - gp.quicksum(model.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in S for k in K))**2 for j in D), GRB.MINIMIZE)",
        'Market share of vehicles': "model.setObjective(gp.quicksum((model.getVarByName(f'avg_reward{j}_{k}') - w[S.index(i)][D.index(j)]) * model.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in S for j in D for k in K), GRB.MAXIMIZE)"
        }

    new_directory = "/Users/admin/Documents/原文稿/jupyter/AAAI/exper_similarity"
    insample_data = read_txt_files_and_extract_data(os.path.join(new_directory, "insample_nonlinear"))
    outsample_data = read_txt_files_and_extract_data(os.path.join(new_directory, "outsample_nonlinear"))

    # 添加标签区分insample和outsample
    insample_data = [['insample'] + row for row in insample_data]
    outsample_data = [['outsample'] + row for row in outsample_data]

    # 合并数据
    all_data = insample_data + outsample_data
    new_data, none_count, none_indices = process_data(all_data)
    #['insample', 'index降低碳排放量', 'sample6', '1715223307.76536.txt', "model.setObjective(gp.quicksum((2 - k) * model.getVarByName(f'x_{i}_dispatch_{k}') for i in range(55) for k in [0, 1, 2]), GRB.MAXIMIZE)"]
    df_results = pd.DataFrame(columns=["Type", "Index", "Optimility Similarity", "Text Similarity"])

    #print(all_data[0])
    for part in all_data:
        index = str(part[1])[5:]
        description = str(part[2]) 
        Type = part[0]
        txt_name = part[3]
        print('type:',Type,'description:',description,',index:',index, 'txt_name:',txt_name)
        gurobi_code1 = str(part[4])
        gurobi_code2 = expression_dict[index]
        result1 = test_new_obj(gurobi_code1)
        #print('test code:',result1)
        result2 = test_new_obj(gurobi_code2)
        #print('gt code:',result2)
        # result_similarity_levenshtein = calculate_levenshtein_similarity(result1, result2)
        # #print("Result Similarity:", result_similarity)
        # result_similarity_cosine = calculate_cosine_similarity(result1, result2)
        # result_similarity_jaccard = calculate_jaccard_similarity(result1, result2)
        result_similarity_jaro = calculate_jaro_winkler_similarity(result1, result2)
        # text_similarity_levenshtein = calculate_levenshtein_distance(gurobi_code1, gurobi_code2)
        # text_similarity_cosine = calculate_cosine_distance(gurobi_code1, gurobi_code2)
        # text_similarity_jaccard = calculate_jaccard_distance(gurobi_code1, gurobi_code2)
        text_similarity_jaro = calculate_jaro_winkler_distance(gurobi_code1, gurobi_code2)
        #print("Text Similarity:", text_similarity)
        new_row = pd.DataFrame({
        "Type": [Type],
        'description':[description],
        "Index": [index],
        'txt_name': [txt_name],
        # "resut Similarity levenshtein": [result_similarity_levenshtein],
        # "resut Similarity cosine": [result_similarity_cosine],
        # "resut Similarity jaccard": [result_similarity_jaccard],
        "resut Similarity jaro": [result_similarity_jaro],
        # "Text Similarity levenshtein": [text_similarity_levenshtein],
        # "Text Similarity cosine": [text_similarity_cosine],
        # "Text Similarity jaccard": [text_similarity_jaccard],
        "Text Similarity jaro": [text_similarity_jaro]
        })
        df_results = pd.concat([df_results, new_row], ignore_index=True)
    # grouped = df_results.groupby(['Type', 'description']).agg({
    #     # 'resut Similarity levenshtein':'mean',
    #     # 'resut Similarity cosine':'mean',
    #     # 'resut Similarity jaccard':'mean',
    #     'resut Similarity jaro':'mean',
    #     # 'Text Similarity levenshtein':'mean',
    #     # 'Text Similarity cosine':'mean',
    #     # 'Text Similarity jaccard':'mean',
    #     'Text Similarity jaro':'mean'
    # }).reset_index()
    # 分组并聚合
    grouped = df_results.groupby(['Type', 'description']).agg({
        'resut Similarity jaro': ['mean', 'std'],
        'Text Similarity jaro': ['mean', 'std']
    }).reset_index()

    # 扁平化列名
    grouped.columns = [
        col[0] if col[1] == '' else f"{col[0]}_{col[1]}"
        for col in grouped.columns.values
    ]

    # 创建Excel写入器
    writer = pd.ExcelWriter("similarity_result_nonlinear.xlsx", engine='xlsxwriter')

    # 写入原始数据
    df_results.to_excel(writer, sheet_name='similarity_result', index=False)

    # 写入分组统计结果
    grouped.to_excel(writer, sheet_name='similarity_groupby', index=False)

    # 保存
    writer.close()  # 注意：在较新版本 pandas 中，应使用 close() 而不是 save()
    os.system(f'say "Finish"')