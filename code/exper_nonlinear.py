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
import sys
import pickle
from langchain.agents import AgentExecutor, create_openai_tools_agent


os.environ["OPENAI_API_KEY"] = ''
llm1 = ChatOpenAI(model="gpt-4o",temperature=0.8)  
llm2 = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.8)  
llm3 = ChatOpenAI(model="gpt-4o-mini",temperature=0.8)  

def process_string_and_add_constraints(input_string, model, csv_directory):
    # Path to the directory containing CSV files
    # csv_directory = '/Users/admin/Documents/jupyter/AAAI/experiment0720/split'
    
    matches = re.findall(r'「(.*?)」', input_string)
    
    # 如果找到了匹配的内容，返回最后一个匹配
    if matches:
        cleaned_string = matches[-1]
    # Remove special characters and extra spaces
    cleaned_string = cleaned_string.replace('「', '').replace('」', '').strip()
    cleaned_string = cleaned_string.replace(' ', '').strip()
    # print(cleaned_string)
    # Updated pattern to match the cleaned string
    pattern = r'(\w+\d+):\[(\d+(?:,\d+)*)\]'
    matches = re.findall(pattern, cleaned_string)
    
    if not matches:
        print("No matches found.")
        sys.exit(1) 
    converted_variables = {}
    
    for key, values in matches:
        # Split values into a list of integers
        values = map(int, values.split(','))
        
        for value in values:
            csv_path = f'{csv_directory}/{key}.csv'
            
            try:
                # Read the CSV file
                df = pd.read_csv(csv_path)
                
                # Get the column name and data for the given index
                if value < len(df.columns):
                    column_name = df.columns[value]
                    column_data = df[column_name]
                    
                    # Calculate the mean value
                    mean_value = column_data.mean()
                    
                    # Adjust the mean value based on the file name
                    if 'cluster' in key:
                        mean_value = math.floor(mean_value)
                    
                    # Add to the dictionary
                    converted_variables[column_name] = mean_value
                    # print('column_name', column_name, 'mean_value', mean_value)  
            except FileNotFoundError:
                pass
                # print(f"File {csv_path} not found.")
            except IndexError:
                pass
                # print(f"Index {value} is out of range in file {csv_path}.")
    
    # Add constraints to the Gurobi model
    num_constraints = len(converted_variables)
    # print(num_constraints)
    for variable, value in converted_variables.items():
        model = add_equal_constr(model, variable, value)

    return model,num_constraints,cleaned_string
def generate_net_demand(area_num,demand_area):
    random.seed(0)
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

def variables_values(variables, csv_path):
    data = pd.read_csv(csv_path)
    result = {}

    for var in variables:
        if var in data.columns:
            # 获取列数据
            column_data = data[var]
            
            # 计算均值
            mean_value = column_data.mean()
            
            # 根据变量名处理均值
            if 'avg_reward' in var:
                result[var] = mean_value
            elif 'cluster' in var:
                result[var] = int(mean_value)
        else:
            # 如果列名不在数据中，给出警告或处理逻辑
            result[var] = None  # 或者使用其他方式标记错误

    return result

def add_equal_constr(model, target_variable_name, target_variable_value):
    model.addConstr(model.getVarByName(target_variable_name) == target_variable_value)
    # print(target_variable_name, '==', target_variable_value)
    return model

def select_random_columns(csv_file_path, k):
    """
    从CSV文件中的第11列到第1043列中随机选择k个列名，并返回一个列表。
    
    :param csv_file_path: CSV文件的路径
    :param k: 要选择的列名数量
    :return: 随机选择的列名列表
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)
    
    # 获取第11列到第1043列的列名
    column_names = df.columns[10:1043]
    
    # 检查是否有足够的列
    if k > len(column_names):
        raise ValueError("k的值不能大于可用的列数")
    
    # 随机选择k个列名
    selected_columns = random.sample(list(column_names), k)
    return selected_columns

def capture_model_variables(model):
    variables_info = []
    # 遍历模型中的所有变量
    for var in model.getVars():
        # 检查变量名是否包含 'u' 或 'x'
        if 'cluster' in var.VarName or 'avg_reward' in var.VarName:
            # 如果变量名包含 'u' 或 'x'，则将变量名和变量的当前值添加到列表中
            variables_info.append((var.VarName, var.X))
    return variables_info

@tool
def sampler(user_target_indicator:list) -> list:
    """这个工具基于用户的目标，结合后端数据库中记录的数据，筛选出对用户目标影响不大的变量，将其参数化。输入的是一个包含用户需求以及指标公式的列表，最终返回一个列表，输出列表包括用户需求、指标公式以及参数化的变量"""
    database = SQLDatabase.from_uri("mysql+pymysql://root:Zhy15810886562!@127.0.0.1/vehicle")  
    agent_executor = create_sql_agent(llm3, db=database, agent_type="openai-tools", verbose=True)
    error_text = ""
    global scope_down, cluster_num, cluster_per_table, reward_num, reward_per_table
    template = '''You are a database administrator and an expert in operation research. 
    Now, the following descriptions are provided, including sets, parameters, and decision variables:
    Sets
    ---
    $$D$$: the ID of demand regions; $$S$$: the ID of supply regions; $$K$$: Set of discrete electric charge levels for shared e-bikes;
    Defined respectively as: D = [4,7,12,15,37,38,44,49] j \in D; S = [0,1,2,3,5,6,8,9,10,11,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,39,40,41,42,43,45,46,47,48] i \in S; K = [0,1,2] k \in K, k=0 represents electric charge at low level, k=2 means high charge level, so you can use determined value or interval to represent the level of electric charge when needed.
    ---
    Parameters
    ---
    $$I$$:The number of idle vehicles with k SOC at supply node i; $$b$$: The operator pays b for each driver who is allocated to demand region j but fails to pick up a customer. When there is a successful pick-up, the booking fee will be paid by the passenger; $$theta$$: a fixed share for operators; $$w_hat$$: The operator pays an inconvenience cost w_hat for a driver allocated from supply region i to demand region j; $$w$$: Allocation cost; $$fee_pkm$$: Mileage cost; $$d$$: The distance from supply region i to demand region j;
    Defined respectively as: b = 5; theta = 0.2; w_hat = d*fee_pkm; w = w_hat+b; fee_pkm = 0.5; inventory_avg = 9
    ---
    Decision Variables
    ---
    $$cluster{{i}}_cluster{{j}}_{{k}}$$: the operator allocates vehicles with k SOC from supply node i to demand node j; $$avg_reward{{j}}_{{k}}$$: the average trip revenue from region j with k SOC.
    ---
    Now, the simple information of tables and the meanings of columns as follows:
    Includes different decision variable valuation information columns in different tables, which may be one of the 'clusters{{i}}_cluster{{j}}_{{k}}' or 'avg_reward{{j}} _{{k}}'. That represent a part of the optimal solution for a unique feature combination programming problem. Table name shows which kind of decision variable is contained.
    The column named 'profit' means the optimal value for each problem.
    You need to find the decision variable columns in all tables of the database that have a smaller impact on the 'profit' column values of each table, and select them.    
    Attention: You must select {scope_down} decision variable columns serial number from all tables and your final output information should include the selected reason, the name of the table and the serial number (counting starting at 1) of the selected columns. Anyway the final output should have {scope_down} serial numbers. Therefore, {cluster_num} serial numbers of decision variable columns should be selected from tables containing 'cluster' in the table name, {cluster_per_table} columns in the average of each table name containing 'cluster', {reward_num} columns should be selected from table name containing the 'reward', and {reward_per_table} columns of each name containing 'reward'.
    Special attention should be paid to the query results of the data table, and no omissions should be made in the final output. Any results that meet the requirements should be output.
    The output format is: reason: 「」, so selected decision variables are:「table_name:[selected_columns_serial_number]」. Here is an example of output format for reference:
    Answer: "「The reason for variable selection are」, so selected decision variables are: 「cluster0:[14,17],reward44:[12,14]」". This is just a format example so your answer cannot be exactly the same. Therefore, you need to combine the actually selected table name and columns number into a dict format and be contained by the symbol '「」' based on your thinking. Format as shown in the example before.
    Be mindful that each of your SQL query statements should not be too general. So make all query code you used more precise strictly! Meanwhile, your database query statement and the statement's return results should not be too lengthy. You mustn't use SQL code like 'select * from' or other general code, these code is too general and will make system shutdown. Make all query code you used more precise.
    Anyway you need to give me an answer strictly in line with my format requirements. Anyway, you must fulfill the requirements.
    Take a deep breath and solve the problem step by step.
'''
    success = False
    while not success: 
        try:
            prompt = PromptTemplate(input_variables=["scope_down","cluster_num","cluster_per_table","reward_num","reward_per_table"],template=error_text+template)
            prompt_sample = prompt.format(scope_down=scope_down,cluster_num=cluster_num,cluster_per_table=cluster_per_table,reward_num=reward_num,reward_per_table=reward_per_table)
            database_result = agent_executor.invoke(prompt_sample)
            success = True  # 如果调用成功，则将success设置为True
        except Exception as e:  # 捕获通用异常以显示错误消息（如果需要）
            error_text = str(e)
            continue
    parameters_text = database_result['output']
    print(parameters_text)
    try:
        matches = re.findall(r'「.*?」', str(parameters_text))
        global Selected_reason, Selected_numbers
        Selected_reason = matches[0]
        Selected_numbers = matches[-1]
        # print(Selected_reason)
        # print('Selected_numbers:',Selected_numbers)
        user_target_indicator = []
        for s in matches:
            user_target_indicator.append(s)
    except:
        template2 = """{parameters_text} The content mention before is based on a database search to select the decision variable columns in all tables of the database, but it often does not meet my expected output format. You must extract its content into two parts, each enclosed with "「」" symbols. The first part should explain the reason for selecting the decision variable columns, and the second part should list the name of the table and the serial number (counting starting at 1) of the selected columns contained by '「」' symbol. Attention: You just need to format adapt the results, don't make excessive changes to them.
        You need to check the format and content. The final output should be a string.  The output format is: reason: 「」, so selected decision variables are:「table_name:[selected_columns_serial_number]」. Here is an example of output format for reference:
        Reason explanation: "「The reason for variable selection are」, so selected decision variables are: 「cluster0:[14,17],reward44:[12,14]」". This is just a format example so your answer cannot be exactly the same. Anyway, you must provide a station selection that is different from the example I provided, according to the format requirements! The text you output should not contain any omitted forms (e.g. '...','-'), and all selected decision variables should be listed."""
        prompt2 = PromptTemplate(input_variables=["parameters_text"],template=template2)
        prompt_sample2 = prompt2.format(parameters_text = parameters_text)
        format_output = llm3.invoke(prompt_sample2)  
        matches = re.findall(r'「.*?」', str(format_output))
        # global Selected_reason, Selected_numbers
        Selected_reason = matches[0]
        Selected_numbers = matches[-1]
        # print(Selected_reason)
        # print('Selected_numbers:',Selected_numbers)
        user_target_indicator = []
        for s in matches:
            user_target_indicator.append(s)
    return user_target_indicator

tools=[sampler]

def agent_prompt():
    #Create prompt
    from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

    #create prompt
    from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
    global chat_history
    prompt = ChatPromptTemplate.from_messages(
        [    
        ("system","You are a very capable assistant, and you need to fulfill my requirements based on the tools I provide to you."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
        ]
    )

    # bind tool
    from langchain_community.tools.convert_to_openai import format_tool_to_openai_function
    llm_tools=llm1.bind(functions=[format_tool_to_openai_function(t) for t in tools])

    #create agent
    chat_history = []

    from langchain.agents.format_scratchpad import format_to_openai_function_messages
    agent=(
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"]
    }
    |prompt
    |llm_tools
    |OpenAIFunctionsAgentOutputParser()
    )

    # initial prompt
    #prompt.pretty_print()
    return agent

if __name__ == "__main__":
    gurobi_model_path = '/Users/admin/Documents/原文稿/jupyter/AAAI/experiment0720/model.mps'
    subdata_csv_path = '/Users/admin/Documents/原文稿/jupyter/AAAI/experiment0720/split'

    # 使用示例
    # target = 'Supply-demand matching degree of vehicles'
    # Index = 'Improve Supply-demand matching degree of vehicles'
    # target = 'Market share of vehicles'
    # Index = 'Improve market share of vehicles'
    target = 'Dispatching efficiency of vehicles'
    Index = 'Improve dispatching efficiency of vehicles'
    # locality = 0.1
    
    csv_file_path = '/Users/admin/Documents/原文稿/jupyter/AAAI/experiment0720/solution_data_2.csv'  # 替换为实际的CSV文件路径

    demand_avg = np.array([-1.34, -72.57, -427.42, -1.8, 11.87, -10.29, -2.72, 544.47, -1.67, -0.78, -0.41, -2.35, 392.5, -12.62,
                    -4.96,
                    70.81, -1.0, -4.46, -1.33, -9.61, -1.67, -3.6, -4.22, -1.14, -2.17, -3.28, -3.07, -4.08, -1.0, -0.94,
                    -9.84, -2.02,
                    -1.82, -1.04, -1.0, -1.79, -91.87, 36.66, 414.2, -0.46, -2.29, -0.84, -1.54, -5.08, 32.29, -1.35,
                    -1.44, -1.49,
                    -2.8, 18.66])
    soc=[0,1,2]
    demand_area = [4, 7, 12, 15, 37, 38, 44, 49]
    supply_area = [0, 1, 2, 3, 5, 6, 8, 9, 10, 11, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                31, 32, 33, 34, 35, 36, 39, 40, 41, 42, 43, 45, 46, 47, 48]
    w1 = 1
    w2 = 1
    booking_fee = 5
    theta = 0.2
    fee_pkm = 0.5
    inventory_avg = 9
    area_num = 50
    dv_num = 1032

    net_demand_soc = generate_net_demand(area_num,demand_area)
    with open(r'/Users/admin/Documents/原文稿/jupyter/AAAI/experiment0720/distance_matrix.pickle', 'rb') as f:  #供给区到需求区的距离矩阵
        distance_matrix= pickle.load(f)
    w_hat = distance_matrix * fee_pkm  
    W = w_hat + booking_fee 

    for locality in [0.1]:
        global scope_down, reward_num, reward_per_table, cluster_num, cluster_per_table
        scope_down = dv_num*locality
        reward_num = math.floor(24/1032*scope_down)
        reward_per_table = reward_num/8
        cluster_num = math.floor((1032-24)/1032*scope_down)
        cluster_per_table = cluster_num/42
        k = math.floor(locality * 1032)  # 选择的列名数量

        

        for t in range(1):
            # Create agent
            agent = agent_prompt()
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
            input_target = '「' + str(Index) + '」'

            input_prompt = """
            The text within "「」" marks my target, which should include the domain of application and the metrics I am concerned with. You need to use the tool to accomplish the following tasks sequentially:
            Using data recorded in the backend database, identify station IDs that have little impact on the metrics and parameterize them, returning results as per the tool's requirements.
            finally, you just answer me 'All agent task has been done!'.without any further explanation.
                """
            input_combine = input_target + input_prompt

            global global_parameters, global_obj_history, global_evaluator_history
            global_obj_history = 0
            global_evaluator_history = 0
            try:
                result = agent_executor.invoke({"input": input_combine, "chat_history": chat_history})
            except Exception as e:
                print(e)
                # 如果出错，播放警告音
                need_run = 5-t
                sys.exit(1) 
            chat_history.extend([HumanMessage(content=input_combine),
                                AIMessage(content=result["output"])
                                ])
            

            LEO = gp.read(gurobi_model_path)
            obj1 = LEO.getObjective()
            # temp_var = LEO.addVar(vtype=gp.GRB.CONTINUOUS,name='u')
            LEO.setObjectiveN(w2 * obj1, index=0, priority=5, name='obj_tree')
            LEO.modelSense = GRB.MAXIMIZE
            LEO.update()
            if target == 'Supply-demand matching degree of vehicles':
                temp_var1 = LEO.addVars(demand_area,vtype=gp.GRB.CONTINUOUS,name=[f'u_{j}'for j in demand_area])
                LEO.update()
                expr1 = gp.quicksum(net_demand_soc[i][k] - gp.quicksum(LEO.getVarByName(f'cluster{i}_cluster{j}_{k}') for j in demand_area) for k in soc for i in supply_area) 
                expr2 = gp.quicksum(LEO.getVarByName(f'u_{j}') for j in demand_area)

                for j in demand_area:
                    LEO.addConstr(LEO.getVarByName(f'u_{j}') >= demand_avg[j]-inventory_avg-gp.quicksum(LEO.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in supply_area for k in soc))
                    LEO.addConstr(LEO.getVarByName(f'u_{j}') >= -1*(demand_avg[j]-inventory_avg-gp.quicksum(LEO.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in supply_area for k in soc)))
                
                LEO.setObjectiveN(-1*w1 * (expr1+expr2),index=1, priority=1, name='obj_indicator')
                LEO.update()

            elif target == 'Market share of vehicles':
                #Market share of vehicles
                # obj2 = gp.quicksum(LEO.getVarByName(f'avg_reward{j}_{k}') * LEO.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in supply_area for j in demand_area for k in soc)
                new_obj_expr = gp.LinExpr()
                u = LEO.addVar(vtype=GRB.CONTINUOUS, name="u")
                LEO.update()
                new_obj_expr.addTerms(1,u)
                LEO.update()
                LEO.addConstr(LEO.getVarByName('u') == gp.quicksum(LEO.getVarByName(f'avg_reward{j}_{k}') * LEO.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in supply_area for j in demand_area for k in soc))

                LEO.setObjectiveN(-1*w1 * new_obj_expr,index=1, priority=1, name='obj_indicator')
                LEO.update()

                
            elif target == 'Dispatching efficiency of vehicles':
                u = LEO.addVar(vtype=GRB.CONTINUOUS, name="u")

                # temp_var1 = LEO.addVars(demand_area,vtype=gp.GRB.CONTINUOUS,name=[f'u_{j}'for j in demand_area])
                LEO.update()
                expr1 = gp.quicksum(W[i][j] * gp.quicksum(LEO.getVarByName(f'cluster{i}_cluster{j}_{k}')for k in soc) for i in supply_area for j in demand_area)
                # expr2 = gp.quicksum(LEO.getVarByName(f'u_{j}') for j in demand_area)
                LEO.addConstr(LEO.getVarByName('u') == gp.quicksum(LEO.getVarByName(f'avg_reward{j}_{k}') * LEO.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in supply_area for j in demand_area for k in soc))

                LEO.setObjectiveN(-1*w1 * (u-expr1),index=1, priority=1, name='obj_indicator')
                LEO.update()
            #     # Dispatching efficiency of vehicles
            #     # obj2 = gp.quicksum((LEO.getVarByName(f'avg_reward{j}_{k}') - w[i][j]) * LEO.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in supply_area for j in demand_area for k in soc)
            #     new_obj_expr = gp.LinExpr()
            #     u = LEO.addVar(vtype=GRB.CONTINUOUS, name="u")
            #     LEO.update()
            #     new_obj_expr.addTerms(1,u)
            #     LEO.update()
            #     LEO.addConstr(LEO.getVarByName('u') == gp.quicksum((LEO.getVarByName(f'avg_reward{j}_{k}') - w[i][j]) * LEO.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in supply_area for j in demand_area for k in soc))

            #     LEO.setObjectiveN(-1*w1 * new_obj_expr,index=1, priority=1, name='obj_indicator')
            #     LEO.update()
            # # new_obj_expr = gp.LinExpr()

            # # new_obj_expr.addTerms(1,temp_var)
            # # LEO.update()


            # Supply-demand matching degree of vehicles
            # obj2 = gp.quicksum(gp.quicksum(net_demand_soc[i][k]-gp.quicksum(base.getVarByName(f'cluster{i}_cluster{j}_{k}')for j in demand_area) for k in soc) for i in supply_area)+gp.quicksum(abs(demand_avg[j]-inventory_avg-gp.quicksum(base.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in supply_area for k in soc)) for j in demand_area)
            # LEO.addConstr(temp_var == obj2,'nonlinear')
            
            scope_down = Selected_numbers
            LEO,selected_num,selected_dv = process_string_and_add_constraints(scope_down,LEO,subdata_csv_path)
            LEO.update()

            # LEO.write('temp.lp')
            # LEO.write('temp.mps')
            # selected_columns = select_random_columns(csv_file_path, k)

            # converted_variables = variables_values(selected_columns, csv_file_path)
            # for variable, value in converted_variables.items():
            #     #model = replace_constraint(model, variable, value)
            #     LEO = add_equal_constr(LEO, variable, value)
            # LEO.update()
            # LEO.write('LEO_nonlinear.lp')
            LEO.setParam(GRB.Param.NonConvex, 2)
            LEO.setParam(GRB.Param.DisplayInterval, 60)
            LEO.setParam(GRB.Param.TimeLimit, 3600)
            start_time = time.time()
            LEO.optimize()
            end_time = time.time()
            LEO_optimize_time = end_time - start_time
            leo_model_vars = capture_model_variables(LEO)
            LEO_obj_value=[]
            for i in range(LEO.NumObj):
                LEO.setParam(GRB.Param.ObjNumber, i)
                print(f"Obj {i + 1} = {LEO.ObjNVal}")
                LEO_obj_value.append(LEO.ObjNVal)
            LEO.dispose()

            base = gp.read(gurobi_model_path)
            obj1 = base.getObjective()    
            # temp_var = base.addVar(vtype=gp.GRB.CONTINUOUS,name='u')
            # new_obj_expr = gp.LinExpr()
            # new_obj_expr.addTerms(1,temp_var)
            # temp_var1 = base.addVars(demand_area,vtype=gp.GRB.CONTINUOUS,name=[f'u_{j}'for j in demand_area])
            # base.update()
            
            # # Market share of vehicles
            # # obj2 = gp.quicksum(base.getVarByName(f'avg_reward{j}_{k}') * base.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in supply_area for j in demand_area for k in soc)
            
            # #Dispatching efficiency of vehicles
            # # obj2 = gp.quicksum((base.getVarByName(f'avg_reward{j}_{k}') - w[i][j]) * base.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in supply_area for j in demand_area for k in soc)
            
            # # Supply-demand matching degree of vehicles
            # # obj2 = gp.quicksum(gp.quicksum(net_demand_soc[i][k]-gp.quicksum(base.getVarByName(f'cluster{i}_cluster{j}_{k}')for j in demand_area) for k in soc) for i in supply_area)+gp.quicksum(abs(demand_avg[j]-inventory_avg-gp.quicksum(base.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in supply_area for k in soc)) for j in demand_area)



            # for j in demand_area:
            #     # print(demand_avg[j])
            #     base.addConstr(base.getVarByName(f'u_{j}') >= demand_avg[j]-inventory_avg-gp.quicksum(base.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in supply_area for k in soc))
            #     base.addConstr(base.getVarByName(f'u_{j}') >= -1*(demand_avg[j]-inventory_avg-gp.quicksum(base.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in supply_area for k in soc)))
            # ###
            # base.addConstr(temp_var == obj2,'nonlinear')
            base.setObjectiveN(w2 * obj1, index=0, priority=5, name='obj_tree')
            base.modelSense = GRB.MAXIMIZE

            if target == 'Supply-demand matching degree of vehicles':
                temp_var1 = base.addVars(demand_area,vtype=gp.GRB.CONTINUOUS,name=[f'u_{j}'for j in demand_area])
                base.update()
                expr1 = gp.quicksum(net_demand_soc[i][k] - gp.quicksum(base.getVarByName(f'cluster{i}_cluster{j}_{k}') for j in demand_area) for k in soc for i in supply_area) 
                expr2 = gp.quicksum(base.getVarByName(f'u_{j}') for j in demand_area)

                for j in demand_area:
                    base.addConstr(base.getVarByName(f'u_{j}') >= demand_avg[j]-inventory_avg-gp.quicksum(base.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in supply_area for k in soc))
                    base.addConstr(base.getVarByName(f'u_{j}') >= -1*(demand_avg[j]-inventory_avg-gp.quicksum(base.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in supply_area for k in soc)))
                
                base.setObjectiveN(-1*w1 * (expr1+expr2),index=1, priority=1, name='obj_indicator')
                base.update()

            elif target == 'Market share of vehicles':
                #Market share of vehicles
                # obj2 = gp.quicksum(base.getVarByName(f'avg_reward{j}_{k}') * base.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in supply_area for j in demand_area for k in soc)
                new_obj_expr = gp.LinExpr()
                u = base.addVar(vtype=GRB.CONTINUOUS, name="u")
                base.update()
                new_obj_expr.addTerms(1,u)
                base.update()
                base.addConstr(base.getVarByName('u') == gp.quicksum(base.getVarByName(f'avg_reward{j}_{k}') * base.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in supply_area for j in demand_area for k in soc))

                base.setObjectiveN(-1*w1 * new_obj_expr,index=1, priority=1, name='obj_indicator')
                base.update()

                
            elif target == 'Dispatching efficiency of vehicles':
                u = base.addVar(vtype=GRB.CONTINUOUS, name="u")

                # temp_var1 = LEO.addVars(demand_area,vtype=gp.GRB.CONTINUOUS,name=[f'u_{j}'for j in demand_area])
                base.update()
                expr1 = gp.quicksum(W[i][j] * gp.quicksum(base.getVarByName(f'cluster{i}_cluster{j}_{k}')for k in soc) for i in supply_area for j in demand_area)
                # expr2 = gp.quicksum(LEO.getVarByName(f'u_{j}') for j in demand_area)
                base.addConstr(base.getVarByName('u') == gp.quicksum(base.getVarByName(f'avg_reward{j}_{k}') * base.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in supply_area for j in demand_area for k in soc))

                base.setObjectiveN(-1*w1 * (u-expr1),index=1, priority=1, name='obj_indicator')
                base.update()
                # temp_var1 = base.addVars(demand_area,vtype=gp.GRB.CONTINUOUS,name=[f'u_{j}'for j in demand_area])
                # # Dispatching efficiency of vehicles
                # # obj2 = gp.quicksum(base.getVarByName(f'avg_reward{j}_{k}') * base.getVarByName(f'cluster{i}_cluster{j}_{k}') - w[i][j] * base.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in supply_area for j in demand_area for k in soc)
                # # obj2 = gp.quicksum((base.getVarByName(f'avg_reward{j}_{k}') - w[i][j]) * base.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in supply_area for j in demand_area for k in soc)
                # new_obj_expr = gp.LinExpr()
                # u = base.addVar(vtype=GRB.CONTINUOUS, name="u")
                # base.update()
                # new_obj_expr.addTerms(1,u)
                # base.update()
                # base.addConstr(base.getVarByName('u') == gp.quicksum(base.getVarByName(f'avg_reward{j}_{k}') * base.getVarByName(f'cluster{i}_cluster{j}_{k}') - w[i][j] * base.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in supply_area for j in demand_area for k in soc))

                # base.setObjectiveN(-1*w1 * new_obj_expr,index=1, priority=1, name='obj_indicator')
                # base.update()

            # base.setObjectiveN(-1*w1 * (expr1+expr2),index=1, priority=1, name='obj_indicator')
            # base.update()
            # base.write('base_nonlinear.lp')
            base.setParam(GRB.Param.NonConvex, 2)
            base.setParam(GRB.Param.DisplayInterval, 60)
            base.setParam(GRB.Param.TimeLimit, 3600)
            start_time = time.time()
            base.optimize()
            end_time = time.time()
            base_optimize_time = end_time - start_time
            base_model_vars = capture_model_variables(base)
            base_obj_value=[]
            for i in range(base.NumObj):
                base.setParam(GRB.Param.ObjNumber, i)
                print(f"Obj {i + 1} = {base.ObjNVal}")
                base_obj_value.append(base.ObjNVal)
            base.dispose()
            optimization_gap = [abs(abs(float(base_obj_value[i]))-abs(float(LEO_obj_value[i])))/(0.00001+abs(float(base_obj_value[i]))) for i in range(len(base_obj_value))]
            tnow = time.time()
            file_path = '/Users/admin/Documents/原文稿/jupyter/AAAI/exper_nonlinear_new/index{}_sample{}_{}.txt'.format(target, locality, tnow)

            f = open(file_path, 'w')
            #'Harvard Ave & E Pine St, E Pine St & 16th Ave, Summit Ave E & E Republican St, 2nd Ave & Pine St, REI / Yale Ave N & John St'
            # f.write('user:{}'.format(selected))
            # f.write('\n')
            # print('Python_formulation:', Python_formulation)
            print('selected_num:',selected_num)
            f.write('selected_num:{}'.format(selected_num))
            f.write('\n')
            print('selected decision variable:', selected_dv)
            f.write('selected decision variable:{}'.format(selected_dv))
            f.write('\n')
            print('LEO_optimize_time:', LEO_optimize_time)
            f.write('LEO_optimize_time:{}'.format(LEO_optimize_time))
            f.write('\n')
            print('LEO_result:', LEO_obj_value)
            f.write('LEO_result:{}'.format(LEO_obj_value))
            f.write('\n')
            print('base_optimize_time:', base_optimize_time)
            f.write('base_optimize_time:{}'.format(base_optimize_time))
            f.write('\n')
            print('base_result', base_obj_value)
            f.write('base_result:{}'.format(base_obj_value))
            f.write('\n')
            # print('Optimization gap',abs(float(base_optimize_time)-float(LEO_optimize_time))/float(base_optimize_time))
            optimization_gap = [abs(abs(float(base_obj_value[i]))-abs(float(LEO_obj_value[i])))/(0.00001+abs(float(base_obj_value[i]))) for i in range(len(base_obj_value))]
            print(f'Optimization gap 1: {optimization_gap[0] * 100:.2f}%')
            print(f'Optimization gap 2: {optimization_gap[1] * 100:.2f}%')
            f.write(f'Optimization gap 1: {optimization_gap[0] * 100:.2f}%')
            f.write('\n')
            f.write(f'Optimization gap 2: {optimization_gap[1] * 100:.2f}%')
            f.write('\n')

            # f.write(f'caculate expression: {expression}')
            # f.write('\n')

            # Write LEO Model Variables and Values
            f.write("LEO Model Variables and Values:\n")
            for var_name, var_value in leo_model_vars:
                f.write(f"{var_name} = {var_value}\n")

            f.write("Base Model Variables and Values:\n")
            for var_name, var_value in base_model_vars:
                f.write(f"{var_name} = {var_value}\n")


            f.close()