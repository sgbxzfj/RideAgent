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
os.environ["OPENAI_API_KEY"] = ''
llm1 = ChatOpenAI(model="gpt-4o",temperature=0.8)  
llm2 = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.8)  
llm3 = ChatOpenAI(model="gpt-4o-mini",temperature=0.8)  
def add_new_obj(new_model_path, sense, text, w1, w2):
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
    with open(r'/Users/admin/Documents/jupyter/AAAI/experiment0720/distance_matrix.pickle', 'rb') as f:  #供给区到需求区的距离矩阵
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

    exec(text, {'model': model, 'gp': gp, "GRB": gp.GRB, 'I': net_demand_soc, "b": booking_fee, "theta": theta, "fee_pkm": fee_pkm, "w_hat": w_hat, "w": w, "D":demand_area, "S":supply_area, "K":soc, "d":distance_matrix, "demand_avg":demand_avg, "inventory_avg":inventory_avg})
    model.update()

    # 新目标函数
    new_obj = model.getObjective()
    #print('extra_obj:',new_obj)

    # 加载旧模型
    new_model = gp.read(new_model_path)
    new_model.update()
    origin_obj = new_model.getObjective()
    # 重建目标函数
    new_obj_expr = gp.LinExpr()
    if isinstance(new_obj, gp.LinExpr):
        #print('LinExpr')
        # 线性表达式处理
        for i in range(new_obj.size()):
            coeff = new_obj.getCoeff(i)
            var_name = new_obj.getVar(i).VarName
            new_var = new_model.getVarByName(var_name)
            try:
                new_var = new_model.getVarByName(var_name)
                new_obj_expr.addTerms(coeff, new_var)
            except gp.GurobiError as e:
                continue
        new_obj_expr.addConstant(new_obj.getConstant())
    elif isinstance(new_obj, gp.QuadExpr):
        new_obj_expr = gp.QuadExpr()
        #print('QuadExpr')
        # 二次表达式处理
        for i in range(new_obj.size()):
            coeff = new_obj.getCoeff(i)
            var1_name = new_obj.getVar1(i).VarName
            var2_name = new_obj.getVar2(i).VarName
            #print(var1_name,var2_name)
            new_var1 = new_model.getVarByName(var1_name)
            new_var2 = new_model.getVarByName(var2_name)
            new_obj_expr.addTerms(coeff, new_var1, new_var2)

        # 处理线性部分
        lin_expr = new_obj.getLinExpr()
        for i in range(lin_expr.size()):
            coeff = lin_expr.getCoeff(i)
            var_name = lin_expr.getVar(i).VarName
            new_var = new_model.getVarByName(var_name)
            new_obj_expr.addTerms(coeff, new_var)

    # 设置新模型的目标函数
    ###
    # 正则表达式匹配规则
    pattern = r'GRB\.(MAXIMIZE|MINIMIZE)'
    # 执行匹配
    match = re.search(pattern, str(text))
    # 根据匹配结果打印
    if match:
        new_model.setObjectiveN(w2 * origin_obj, index=0, priority=5, name='obj_tree')
        if sense == 'MAXIMIZE':
            new_model.modelSense = GRB.MAXIMIZE
        elif sense == 'MINIMIZE':
            new_model.modelSense = GRB.MINIMIZE
    if match.group(1) == 'MAXIMIZE':
        new_model.setObjectiveN(-w1 * new_obj_expr,index=1, priority=1, name='obj_indicator')
    elif match.group(1) == 'MINIMIZE':
        new_model.setObjectiveN(w1 * new_obj_expr,index=1, priority=1, name='obj_indicator')
    new_model.update()

    #print(new_model.getObjective())
    return new_model

def add_equal_constr(model, target_variable_name, target_variable_value):
    model.addConstr(model.getVarByName(target_variable_name) == target_variable_value)
    # print(target_variable_name, '==', target_variable_value)
    return model

def variables_obtain(spot_id,model):
    #print('spot_id',spot_id)
    indices = [int(num) for num in spot_id.split(',')]

    # 加载Gurobi模型
    # 初始化决策变量名称列表
    decision_variables = []

    # 遍历模型中的每一个变量
    for var in model.getVars():
        # 提取变量名称
        var_name = var.VarName
        # 检查变量名称是否包含 'dispatch' 或 'exchange' 以及索引 i
        for i in indices:
            if f"cluster{i}_cluster" in var_name:
                decision_variables.append(var_name)
    return decision_variables

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

def process_constraints(user_input, model, csv_path):
    """
    Process and replace constraints based on the decision variables extracted from the text content.

    Parameters:
    - text_content: The text containing the decision variables enclosed in 「」.
    - model: The Gurobi model where constraints will be replaced.

    Returns:
    - The updated model after replacing the constraints.
    """
    # Extract decision variables from the text content


    # Split the string by commas that are not inside square brackets
    decision_variables = variables_obtain(user_input,model)

    # Convert variable names and get values
    converted_variables = variables_values(decision_variables, csv_path)

    # Replace constraints in the model based on the converted variable names and their values
    for variable, value in converted_variables.items():
        #model = replace_constraint(model, variable, value)
        model = add_equal_constr(model, variable, value)
    return model

def extract_numbers(selected_spot_numbers):
    # 去掉字符串中的「」符号
    clean_string = selected_spot_numbers.strip("「」")
    
    # 分割字符串得到数字列表
    numbers = clean_string.split(',')
    
    # 去除每个数字的空格
    stripped_numbers = [num.strip() for num in numbers]
    
    # 重新连接成逗号分隔的字符串
    partial_spot_numbers = ','.join(stripped_numbers)
    
    return partial_spot_numbers

def parsing_text(re_search):
    if re_search:
        # 检查匹配结果是 MINIMIZE 还是 MAXIMIZE
        objective = 'MINIMIZE' if 'MINIMIZE' in re_search.group() else 'MAXIMIZE'
        new_result = f'model.setObjective({re_search.group(1)} GRB.{objective})'
        new_result = f'「{new_result}」'
    else:
        new_result = None
    #print(new_result)    
    matches2 = re.findall(r'「(.*?)」', str(new_result))
    for match in matches2:
        print(match)
        temp_text = str(match)
    return temp_text

def get_optimal_values(model):
    return {var.name: var.value() for var in model.variables()}

def capture_model_variables(model):
    variables_info = []
    # 遍历模型中的所有变量
    for var in model.getVars():
        # 检查变量名是否包含 'u' 或 'x'
        if 'cluster' in var.VarName or 'avg_reward' in var.VarName:
            # 如果变量名包含 'u' 或 'x'，则将变量名和变量的当前值添加到列表中
            variables_info.append((var.VarName, var.X))
    return variables_info

def evaluate_expression(expression_dict, key):
    global randseed, expression
    expression = expression_dict[key]
    expression = re.sub(r'gp\.quicksum', 'sum', expression)
    expression = re.sub(r'model.getVarByName\((.*?)\)', r'model.getVarByName(\1).X', expression)
    # 移除表达式中关于设置目标函数的部分
    expression = re.sub(r'model.setObjective\((.*?), GRB.(MINIMIZE|MAXIMIZE)\)', r'\1', expression)
    print(expression)
    # 动态执行表达式
    # total_value = eval(expression, {'model': model, 'gp': gp, "GRB": gp.GRB, 'I': net_demand_soc, "b": booking_fee, "theta": theta, "fee_pkm": fee_pkm, "w_hat": w_hat, "w": w, "D":demand_area, "S":supply_area, "K":soc, "d":distance_matrix})
    return expression

def sample_expressions(n_samples, include_mandatory, mandatory_key, expression_dict):
    """
    Sample expressions from a dictionary, either ensuring the inclusion or exclusion
    of a specified mandatory item.

    Args:
    n_samples (int): Number of items to sample from the dictionary.
    include_mandatory (bool): If True, include the mandatory item; if False, exclude it.
    mandatory_key (str): The key of the item to include or exclude.
    expression_dict (dict): The dictionary from which to sample.
    seed (int): Random seed for reproducibility.

    Returns:
    dict: A dictionary containing sampled expressions.
    """
    keys = list(expression_dict.keys())
    sampled_dict = {}

    if n_samples == 0:
        return 'None'

    if include_mandatory:
        # Ensure the mandatory item is included if specified
        if mandatory_key in keys:
            sampled_dict[mandatory_key] = expression_dict[mandatory_key]
            keys.remove(mandatory_key)
            n_samples -= 1  # Reduce the number of items to sample since one is already included
    else:
        # Ensure the mandatory key is excluded
        if mandatory_key in keys:
            keys.remove(mandatory_key)

    # Randomly sample the remaining number of keys from the list
    sampled_keys = random.sample(keys, min(n_samples, len(keys)))

    # Add the sampled items to the dictionary
    for key in sampled_keys:
        sampled_dict[key] = expression_dict[key]

    return sampled_dict

def remove_backslashes_from_getvar(input_string):
    # 正则表达式用于匹配getVarByName的调用
    pattern = r'getVarByName\((.*?)\)'

    # 使用正则表达式的sub方法，以及lambda函数进行匹配替换
    result = re.sub(pattern, lambda m: "getVarByName(" + m.group(1).replace("\\", "") + ")", input_string)
    
    return result

def play_sound(sound_text):
    # 使用afplay来播放音频文件，确保文件路径正确
    os.system(f'say "{sound_text}"')
    #os.system(f'afplay {sound_file}')

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

def process_string_and_add_constraints(input_string,model):
    # Path to the directory containing CSV files
    csv_directory = '/Users/admin/Documents/jupyter/AAAI/experiment0720/split'
    cleaned_string = input_string.replace('「', '').replace('」', '')
    print(cleaned_string)
    # Extract key-value pairs from the input string
    # pattern = r'(\w+\d+): \[(\d+)\]'

    pattern = r'(\w+\d+): \[(\d+)\]'
    matches = re.findall(pattern, cleaned_string)
    print(matches)
    converted_variables = {}
    
    for key, value in matches:
        # print(key, value)
        value = int(value)
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
                print('column_name',column_name,'mean_value',mean_value)  
        except FileNotFoundError:
            print(f"File {csv_path} not found.")
        except IndexError:
            print(f"Index {value} is out of range in file {csv_path}.")
    
    # Add constraints to the Gurobi model
    for variable, value in converted_variables.items():
        model = add_equal_constr(model, variable, value)
    
    return model

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
        play_sound('zero sample')
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
from langchain_core.tools import tool
from langchain import PromptTemplate
import re
#Create tools
#[domain, target, target description,target code,partial variable,]

@tool
def recognizer(user_input:str) -> list:
    """这个工具用于提取出用户想要作用的领域和想要实现的指标。输入的是用户的需求字符串格式，最终返回一个列表，输出列表中包含作用领域和指标的提取结果"""
    template = """Based on the {input}, you need to extract the user's domain of activity and the goals they want to achieve. The response format must be exactly as follows: domain:「」, target:「」, with no alterations. Here, "domain" describes the real-world area of application, and "target" describes the specific metrics or objectives they wish to improve. Extracting the information directly without any summarization.
    For example:domain:「electric vehicles」,target:「Improve future service Level of vehicles'.」.You should pay special attention to the user's location requirements in your identification results and truthfully record them.
    """
    prompt = PromptTemplate(input_variables=["input"],template=template)
    prompt_intent = prompt.format(input=user_input)
    #print(prompt_intent)
    response = llm2.invoke(prompt_intent)
    print(response)
    matches = re.findall(r'「(.*?)」', str(response))
    global Domain, Target
    try:
        Domain = matches[0]
        Target = matches[1]
        return matches
    except:
        Target = Index
        return Target

@tool
def indicator(user_target:list)-> list:
    """这个工具基用于结合后端的优化模型获取指标的可能公式构成，并通过LLM确保该公式的gurobi表述能够正常运行。输入是包含用户需求的列表，最终返回一个列表，输出列表中用户需求以及指标公式构成"""
    global groundtruth, Index
    template = '''You are an assistant to a professor of operations research, possessing a strong foundation in mathematics and operations research. You need to complete tasks according to requirements, while ensuring the output format meets the specified standards. The requirements are as follows:
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
If you need to use a decision variable, follow this form: model.addVar(name=f"cluster{{i}}_cluster{{j}}_{{k}}"); model.addVar(name=f"avg_reward{{j}}_{{k}}"). But MUST NOT contain any product of decision variables in the objective function formulated by you anyway. Like cluster{{i}}_cluster{{j}}_{{k}} * avg_reward{{j}}_{{k}}
---
Attention: when you formulate an expression which include parameters: I,w,w_hat,d,demand_avg you need to use this parameters like I[i][k], w[i][j], w_hat[i][j], d[i][j], demand_avg[j] because they are array type. 
Based on the above parameters, you are required to construct an objective funcation instead of constraints to depict {target}, which must be sensible, containing the mentioned parameters and decision variables without creation of additional elements, and the formula must be convex. Must not use quadratic terms(**2,^2) in fomula. Must not use the maximum value (max/min) or absolute value (abs) in the formula. Must not contain the product of two decision variables like:avg_reward{{j}}_{{k}}*cluster{{i}}_cluster{{j}}_{{k}}'. Must not contain fractions. Must not contain any line continuation character.
The output must contain the following two points, and the format must adhere to the following:
1. Start with an explanation of the formula, explaining why it is set up this way, and be sure to enclose the formula explanation in “「」”;
2. Then provide the corresponding python statement for the formula, ensuring it is enclosed in “「」” and only includes the previously mentioned sets, parameters, and decision variables in the format provided;
For example: Formula explanation: “「The meaning of the formula is:」”, python representation: “「」”.
Now I will give you some groundtruth formulation, If the content I provide you is similar to the goal, you can refer to it.
{answer}
The output format must follow the example above, enclosed in “「」”, but the content is for you to decide, the formula should not be complex, just reasonable. Python code mast not contain special symbols such as line breaks({{\n}}).
Make sure the output standard is followed, and the representation of variables, parameters, and subscripts conforms to the defined method, with no latex representation in the content.
Take a deep breath and solve the problem step by step.
'''
    prompt = PromptTemplate(input_variables=["target","answer"],template=template)
    prompt_scope_down = prompt.format(target=Index,answer=groundtruth)
    # print(prompt_scope_down)
    result = llm1.invoke(prompt_scope_down)
    matches = re.findall(r'「(.*?)」', str(result))
    global Formulation_explanation, Python_formulation 
    Formulation_explanation = matches[0]
    Python_formulation = matches[1]
        #print(match)
    template2 = """    
    The current objective function {formulation} consists of the following sets, parameters, and decision variables:
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
    If you need to use a decision variable, follow this form: model.addVar(name=f"cluster{{i}}_cluster{{j}}_{{k}}"); model.addVar(name=f"avg_reward{{j}}_{{k}}"). But MUST NOT contain any product of decision variables in the objective function formulated by you anyway, like model.getVarByName(f'avg_reward{{j}}_{{k}}') * model.getVarByName(f'cluster{{i}}_cluster{{j}}_{{k}}')
    ---
    You need to rewrite the basic python syntax expression {formulation} as the objective function code for gurobi (model.setObjective()). You must ensure that your gurobi code is complete, containing decision variable identification (model.getVarByName), summing (sum or gp.quicksum), the objective function keyword (model.setObjective()), and the sense of the objective function must in (GRB.MAXIMIZE or GRB.MINIMIZE). 
    Attention: when you formulate an expression which include parameters: I,w,w_hat,d,demand_avg you need to use this parameters like I[i][k], w[i][j], w_hat[i][j], d[i][j], demand_avg[j] because they are array type. 
    Now I will give you some groundtruth formulation, If the content I provide you is similar to the goal, you can refer to it.
    {answer}
    When you attempt to use getVarByName to fetch a decision variable, you must ensure the object being used is indeed a decision variable, and only decision variables should use model.getVarByName, while the rest are treated as parameters. 
    The content in the code must conform to their definitions, the output must be in gurobi syntax. Mustn't contain any line continuation character and the format must comply with: gurobi code is: “「」”.
    The gurobi code must obtain r'model.setObjective\((.*?)\s*GRB\.(MINIMIZE|MAXIMIZE)'
    For example:    gurobi code is: "「“”“model.setObjective(gp.quicksum(model.getVarByName(f'avg_reward{{j}}_{{k}}') for j in D for k in K), GRB.MAXIMIZE)“”“」"
    The output format must follow the example above, enclosed in “「」” and it cannot be the same as the example I gave, fill in the sentences you translated. Gurobi code must write as one line and not writing without line breaks({{\n}}).
    Take a deep breath and solve the problem step by step.
"""
    prompt2 = PromptTemplate(input_variables=["formulation","answer"],template=template2)
    gurobi_code = prompt2.format(formulation=Python_formulation,answer=groundtruth)
    result2 = llm1.invoke(gurobi_code)
    # print(result2.content)
    pattern = r'model.setObjective\((.*?)\s*GRB\.(MINIMIZE|MAXIMIZE)'
    match = re.search(pattern,str(result2))
    global temp_text
    temp_text = parsing_text(match)
    temp_text = remove_backslashes_from_getvar(temp_text)
    invoke_count = 0
    error_count = 0
    while True:
    # 获取用户输入
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
        with open(r'/Users/admin/Documents/jupyter/AAAI/experiment0720/distance_matrix.pickle', 'rb') as f:  #供给区到需求区的距离矩阵
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

        try:
            # 尝试执行用户输入的内容
            print('global obj:',temp_text)
            error_count += 1 
            exec(temp_text, {'model': model, 'gp': gp, "GRB": gp.GRB, 'I': net_demand_soc, "b": booking_fee, "theta": theta, "fee_pkm": fee_pkm, "w_hat": w_hat, "w": w, "D":demand_area, "S":supply_area, "K":soc, "d":distance_matrix, "demand_avg":demand_avg, "inventory_avg":inventory_avg})
            model.update()
            print("执行成功，目标函数可行。")
            original_objective = model.getObjective()

            new_objective = gp.LinExpr(1)

            try:
                model.setObjectiveN(original_objective, index=0, priority=1, weight=1.0, name="OriginalObjective")
                model.update()

                model.setObjectiveN(new_objective, index=1, priority=0, weight=1.0, name="NewObjective")
                model.update()
                model.dispose()
            except:
                model.dispose()
                print('set objective N error')
                play_sound('set objective N error')
                sys.exit(1) 
            break
        except Exception as e:
            # 打印错误信息并提示用户重新输入
            print(error_count)
            print('current obj:',temp_text)
            print(e)
            error_traceback = traceback.format_exc()  # 获取详细的错误堆栈信息
            invoke_count += 1
            if invoke_count > 2:
                print('错误修改失败，重新生成新的formulation')
                temp_text = ''
                new_prompt = """The previously generated code has an issue. You need to regenerate it and do not refer to the previous information at all."""
                result = llm1.invoke(new_prompt+prompt_scope_down)
                pattern = r'model.setObjective\((.*?)\s*GRB\.(MINIMIZE|MAXIMIZE)'
                match = re.search(pattern,str(result2))
                temp_text = parsing_text(match)
                temp_text = remove_backslashes_from_getvar(temp_text)
                matches = re.findall(r'「(.*?)」', str(result))

                    #print(match)
                prompt2 = PromptTemplate(input_variables=["formulation","answer"],template=template2)
                gurobi_code = prompt2.format(formulation=Python_formulation,answer=groundtruth)
                result2 = llm2.invoke(gurobi_code)
                #print(result2.content)
                pattern = r'model.setObjective\((.*?)\s*GRB\.(MINIMIZE|MAXIMIZE)'
                match = re.search(pattern,str(result2))
                #global temp_text
                temp_text = parsing_text(match)
                temp_text = remove_backslashes_from_getvar(temp_text)                
                invoke_count = 0
                continue
            elif error_count > 6:
                play_sound('indicator error!')
                sys.exit(1) 
                
            else:
                print("正在重新调试目标函数。")
                template3 = """You are a programmer specializing in operations research, meticulous and proficient in Gurobi and Python syntax. You are now required to correct the erroneous code.
                The current Gurobi code: {gurobi_code} has an error, with the detailed error information as follows: {error_traceback}. You need to modify the Gurobi code to address the errors specifically, ensuring it can run properly without altering its overall structure. The output must comply with the format requirements.
                The current parameters and decision variables are as follows:
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
                If you need to use a decision variable, follow this form: model.addVar(name=f"cluster{{i}}_cluster{{j}}_{{k}}"); model.addVar(name=f"avg_reward{{j}}_{{k}}"). But MUST NOT contain any product of decision variables in the objective function formulated by you anyway, like model.getVarByName(f'avg_reward{{j}}_{{k}}') * model.getVarByName(f'cluster{{i}}_cluster{{j}}_{{k}}')
                ---
                Attention: when you formulate an expression which include parameters: I,w,w_hat,d,demand_avg you need to use this parameters like I[i][k], w[i][j], w_hat[i][j], d[i][j], demand_avg[j] because they are array type. 
                Now I will give you some groundtruth formulations, If the content I provide you is similar to the goal, you can refer to it.
                {answer}
                Your reply should be formatted as follows: (YOU MUST STICK TO THIS FORMAT ONLY!)
                Modified Gurobi code is: “「」”
                The output format must follow the above requirements, enclosing your modified Gurobi code in “「」” symbols.
                Take a deep breath and solve the problem step by step.
                """
                prompt3 = PromptTemplate(input_variables=["gurobi_code","error_traceback","answer"],template=template3)
                gurobi_prompt = prompt3.format(gurobi_code=temp_text, error_traceback=error_traceback,answer=groundtruth)
                result3 = llm2.invoke(gurobi_prompt)
                #print(result3)
                match = re.search(pattern,str(result3))
                temp_text = parsing_text(match)
                temp_text = remove_backslashes_from_getvar(temp_text)                
                print('corrector obj',temp_text)
                #print(f"执行错误:{e}。详细错误信息为:{error_traceback}。")
    global Gurobi_formulation
    Gurobi_formulation = temp_text
    user_target_indicator = user_target
    time.sleep(3)
    return user_target_indicator

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
        Reason explanation: "「The reason for variable selection are」, so selected decision variables are: 「cluster0:[14,17],reward44:[12,14]」". This is just a format example so your answer cannot be exactly the same. Anyway, you must provide a station selection that is different from the example I provided, according to the format requirements!"""
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

tools=[recognizer,indicator,sampler]
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
    # init definition
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    global csv_path, Index
    randseed = 0  # 随机种子
    dv_num = 1032
    
    ###!!!personalized!!!###
    locality = 0.4
    Index = 'Reduce idle vehicles cost'
    In_sample = False  # 实验是否是in_sample的，True则从ground truth中抽样出index的gurobi语句和groundtruth_sample_num-1个随机指标
    groundtruth_sample_num = 8  # 从groundtruth字典中抽样放入indicator中的个数
    ###!!!personalized!!!###

    gurobi_model_path = '/Users/admin/Documents/jupyter/AAAI/experiment0720/model.mps'
    csv_path = '/Users/admin/Documents/jupyter/AAAI/experiment0720/data/total_data.csv'
    subdata_csv_path = '/Users/admin/Documents/jupyter/AAAI/experiment0720/split'
    supply_num = 42
    
    S = [0,1,2,3,5,6,8,9,10,11,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,39,40,41,42,43,45,46,47,48]
    global scope_down, reward_num, reward_per_table, cluster_num, cluster_per_table
    scope_down = dv_num*locality
    reward_num = math.floor(24/1032*scope_down)
    reward_per_table = reward_num/8
    cluster_num = math.floor((1032-24)/1032*scope_down)
    cluster_per_table = cluster_num/42

    for t in range(1):
    ###!!!personalized!!!###
        # selected = random.sample(S, math.floor(supply_num * locality))

        # user_input = Index+" at the sites ID '"+str(selected)+"'." 
        print('\n')
        print('\n')
        print('第',t,'次')
        print('\n')
        print('\n')
        expression_dict = {
        'Reduce proportion of idle vehicles': "model.setObjective(gp.quicksum((I[i][k] - gp.quicksum(model.getVarByName(f'cluster{i}_cluster{j}_{k}') for j in D)) for i in S for k in K), GRB.MINIMIZE)",
        'Reduce idle vehicles cost': "model.setObjective(gp.quicksum((k+1) * (I[i][k] - gp.quicksum(model.getVarByName(f'cluster{i}_cluster{j}_{k}') for j in D)) for i in S for k in K), GRB.MINIMIZE)",
        'Improve number of high-powered taxis in demand areas': "model.setObjective(gp.quicksum(model.getVarByName(f'cluster{i}_cluster{j}_2') for i in S for j in D) , GRB.MAXIMIZE)",
        'Improve future service Level of vehicles':"model.setObjective(gp.quicksum(k * model.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in S for j in D for k in K) , GRB.MAXIMIZE)",
        'Reduce scheduled vehicle response time': "model.setObjective(gp.quicksum(model.getVarByName(f'cluster{i}_cluster{j}_{k}') * d[i][j] for i in S for j in D for k in K), GRB.MINIMIZE)",
        'Reduce complaint rate of vehicles': "model.setObjective(gp.quicksum((k+1) * model.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in S for j in D for k in K), GRB.MAXIMIZE)",
        'Improve service Level of vehicles': "model.setObjective(gp.quicksum((k+1) * model.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in S for j in D for k in K), GRB.MAXIMIZE)",
        'Reduce average travel price of vehicles': "model.setObjective(gp.quicksum(model.getVarByName(f'avg_reward{j}_{k}') for j in D for k in K), GRB.MINIMIZE)",
        'Improve order completion rate of vehicles': "model.setObjective(gp.quicksum((k+1) * model.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in S for j in D for k in K), GRB.MAXIMIZE) ",
        'Reduce average waiting time of vehicles': "model.setObjective(gp.quicksum(model.getVarByName(f'cluster{i}_cluster{j}_{k}') * d[i][j] for i in S for j in D for k in K), GRB.MINIMIZE)",
        'Improve number of pre-allocated vehicles': "model.setObjective(gp.quicksum(model.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in S for j in D for k in K), GRB.MAXIMIZE)",
        'Improve average passenger capacity of vehicles': "model.setObjective(gp.quicksum((k+1) * model.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in S for j in D for k in K), GRB.MAXIMIZE)",
        'Improve number of users covered by vehicles': "model.setObjective(gp.quicksum((k+1) * model.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in S for j in D for k in K), GRB.MAXIMIZE)",
        'Improve user satisfaction of vehicles': "model.setObjective(gp.quicksum((k+1) * model.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in S for j in D for k in K), GRB.MAXIMIZE)",
        'Improve demand satisfaction rate': "model.setObjective(gp.quicksum((k+1) * model.getVarByName(f'cluster{i}_cluster{j}_{k}') for i in S for j in D for k in K), GRB.MAXIMIZE)"
    }
        global groundtruth
        groundtruth = sample_expressions(groundtruth_sample_num, In_sample, Index, expression_dict)

        # Create agent
        agent = agent_prompt()
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        input_target = '「' + str(Index) + '」'

        input_prompt = """
        The text within "「」" marks my target, which should include the domain of application and the metrics I am concerned with. You need to use the tool to accomplish the following tasks sequentially:
        1. You need to extract the domain of action and the desired metrics based on {input}, returning them as each tool's requirements.
        2. Based on the extracted metrics, obtain possible formulation and iterate to test these formulation until it is  feasible, returning results as per the tool's requirements.
        3. Based on the extracted metrics, and using data recorded in the backend database, identify station IDs that have little impact on the metrics and parameterize them, returning results as per the tool's requirements.
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
            play_sound(f'error')
            sys.exit(1) 
        chat_history.extend([HumanMessage(content=input_combine),
                             AIMessage(content=result["output"])
                             ])
        ###
        global LEO_optimize_time, solve_result, temp_text, Selected_numbers
        try:
            # model_add_newobj = add_new_obj(gurobi_model_path, Gurobi_formulation, w1=1, w2=1)
            LEO = add_new_obj(gurobi_model_path,'MAXIMIZE',Gurobi_formulation,1,1)
        except Exception as e:
            print(e)
            # 如果出错，播放警告音
            need_run = 5-t
            play_sound(f'error')
            sys.exit(1) 
        scope_down = Selected_numbers
        try:
            LEO,selected_num,selected_dv = process_string_and_add_constraints(scope_down,LEO,subdata_csv_path)
        except Exception as e:
            play_sound(f'sample error') 
            sys.exit(1)
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
        base = add_new_obj(gurobi_model_path,'MAXIMIZE',Gurobi_formulation,1,1)
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
        if In_sample:
            file_path = '/Users/admin/Documents/jupyter/AAAI/exper_sample/insample/index{}_sample{}_{}.txt'.format(Index, selected_num, tnow)
        else:
            file_path = '/Users/admin/Documents/jupyter/AAAI/exper_sample/outsample/index{}_sample{}_{}.txt'.format(Index, selected_num, tnow)
        f = open(file_path, 'w')
        #'Harvard Ave & E Pine St, E Pine St & 16th Ave, Summit Ave E & E Republican St, 2nd Ave & Pine St, REI / Yale Ave N & John St'
        # f.write('user:{}'.format(selected))
        # f.write('\n')
        print('selected_num:',selected_num)
        f.write('selected_num:{}'.format(selected_num))
        f.write('\n')
        print('selected decision variable:', selected_dv)
        f.write('selected decision variable:{}'.format(selected_dv))
        f.write('\n')
        print('Formulation_explanation:', Formulation_explanation)
        f.write('Formulation_explanation:{}'.format(Formulation_explanation))
        f.write('\n')
        print('temp_text:',temp_text)
        f.write('temp_text:{}'.format(temp_text))
        f.write('\n')
        # print('Python_formulation:', Python_formulation)
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

    # 播放运行结束的音效
    play_sound('Finish!')
