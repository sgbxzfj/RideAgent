import os
import pandas as pd
import re
import pickle
import numpy as np
import math

# @source catpaw @line_count 26
def _list_pattern_folders(root_dir):
    """列出所有存在实验结果的需求模式文件夹"""
    if not os.path.isdir(root_dir):
        return []
    return sorted(
        [
            name for name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, name))
        ]
    )

PATTERN_DATA_DIR = os.path.join('/Users/admin/Documents/原文稿/jupyter/AAAI/experiment0720/data', 'demand_pattern')
DEFAULT_PATTERN = 'base'
CSV_PATH_TEMPLATE = os.path.join(PATTERN_DATA_DIR, 'total_data_{}.csv')
LEGACY_TOTAL_DATA = os.path.join('/Users/admin/Documents/原文稿/jupyter/AAAI/experiment0720/data', 'total_data.csv')

# @source catpaw @line_count 18
def get_total_data_csv(pattern=None):
    """根据需求模式返回对应的 total_data 路径，缺省使用 base"""
    scenario = pattern or DEFAULT_PATTERN
    pattern_csv = CSV_PATH_TEMPLATE.format(scenario)
    if os.path.exists(pattern_csv):
        return pattern_csv
    if os.path.exists(LEGACY_TOTAL_DATA):
        return LEGACY_TOTAL_DATA
    raise FileNotFoundError(f"Cannot locate total_data file for pattern '{scenario}'")

# 兼容旧接口
csv_path = get_total_data_csv()
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

# @source catpaw @line_count 20
def get_pattern_factors(pattern):
    """返回给定需求模式下均值和方差的缩放系数"""
    scenario_factors = {
        "base": (1.0, 1.0),
        "mean_up_10pct": (1.1, 1.0),
        "mean_down_10pct": (0.9, 1.0),
        "std_up_10pct": (1.0, 1.1),
        "std_down_10pct": (1.0, 0.9),
        "swap_both_4": (1.0, 1.0),
        "swap_both_8": (1.0, 1.0),
    }
    if not pattern:
        return scenario_factors["base"]
    return scenario_factors.get(pattern, scenario_factors["base"])
global_dict = {'D': [4, 7, 12, 15, 37, 38, 44, 49],
               'K': [0,1,2]}
#'S': [0,1,2,3,5,6,8,9,10,11,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,39,40,41,42,43,45,46,47,48],


def generate_net_demand(area_num, demand_area, pattern=None, seed=0):
    """按照需求模式生成 net_demand_soc，与 exper_demand_pattern 一致"""
    rng = np.random.default_rng(seed)
    base_demand_avg = np.array([
        -1.34, -72.57, -427.42, -1.8, 11.87, -10.29, -2.72, 544.47, -1.67, -0.78,
        -0.41, -2.35, 392.5, -12.62, -4.96, 70.81, -1.0, -4.46, -1.33, -9.61,
        -1.67, -3.6, -4.22, -1.14, -2.17, -3.28, -3.07, -4.08, -1.0, -0.94,
        -9.84, -2.02, -1.82, -1.04, -1.0, -1.79, -91.87, 36.66, 414.2, -0.46,
        -2.29, -0.84, -1.54, -5.08, 32.29, -1.35, -1.44, -1.49, -2.8, 18.66
    ])
    base_demand_std = np.array([
        0.81, 48.52, 306.76, 1.64, 11.88, 5.19, 1.82, 259.58, 1.06, 0.65,
        0.61, 1.55, 171.99, 5.53, 3.08, 42.4, 0.0, 3.09, 2.15, 4.98,
        1.81, 4.77, 2.45, 2.19, 1.96, 9.07, 2.41, 2.86, 0.0, 0.56,
        5.26, 1.9, 1.1, 0.6, 0.41, 1.67, 112.49, 112.69, 193.06, 0.76,
        1.91, 0.37, 1.34, 3.42, 22.54, 0.96, 1.03, 2.11, 2.5, 12.25
    ])
    mean_factor, std_factor = get_pattern_factors(pattern)
    demand_avg = base_demand_avg * mean_factor
    demand_std = base_demand_std * std_factor
    taxi_storage = rng.integers(8, 10, size=area_num)
    taxi_demand = rng.normal(demand_avg, np.log(demand_std + 1), size=len(demand_avg))
    taxi_netdemand = taxi_demand - taxi_storage
    net_demand_soc = []
    for i in range(area_num):
        if i in demand_area:
            taxi_netdemand[i] = max(taxi_netdemand[i], 0)
            p_soc = np.array([
                rng.integers(0, 3),
                rng.integers(2, 6),
                rng.integers(5, 11)
            ])
        else:
            taxi_netdemand[i] = min(taxi_netdemand[i], 0)
            p_soc = np.array([
                rng.integers(0, 3),
                rng.integers(2, 6),
                rng.integers(3, 7)
            ])
        p_soc = p_soc / p_soc.sum()
        net_demand_soc.append(taxi_netdemand[i, np.newaxis] * p_soc)
    return np.stack(tuple(net_demand_soc), axis=0)


# # @source catpaw @line_count 18
# def load_net_demand_soc(pattern, area_num, demand_area, fallback_generate=True):
#     """从文件中加载 net_demand_soc，与实验脚本保持一致"""
#     if pattern:
#         net_demand_path = os.path.join(
#             PATTERN_DATA_DIR,
#             f'net_demand_soc_{pattern}.pickle'
#         )
#         if os.path.exists(net_demand_path):
#             with open(net_demand_path, 'rb') as f:
#                 return pickle.load(f)
#     if fallback_generate:
#         return generate_net_demand(area_num, demand_area, pattern=pattern)
#     raise FileNotFoundError(f"net_demand_soc file not found for pattern: {pattern}")


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


def _load_baseline_results(baseline_csv, pattern_name, column_order):
    """
    将 satisfaction_sample.csv 中的 base 结果转换为与主表一致的列格式。
    """
    if not os.path.exists(baseline_csv):
        return pd.DataFrame(columns=column_order)

    baseline_df = pd.read_csv(baseline_csv)
    expected_cols = {
        'sample',
        'index',
        'selected num',
        'LEO-CPU time',
        'Gurobi-CPU time',
        'RF-obj gap(%)',
        'QR-obj gap(%)'
    }
    if not expected_cols.issubset(set(baseline_df.columns)):
        raise ValueError(f"Baseline CSV 缺少必要列: {sorted(expected_cols - set(baseline_df.columns))}")

    baseline_df = baseline_df.copy()
    baseline_df['pattern'] = pattern_name
    baseline_df.rename(
        columns={
            'RF-obj gap(%)': 'obj_gap1(%)',
            'QR-obj gap(%)': 'obj_gap2(%)'
        },
        inplace=True
    )
    baseline_df['save_time'] = baseline_df['Gurobi-CPU time'] - baseline_df['LEO-CPU time']
    baseline_df['save_time_percent(%)'] = baseline_df.apply(
        lambda row: (row['save_time'] / row['Gurobi-CPU time'] * 100) if row['Gurobi-CPU time'] else 0.0,
        axis=1
    )
    # 重排列顺序以保持一致
    return baseline_df[column_order]


# @source catpaw @line_count 58
def _summarize_baseline_sample_bins(
    baseline_df: pd.DataFrame,
    output_csv: str = "satisfaction_rf_bins_summary.csv",
) -> None:
    """
    根据 selected num 的区间，对 baseline 结果进行聚合并输出均值±置信区间。
    区间固定为 [0,50], [50,100], [100,150], [150,250]。
    """
    if baseline_df.empty:
        return

    bin_specs = [(0, 50), (50, 100), (100, 150), (150, 250)]

    bin_labels = [f"[{low},{high}]" for low, high in bin_specs]

    def assign_bin(value: float) -> str | None:
        for (low, high), label in zip(bin_specs, bin_labels):
            if low <= value < high or (high == bin_specs[-1][1] and value <= high):
                return label
        return None

    def mean_and_ci(series: pd.Series):
        n = len(series)
        mean_val = series.mean()
        if n > 1:
            ci = 1.96 * series.std(ddof=1) / math.sqrt(n)
        else:
            ci = 0.0
        return mean_val, ci

    def format_mean_ci(mean_val, ci_val, suffix=""):
        if pd.isna(mean_val):
            return ""
        if ci_val == 0.0:
            return f"{mean_val:.2f}{suffix}"
        return f"{mean_val:.2f}{suffix} ± {ci_val:.2f}{suffix}"

    df = baseline_df.copy()
    df["selected_num_range"] = df["selected num"].apply(assign_bin)
    df = df.dropna(subset=["selected_num_range"])
    df["selected_num_range"] = pd.Categorical(
        df["selected_num_range"], categories=bin_labels, ordered=True
    )

    rows = []
    group_cols = ["sample", "selected_num_range"]
    metric_cols = [
        ("LEO-CPU time", ""),
        ("Gurobi-CPU time", ""),
        ("obj_gap1(%)", "%"),
        ("obj_gap2(%)", "%"),
        ("save_time", ""),
        ("save_time_percent(%)", "%"),
    ]

    for keys, group in df.groupby(group_cols):
        sample, num_range = keys
        entry = {
            "sample": sample,
            "selected_num_range": num_range,
            "count": len(group),
        }
        for col, suffix in metric_cols:
            mean_val, ci_val = mean_and_ci(group[col])
            entry[col] = format_mean_ci(mean_val, ci_val, suffix)
        rows.append(entry)

    if not rows:
        return

    summary_df = pd.DataFrame(rows)
    summary_df["selected_num_range"] = pd.Categorical(
        summary_df["selected_num_range"], categories=bin_labels, ordered=True
    )
    summary_df.sort_values(by=["sample", "selected_num_range"], inplace=True, ignore_index=True)
    summary_df.to_csv(output_csv, index=False, encoding="utf_8_sig")


# @source catpaw @line_count 52
def _summarize_baseline_by_index(
    baseline_df: pd.DataFrame,
    output_csv: str = "satisfaction_rf_index_summary.csv",
) -> pd.DataFrame | None:
    """
    像 pattern 汇总一样，按 sample/index 计算均值 ± 置信区间。
    """
    if baseline_df.empty:
        return None

    def mean_and_ci(series: pd.Series):
        n = len(series)
        mean_val = series.mean()
        if n > 1:
            ci = 1.96 * series.std(ddof=1) / math.sqrt(n)
        else:
            ci = 0.0
        return mean_val, ci

    def format_mean_ci(mean_val, ci_val, suffix=""):
        if pd.isna(mean_val):
            return ""
        if ci_val == 0.0:
            return f"{mean_val:.2f}{suffix}"
        return f"{mean_val:.2f}{suffix} ± {ci_val:.2f}{suffix}"

    records = []
    metric_defs = [
        ("LEO-CPU time", ""),
        ("Gurobi-CPU time", ""),
        ("obj_gap1(%)", "%"),
        ("obj_gap2(%)", "%"),
        ("save_time", ""),
        ("save_time_percent(%)", "%"),
    ]

    for (sample, index_name), group in baseline_df.groupby(["sample", "index"]):
        entry = {
            "sample": sample,
            "index": index_name,
            "count": len(group),
        }
        for column, suffix in metric_defs:
            mean_val, ci_val = mean_and_ci(group[column])
            entry[column] = format_mean_ci(mean_val, ci_val, suffix)
        records.append(entry)

    if not records:
        return None

    summary_df = pd.DataFrame(records).sort_values(
        by=["sample", "index"], ignore_index=True
    )
    summary_df.to_csv(output_csv, index=False, encoding="utf_8_sig")
    return summary_df


if __name__ == '__main__':
    columns = [
        'pattern',
        'sample',
        'index',
        'selected num',
        'LEO-CPU time',
        'Gurobi-CPU time',
        'obj_gap1(%)',
        'obj_gap2(%)',
        'save_time',
        'save_time_percent(%)'
    ]
    result_all_df = pd.DataFrame(columns=columns)
    root_dir = 'exper_rf'
    pattern_folders = _list_pattern_folders(root_dir)
    pattern_tables = {}
    for pattern in pattern_folders:
        for flag in ['insample', 'outsample']:
            original_file_folder = os.path.join(root_dir, pattern, flag)
            if not os.path.isdir(original_file_folder):
                continue
            for file in os.listdir(original_file_folder):
                if not file.endswith(".txt"):
                    continue
                file_path = os.path.join(original_file_folder, file)
                with open(file_path, 'r') as f:
                    text = f.read()
                params = file.split('_')
                index = params[0][5:]
                position_1 = text.find('selected_num:')
                position_2 = text.find('selected decision')
                selected_num_text = text[position_1:position_2]
                selected_num = int(re.findall("\d+", selected_num_text)[0])
                p_LEO_time_text = text.find('LEO_optimize_time:')
                p_LEO_result_text = text.find('LEO_result:')
                LEO_time_text = text[p_LEO_time_text:p_LEO_result_text]
                LEO_time = float(re.findall("\d+\.?\d*", LEO_time_text)[0])
                p_gurobi_time_text = text.find('base_optimize_time:')
                p_gurobi_result_text = text.find('base_result:')
                gurobi_time_text = text[p_gurobi_time_text:p_gurobi_result_text]
                gurobi_time = float(re.findall("\d+\.?\d*", gurobi_time_text)[0])
                gap1_position = text.find('Optimization gap 1:')
                gap2_position = text.find('Optimization gap 2:')
                gap_end_position = text.find('LEO Model Variables and Values:')
                gap1_text = text[gap1_position:gap2_position]
                gap1_str = re.findall('\d+\.?\d*%', gap1_text)
                gap1 = float(gap1_str[0][:-1])
                gap2_text = text[gap2_position:gap_end_position]
                gap2_str = re.findall('\d+\.?\d*%', gap2_text)
                gap2 = float(gap2_str[0][:-1])
                save_time = gurobi_time - LEO_time
                save_time_percent = (save_time / gurobi_time * 100) if gurobi_time else 0.0
                row = [pattern, flag, index, selected_num, LEO_time, gurobi_time, gap1, gap2, save_time, save_time_percent]
                result_all_df.loc[len(result_all_df)] = row
                pattern_tables.setdefault(pattern, []).append(row[1:])  # exclude pattern column for sheet tables
    result_all_df.to_csv('satisfaction_rf.csv', index=False, encoding="utf_8_sig")

    # 追加 baseline（base 场景）结果
    baseline_csv = 'satisfaction_sample_copy.csv'
    baseline_pattern = 'base'
    baseline_df = _load_baseline_results(baseline_csv, baseline_pattern, columns)
    if not baseline_df.empty:
        result_all_df = pd.concat([result_all_df, baseline_df], ignore_index=True)
        for _, baseline_row in baseline_df.iterrows():
            pattern_tables.setdefault(baseline_pattern, []).append(
                [baseline_row[col] for col in columns[1:]]
            )
        if baseline_pattern not in pattern_folders:
            pattern_folders.append(baseline_pattern)
        _summarize_baseline_sample_bins(baseline_df)
        baseline_index_summary_df = _summarize_baseline_by_index(baseline_df)
    else:
        baseline_index_summary_df = None

    def _compute_summary(sample_label):
        sample_df = result_all_df[result_all_df['sample'] == sample_label]
        if sample_df.empty:
            return None

        def mean_and_ci(series):
            n = len(series)
            mean_val = series.mean()
            if n > 1:
                ci = 1.96 * series.std(ddof=1) / math.sqrt(n)
            else:
                ci = 0.0
            return mean_val, ci

        def format_mean_ci(mean_val, ci_val, suffix=''):
            if pd.isna(mean_val):
                return ''
            if ci_val == 0.0:
                return f'{mean_val:.2f}{suffix}'
            return f'{mean_val:.2f}{suffix} ± {ci_val:.2f}{suffix}'

        rows = []
        for pattern in pattern_folders:
            group = sample_df[sample_df['pattern'] == pattern]
            if group.empty:
                continue
            count = len(group)
            obj1_mean, obj1_ci = mean_and_ci(group['obj_gap1(%)'])
            obj2_mean, obj2_ci = mean_and_ci(group['obj_gap2(%)'])
            leo_mean, leo_ci = mean_and_ci(group['LEO-CPU time'])
            gurobi_mean, gurobi_ci = mean_and_ci(group['Gurobi-CPU time'])
            save_mean, save_ci = mean_and_ci(group['save_time'])
            save_pct_value = (save_mean / gurobi_mean * 100) if gurobi_mean else 0.0
            _, save_pct_ci = mean_and_ci(group['save_time_percent(%)'])

            prefix = f'{sample_label.capitalize()}'
            ci_label = ' (含±CI)'
            rows.append({
                'pattern': pattern,
                f'{prefix}_count': count,
                f'{prefix}_Obj_GAP1{ci_label}': format_mean_ci(obj1_mean, obj1_ci, '%'),
                f'{prefix}_Obj_GAP2{ci_label}': format_mean_ci(obj2_mean, obj2_ci, '%'),
                f'{prefix}_LEO_time{ci_label}': format_mean_ci(leo_mean, leo_ci),
                f'{prefix}_gurobi_time{ci_label}': format_mean_ci(gurobi_mean, gurobi_ci),
                f'{prefix}_save_time{ci_label}': format_mean_ci(save_mean, save_ci),
                f'{prefix}_save_time_percent{ci_label}': format_mean_ci(save_pct_value, save_pct_ci, '%')
            })
        if not rows:
            return None
        summary_df = pd.DataFrame(rows)
        return summary_df

    insample_summary_df = _compute_summary('insample')
    outsample_summary_df = _compute_summary('outsample')

    if insample_summary_df is not None:
        insample_summary_df.to_csv('satisfaction_rf_insample_summary.csv', index=False, encoding="utf_8_sig")
    if outsample_summary_df is not None:
        outsample_summary_df.to_csv('satisfaction_rf_outsample_summary.csv', index=False, encoding="utf_8_sig")

    if pattern_tables or insample_summary_df is not None or outsample_summary_df is not None:
        def _sanitize_sheet_name(name):
            sanitized = re.sub(r'[^0-9A-Za-z_]', '_', name)
            return sanitized[:31] or "pattern"

        with pd.ExcelWriter('satisfaction_rf.xlsx', engine='xlsxwriter') as writer:
            if pattern_tables:
                for pattern, rows in pattern_tables.items():
                    df = pd.DataFrame(rows, columns=columns[1:])
                    sheet_name = _sanitize_sheet_name(pattern)
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            if insample_summary_df is not None:
                insample_summary_df.to_excel(writer, sheet_name='insample_summary', index=False)
            if outsample_summary_df is not None:
                outsample_summary_df.to_excel(writer, sheet_name='outsample_summary', index=False)
            if baseline_index_summary_df is not None:
                baseline_index_summary_df.to_excel(writer, sheet_name='baseline_index_summary', index=False)
