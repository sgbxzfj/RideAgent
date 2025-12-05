import os
import pandas as pd
import re
import math

def compute_mean_ci(series):
    n = len(series)
    if n == 0:
        return float('nan'), 0.0
    mean_val = series.mean()
    if n > 1:
        ci = 1.96 * series.std(ddof=1) / math.sqrt(n)
    else:
        ci = 0.0
    return mean_val, ci

def format_mean_ci(mean_val, ci_val, as_percent=False):
    if pd.isna(mean_val):
        return ""
    suffix = "%" if as_percent else ""
    if ci_val == 0.0:
        return f"{mean_val:.2f}{suffix}"
    else:
        return f"{mean_val:.2f}{suffix} ± {ci_val:.2f}{suffix}"

if __name__ == '__main__':
    # DataFrame with new 'scenario' column
    result_all_df = pd.DataFrame(columns=[
        'scenario', 'param', 'txt_id',
        'LEO-CPU time', 'Gurobi-CPU time',
        'obj_gap1(%)', 'obj_gap2(%)'
    ])
    
    original_file_folder = '/Users/admin/Documents/原文稿/jupyter/AAAI/exper_nonlinear_new/'

    for file in os.listdir(original_file_folder):
        if not file.endswith(".txt"):
            continue

        file_path = os.path.join(original_file_folder, file)
        try:
            with open(file_path, encoding='gb18030', errors='ignore') as f:
                text = f.read()
        except Exception as e:
            print(f"Read error: {file} - {e}")
            continue

        # === Step 1: Extract scenario (after 'index') and param ===
        if not file.startswith('index'):
            print(f"File does not start with 'index': {file}")
            continue

        # Remove '.txt'
        base_name = file[:-4]  # e.g., "indexMarket share of vehicles_sample0.2_1753588448.2299972"

        # Split by '_sample' — this is the key separator!
        if '_sample' not in base_name:
            print(f"No '_sample' found in filename: {file}")
            continue

        # Split into [prefix, rest]
        prefix_part, sample_part = base_name.split('_sample', 1)
        # prefix_part = "indexMarket share of vehicles"
        scenario = prefix_part[5:]  # remove 'index' → "Market share of vehicles"

        # Now extract param from sample_part, which looks like "0.2_1753588448.2299972"
        param_str = sample_part.split('_', 1)[0]  # get "0.2"
        try:
            param_val = float(param_str)
        except ValueError:
            print(f"Invalid param in {file}: {param_str}")
            continue

        # === Step 2: Extract metrics from text ===
        # LEO time
        leo_match = re.search(r'LEO_optimize_time:\s*([\d.]+)', text)
        gurobi_match = re.search(r'base_optimize_time:\s*([\d.]+)', text)
        gap1_match = re.search(r'Optimization gap 1:\s*([\d.]+)%', text)
        gap2_match = re.search(r'Optimization gap 2:\s*([\d.]+)%', text)

        if not all([leo_match, gurobi_match, gap1_match, gap2_match]):
            print(f"Missing metric in {file}")
            continue

        LEO_time = float(leo_match.group(1))
        gurobi_time = float(gurobi_match.group(1))
        gap1 = float(gap1_match.group(1))
        gap2 = float(gap2_match.group(1))

        # txt_id: the timestamp part or full suffix
        txt_id = sample_part.split('_', 1)[1] if '_' in sample_part else sample_part

        # === Append row ===
        result_all_df.loc[len(result_all_df)] = [
            scenario, param_val, txt_id,
            LEO_time, gurobi_time,
            gap1, gap2
        ]

    # --- Save raw data ---
    raw_output = '/Users/admin/Documents/原文稿/jupyter/AAAI/nonlinear_result.xlsx'
    result_all_df.to_excel(raw_output, index=False)
    print(f"✅ Raw data saved to {raw_output}")

    # --- Summary: group by ['scenario', 'param'] ---
    summary_rows = []
    grouped = result_all_df.groupby(['scenario', 'param'])
    for (scenario, param_val), group in grouped:
        n = len(group)
        leo_mean, leo_ci = compute_mean_ci(group['LEO-CPU time'])
        gurobi_mean, gurobi_ci = compute_mean_ci(group['Gurobi-CPU time'])
        gap1_mean, gap1_ci = compute_mean_ci(group['obj_gap1(%)'])
        gap2_mean, gap2_ci = compute_mean_ci(group['obj_gap2(%)'])

        summary_rows.append({
            'scenario': scenario,
            'param': param_val,
            'count': n,
            'LEO-CPU time (含±CI)': format_mean_ci(leo_mean, leo_ci),
            'Gurobi-CPU time (含±CI)': format_mean_ci(gurobi_mean, gurobi_ci),
            'Obj_GAP1 (含±CI)': format_mean_ci(gap1_mean, gap1_ci, as_percent=True),
            'Obj_GAP2 (含±CI)': format_mean_ci(gap2_mean, gap2_ci, as_percent=True)
        })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        # Sort by scenario, then param
        summary_df = summary_df.sort_values(['scenario', 'param']).reset_index(drop=True)
        summary_output = '/Users/admin/Documents/原文稿/jupyter/AAAI/nonlinear_summary.xlsx'
        summary_df.to_excel(summary_output, index=False)
        print(f"✅ Summary saved to {summary_output}")
        print("\n📊 Summary (grouped by scenario and param):")
        print(summary_df.to_string(index=False))
    else:
        print("⚠️ No valid data to summarize.")