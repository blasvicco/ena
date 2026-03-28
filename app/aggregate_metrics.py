import os
import math
from collections import defaultdict

def mean(data):
    return sum(data) / len(data) if len(data) > 0 else 0

def std(data):
    if len(data) <= 1:
        return 0
    m = mean(data)
    variance = sum((x - m) ** 2 for x in data) / len(data)
    return math.sqrt(variance)

def parse_txt_file(filepath):
    """Parses a single outputs.txt file."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data = {'Training': defaultdict(dict), 'Test': defaultdict(dict)}
    phase = None

    for line in lines:
        line = line.strip()
        if line.startswith('Training'):
            phase = 'Training'
        elif line.startswith('Test'):
            phase = 'Test'
        elif line.startswith('eca_agent_'):
            parts = line.split()
            agent_name = parts[0]
            if phase == 'Training':
                data[phase][agent_name] = {
                    'Env_0_Avg': float(parts[1]),
                    'Env_1_Avg': float(parts[2]),
                    'Env_2_Avg': float(parts[3]),
                    'Reliability (%)': float(parts[4].strip('%')),
                    'Stability Gap': float(parts[5]),
                    'Efficiency': float(parts[6]) if len(parts) > 6 else 0.0
                }
            elif phase == 'Test':
                data[phase][agent_name] = {
                    'Env_0_Avg': float(parts[1]),
                    'Env_1_Avg': float(parts[2]),
                    'Env_2_Avg': float(parts[3]),
                    'Env_3_Avg': float(parts[4]),
                    'Reliability (%)': float(parts[5].strip('%')),
                    'Stability Gap': float(parts[6])
                }
    return data

def main():
    all_data = {'Training': defaultdict(lambda: defaultdict(list)), 'Test': defaultdict(lambda: defaultdict(list))}

    for i in range(1, 11):
        dirname = f"exp_{i:02d}"
        filepath = os.path.join(dirname, "outputs.txt")
        if os.path.exists(filepath):
            data = parse_txt_file(filepath)
            for phase in ['Training', 'Test']:
                for agent, metrics in data[phase].items():
                    for metric, value in metrics.items():
                        all_data[phase][agent][metric].append(value)
        else:
            print(f"Warning: {filepath} not found.")

    with open('outputs.txt', 'w') as out_f:
        out_f.write("Averages and Standard Deviations Across 10 Experiments\n")
        out_f.write("=" * 60 + "\n\n")

        for phase in ['Training', 'Test']:
            out_f.write(f"--- {phase} Phase ---\n")
            for agent, metrics in all_data[phase].items():
                out_f.write(f"Agent: {agent}\n")
                for metric, values in metrics.items():
                    if len(values) > 0:
                        mean_val = mean(values)
                        std_val = std(values)
                        min_val = min(values)
                        max_val = max(values)
                        out_f.write(f"  {metric:.<20}: Mean = {mean_val:8.2f} | Std = {std_val:8.2f} | Min = {min_val:8.2f} | Max = {max_val:8.2f}\n")
                out_f.write("\n")

if __name__ == '__main__':
    main()
