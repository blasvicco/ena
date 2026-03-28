import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def parse_txt_file(filepath):
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
        elif line.startswith('eca_agent_') or line.startswith('ena_agent_') or line.startswith('ENA-'):
            parts = line.split()
            # Normalize agent names to paper-consistent labels
            _name_map = {
                'eca_agent_01': 'ENA-01', 'eca_agent_02': 'ENA-02', 'eca_agent_03': 'ENA-03',
                'ena_agent_01': 'ENA-01', 'ena_agent_02': 'ENA-02', 'ena_agent_03': 'ENA-03',
            }
            agent_name = _name_map.get(parts[0], parts[0])
            if phase == 'Training':
                data[phase][agent_name] = {
                    'Reliability': float(parts[4].strip('%')),
                    'Stability Gap': float(parts[5])
                }
            elif phase == 'Test':
                data[phase][agent_name] = {
                    'Reliability': float(parts[5].strip('%')),
                    'Stability Gap': float(parts[6])
                }
    return data

def main():
    records = []
    
    for i in range(1, 11):
        dirname = f"exp_{i:02d}"
        filepath = os.path.join(dirname, "outputs.txt")
        if os.path.exists(filepath):
            data = parse_txt_file(filepath)
            for agent, metrics in data['Test'].items():
                records.append({
                    'Experiment': i,
                    'Agent': agent,
                    'Reliability (%)': metrics['Reliability'],
                    'Stability Gap': metrics['Stability Gap']
                })
        else:
            print(f"Warning: {filepath} not found.")

    if not records:
        print("No data found to plot.")
        return

    df = pd.DataFrame(records)
    
    # 1. Boxplot for Reliability
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='Agent', y='Reliability (%)', palette='Set2')
    sns.stripplot(data=df, x='Agent', y='Reliability (%)', color='black', alpha=0.5, jitter=True)
    plt.title('Test Phase Reliability Distribution Across 10 Experiments')
    plt.ylabel('Test Phase Reliability (%)')
    plt.tight_layout()
    plt.savefig('reliability_box.png', dpi=300)
    plt.close()
    
    # 2. Barplot for Stability Gap
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='Agent', y='Stability Gap', capsize=.1, errorbar='sd', palette='Set1')
    sns.stripplot(data=df, x='Agent', y='Stability Gap', color='black', alpha=0.5, jitter=True)
    plt.title('Test Phase Stability Gap (Mean $\pm$ Std)')
    plt.ylabel('Score Delta (Pre - Post)')
    plt.tight_layout()
    plt.savefig('stability_bar.png', dpi=300)
    plt.close()
    
    print("Saved reliability_box.png and stability_bar.png successfully.")

if __name__ == '__main__':
    main()
