import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import pickle
import glob
import csv

def load_processed_data(data_dir):
    """Загрузка обработанных данных"""
    try:
        # Загрузка целевых метрик
        lut_df = pd.read_csv(os.path.join(data_dir, 'graph_target_lut.csv'))
        dsp_df = pd.read_csv(os.path.join(data_dir, 'graph_target_dsp.csv'))
        cp_syn_df = pd.read_csv(os.path.join(data_dir, 'graph_target_cp_synthesis.csv'))
        cp_impl_df = pd.read_csv(os.path.join(data_dir, 'graph_target_cp_implementation.csv'))
        
        # Объединение данных
        df = pd.DataFrame({
            'LUT': lut_df['LUT'],
            'DSP': dsp_df['DSP'],
            'CP_syn': cp_syn_df['CP'],
            'CP_impl': cp_impl_df['CP']
        })
        # Преобразуем все столбцы к числовому типу
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None

def create_correlation_plots(df, metrics):
    """Создание графиков корреляций"""
    # Создаем сетку для графиков корреляций
    n = len(metrics)
    fig, axes = plt.subplots(n-1, n-1, figsize=(15, 15))
    fig.suptitle('Корреляции между метриками', fontsize=16, y=1.02)
    
    # Заполняем графики
    for i in range(n-1):
        for j in range(n-1):
            if j > i:  # Верхний треугольник - scatter plots
                ax = axes[i, j]
                sns.scatterplot(data=df, x=metrics[j+1], y=metrics[i], ax=ax, alpha=0.5)
                corr = df[metrics[i]].corr(df[metrics[j+1]])
                ax.set_title(f'r = {corr:.2f}')
            elif j < i:  # Нижний треугольник - heatmaps
                ax = axes[i, j]
                sns.kdeplot(data=df, x=metrics[j], y=metrics[i], ax=ax, cmap='viridis')
            else:  # Диагональ - гистограммы
                ax = axes[i, j]
                sns.histplot(data=df, x=metrics[i], ax=ax, kde=True)
                ax.set_title(metrics[i])
    
    plt.tight_layout()
    plt.savefig('outputs/data_analysis/correlations_detailed.png', bbox_inches='tight', dpi=300)
    plt.close()

def analyze_data(data_dir):
    """Анализ данных"""
    # Загрузка данных
    df = load_processed_data(data_dir)
    if df is None:
        print("Нет данных для анализа")
        return None, None, None
    
    # Основные метрики для анализа
    metrics = ['LUT', 'DSP', 'CP_syn', 'CP_impl']
    
    # Основные статистики
    stats = {}
    for metric in metrics:
        stats[metric] = {
            'mean': df[metric].mean(),
            'median': df[metric].median(),
            'std': df[metric].std(),
            'min': df[metric].min(),
            'max': df[metric].max(),
            'q1': df[metric].quantile(0.25),
            'q3': df[metric].quantile(0.75),
            'skew': df[metric].skew(),
            'kurtosis': df[metric].kurtosis()
        }
    
    # Корреляции
    correlations = df[metrics].corr()
    
    # Визуализация распределений
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        sns.histplot(data=df, x=metric, kde=True)
        plt.title(f'Распределение {metric}')
        plt.axvline(df[metric].mean(), color='r', linestyle='--', label='Среднее')
        plt.axvline(df[metric].median(), color='g', linestyle='--', label='Медиана')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('outputs/data_analysis/data_distributions.png')
    plt.close()
    
    # Матрица корреляций
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0, fmt='.3f')
    plt.title('Матрица корреляций')
    plt.tight_layout()
    plt.savefig('outputs/data_analysis/correlation_matrix.png')
    plt.close()
    
    # Создание детальных графиков корреляций
    create_correlation_plots(df, metrics)
    
    # Формирование текстового отчета
    report = []
    report.append("=== Анализ данных ===\n")
    report.append(f"Общее количество решений: {len(df)}\n")
    
    for metric in metrics:
        report.append(f"\n--- {metric} ---")
        report.append(f"Среднее: {stats[metric]['mean']:.2f}")
        report.append(f"Медиана: {stats[metric]['median']:.2f}")
        report.append(f"Стандартное отклонение: {stats[metric]['std']:.2f}")
        report.append(f"Минимум: {stats[metric]['min']:.2f}")
        report.append(f"Максимум: {stats[metric]['max']:.2f}")
        report.append(f"Q1: {stats[metric]['q1']:.2f}")
        report.append(f"Q3: {stats[metric]['q3']:.2f}")
        report.append(f"Асимметрия: {stats[metric]['skew']:.2f}")
        report.append(f"Эксцесс: {stats[metric]['kurtosis']:.2f}")
    
    report.append("\n=== Корреляции ===")
    report.append(correlations.round(3).to_string())
    
    report.append("\n=== Экстремальные значения ===")
    for metric in metrics:
        report.append(f"\nТоп 5 максимальных значений {metric}:")
        top_5 = df.nlargest(5, metric)[[metric]]
        report.append(top_5.to_string())
        
        report.append(f"\nТоп 5 минимальных значений {metric}:")
        bottom_5 = df.nsmallest(5, metric)[[metric]]
        report.append(bottom_5.to_string())
    
    # Сохранение отчета
    with open('outputs/data_analysis/analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    # Сохранение статистики в CSV
    stats_df = pd.DataFrame(stats).T
    stats_df.to_csv('outputs/data_analysis/data_statistics.csv', encoding='utf-8')
    
    # Сохранение корреляций в CSV
    correlations.to_csv('outputs/data_analysis/correlations.csv', encoding='utf-8')
    
    # Сохранение экстремальных значений
    extremes = {}
    for metric in metrics:
        extremes[f'{metric}_top5'] = df.nlargest(5, metric)[metric].tolist()
        extremes[f'{metric}_bottom5'] = df.nsmallest(5, metric)[metric].tolist()
    pd.DataFrame(extremes).to_csv('outputs/data_analysis/extreme_values.csv', encoding='utf-8')
    
    return stats, correlations, df

def analyze_graph_structures(all_cases_dir, report_path):
    structure_stats = []
    case_dirs = sorted(glob.glob(os.path.join(all_cases_dir, 'case_*')))
    for case_dir in case_dirs:
        nodes_path = os.path.join(case_dir, 'processed', 'nodes.pkl')
        if not os.path.exists(nodes_path):
            continue
        try:
            df = pd.read_csv(nodes_path)
            total_nodes = len(df)
            n_inputs = df['id'].str.startswith('in').sum()
            n_outputs = df['id'].str.startswith('o').sum()
            n_ops = df['id'].str.startswith('m').sum()
            # Определяем операции сложения и умножения по признакам (например, f2 - add, f3 - mul, если это так)
            n_add = 0
            n_mul = 0
            if 'f2' in df.columns and 'f3' in df.columns:
                n_add = df[(df['id'].str.startswith('m')) & (df['f2'] == 1)].shape[0]
                n_mul = df[(df['id'].str.startswith('m')) & (df['f3'] == 1)].shape[0]
            structure_stats.append({
                'case': os.path.basename(case_dir),
                'total_nodes': total_nodes,
                'inputs': n_inputs,
                'outputs': n_outputs,
                'operations': n_ops,
                'add_ops': n_add,
                'mul_ops': n_mul
            })
        except Exception as e:
            print(f"Ошибка при анализе структуры графа в {case_dir}: {e}")
            continue
    # Сохраняем в CSV
    stats_df = pd.DataFrame(structure_stats)
    stats_csv_path = os.path.join('outputs/data_analysis', 'graph_structure_statistics.csv')
    stats_df.to_csv(stats_csv_path, index=False)
    # Добавляем краткую сводку в текстовый отчет
    if len(stats_df) > 0:
        summary = []
        summary.append("\n=== Структура графов ===")
        summary.append(f"Анализировано кейсов: {len(stats_df)}")
        summary.append(f"Среднее число узлов: {stats_df['total_nodes'].mean():.1f}")
        summary.append(f"Среднее число входов: {stats_df['inputs'].mean():.1f}")
        summary.append(f"Среднее число выходов: {stats_df['outputs'].mean():.1f}")
        summary.append(f"Среднее число операций: {stats_df['operations'].mean():.1f}")
        summary.append(f"Среднее число сложений: {stats_df['add_ops'].mean():.1f}")
        summary.append(f"Среднее число умножений: {stats_df['mul_ops'].mean():.1f}")
        with open(report_path, 'a', encoding='utf-8') as f:
            f.write('\n'.join(summary) + '\n')
    return stats_df

if __name__ == "__main__":
    # Путь к данным
    data_path = "outputs/processed_data"
    
    # Создание директории для результатов
    os.makedirs('outputs/data_analysis', exist_ok=True)
    
    # Анализ данных
    stats, correlations, df = analyze_data(data_path)
    print("Анализ завершен. Результаты сохранены в директории outputs/data_analysis/")
    # Новый функционал: анализ структуры графов
    analyze_graph_structures('all_cases', 'outputs/data_analysis/analysis_report.txt')
