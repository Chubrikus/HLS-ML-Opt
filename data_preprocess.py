import pandas as pd
import numpy as np
import pickle
import os
import glob
import json

def find_available_cases():
    """Автоматическое определение доступных кейсов"""
    case_dirs = glob.glob("all_cases/case_[0-9]*")  # Изменено для поиска всех кейсов
    available_cases = set()
    
    for case_dir in case_dirs:
        print(f"Checking directory: {case_dir}")  # Отладочное сообщение
        try:
            case_num = int(os.path.basename(case_dir).split('_')[1])
            # Проверяем наличие только необходимых файлов
            dfg_file = os.path.join(case_dir, f"DFG_case_1.txt")
            json_file = os.path.join(case_dir, f"case_1_all_data.json")
            
            files_exist = os.path.exists(dfg_file) and os.path.exists(json_file)
            print(f"Files exist: {files_exist}")  # Отладочное сообщение
            
            if files_exist:
                available_cases.add(case_num)
                print(f"Found case {case_num} in {case_dir}")
                print(f"Files found: {os.path.basename(dfg_file)}, {os.path.basename(json_file)}")
        except Exception as e:
            print(f"Error processing directory {case_dir}: {str(e)}")
            continue
    
    available_cases = sorted(list(available_cases))
    print(f"Found {len(available_cases)} unique available cases: {available_cases}")
    return available_cases

# Автоматическое определение доступных кейсов
case_index = find_available_cases()

def to_binary(number):
    """Преобразование числа в 5-битное бинарное представление"""
    if number < 2 or number > 32:
        return 0,0,0,0,0
    binary_map = {
        2: (0,0,0,0,1),
        3: (0,0,0,1,0),
        4: (0,0,0,1,1),
        5: (0,0,1,0,0),
        6: (0,0,1,0,1),
        7: (0,0,1,1,0),
        8: (0,0,1,1,1),
        9: (0,1,0,0,0),
        10: (0,1,0,0,1),
        11: (0,1,0,1,0),
        12: (0,1,0,1,1),
        13: (0,1,1,0,0),
        14: (0,1,1,0,1),
        15: (0,1,1,1,0),
        16: (0,1,1,1,1),
        32: (1,1,1,1,1)
    }
    return binary_map.get(number, (0,0,0,0,0))

def load_metrics(case_dir):
    """Загрузка метрик из JSON файла"""
    json_file = os.path.join(case_dir, "case_1_all_data.json")
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Извлекаем метрики для всех решений
        lut_values = []
        dsp_values = []
        cp_synth_values = []
        cp_impl_values = []
        lut_ops = []
        
        for solution in data.values():
            if 'LUT' in solution:
                lut_values.append(solution['LUT'])
            if 'DSP' in solution:
                dsp_values.append(solution['DSP'])
            if 'CP_post_synthesis' in solution:
                cp_synth_values.append(solution['CP_post_synthesis'])
            if 'CP_post_implementation' in solution:
                cp_impl_values.append(solution['CP_post_implementation'])
            if 'LUT_op' in solution:
                lut_ops.append(solution['LUT_op'])
        
        return {
            'LUT': lut_values,
            'DSP': dsp_values,
            'CP_synthesis': cp_synth_values,
            'CP_implementation': cp_impl_values,
            'LUT_op': lut_ops
        }
    except Exception as e:
        print(f"Error loading metrics from {json_file}: {str(e)}")
        return {
            'LUT': [],
            'DSP': [],
            'CP_synthesis': [],
            'CP_implementation': [],
            'LUT_op': []
        }

def process_case(case_num):
    """Обработка одного кейса"""
    print(f"Processing case {case_num}")
    
    # Пути к файлам
    case_dir = os.path.join("all_cases", f"case_{case_num:02d}")
    dfg_file = os.path.join(case_dir, "DFG_case_1.txt")
    
    # Загрузка метрик
    metrics = load_metrics(case_dir)
    
    # Структуры для хранения информации о графе
    node_name = []  # имена узлов
    node_features = []  # характеристики узлов
    edge_source = []  # исходные узлы рёбер
    edge_end = []  # конечные узлы рёбер
    node_dir = []  # индексы узлов умножения
    node_number_mapping = {}  # маппинг имён узлов в индексы
    
    input_node = 0
    inter_node = 0
    out_node = 0
    node_count = 0
    edge_count = 0
    
    # Чтение DFG файла
    try:
        with open(dfg_file, "r") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading DFG file {dfg_file}: {str(e)}")
        return None
    
    section = ""
    for line in lines:
        line = line.strip()
        
        # Пропускаем пустые строки и комментарии
        if not line or line.startswith('#'):
            if "Primary Inputs:" in line:
                section = "inputs"
            elif "Intermediate Operations:" in line:
                section = "operations"
            elif "Edges:" in line:
                section = "edges"
            continue
            
        # Обработка входных узлов
        if section == "inputs" and line.startswith('in'):
            input_node += 1
            node_count += 1
            
            parts = line.split()
            node_id = parts[0]
            precision = int(parts[1][3:])  # Убираем 'INT' из строки
            
            node_name.append(node_id)
            p_binary = to_binary(precision)
            node_features.append([1,0,0,0,p_binary[0],p_binary[1],p_binary[2],p_binary[3],p_binary[4],0,0])
            
        # Обработка промежуточных узлов
        elif section == "operations" and line.startswith('m'):
            inter_node += 1
            node_count += 1
            
            parts = line.split()
            node_id = parts[0]
            op_type = parts[1]
            precision = int(parts[2][3:])  # Убираем 'INT' из строки
            
            node_name.append(node_id)
            p_binary = to_binary(precision)
            
            if op_type == '+':
                node_features.append([0,1,0,0,p_binary[0],p_binary[1],p_binary[2],p_binary[3],p_binary[4],0,0])
            else:  # '*'
                node_features.append([0,0,1,0,p_binary[0],p_binary[1],p_binary[2],p_binary[3],p_binary[4],1,0])
                node_dir.append(node_count-1)
                
            node_number = int(node_id[1:])
            node_number_mapping[node_number] = node_count-1
            
        # Обработка выходных узлов
        elif section == "edges" and line.startswith('o'):
            out_node += 1
            node_count += 1
            node_name.append(line)
            node_features.append([0,0,0,1,0,0,0,0,0,0,0])
            
        # Обработка рёбер
        elif section == "edges" and not line.startswith('o'):
            edge_count += 1
            parts = line.split()
            edge_source.append(parts[0])
            edge_end.append(parts[1])
    
    # Создание DataFrame для рёбер
    edge_data = pd.DataFrame({'source': edge_source, 'target': edge_end})
    
    # Создание DataFrame для метаданных
    meta_data = pd.DataFrame({
        'input': [input_node],
        'inter': [inter_node],
        'output': [out_node],
        'mul': [len(node_dir)],
        'edge': [edge_count]
    })
    
    # Создание DataFrame для узлов
    node_df = pd.DataFrame({
        'id': node_name,
        'f0': [f[0] for f in node_features],
        'f1': [f[1] for f in node_features],
        'f2': [f[2] for f in node_features],
        'f3': [f[3] for f in node_features],
        'f4': [f[4] for f in node_features],
        'f5': [f[5] for f in node_features],
        'f6': [f[6] for f in node_features],
        'f7': [f[7] for f in node_features],
        'f8': [f[8] for f in node_features],
        'f9': [f[9] for f in node_features],
        'f10': [f[10] for f in node_features]
    })
    
    # Создание директории для сохранения
    processed_dir = os.path.join(case_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Сохранение данных
    edge_data.to_csv(os.path.join(processed_dir, "edges.pkl"), index=False)
    meta_data.to_csv(os.path.join(processed_dir, "metadata.pkl"), index=False)
    node_df.to_csv(os.path.join(processed_dir, "nodes.pkl"), index=False)
    
    with open(os.path.join(processed_dir, "node_mapping.pkl"), 'wb') as f:
        pickle.dump(node_number_mapping, f)
        
    with open(os.path.join(processed_dir, "multiplication_nodes.pkl"), 'wb') as f:
        pickle.dump(node_dir, f)
        
    # Сохранение метрик
    with open(os.path.join(processed_dir, "metrics.pkl"), 'wb') as f:
        pickle.dump(metrics, f)
        
    return {
        'edges': edge_data,
        'metadata': meta_data,
        'nodes': node_df,
        'node_mapping': node_number_mapping,
        'multiplication_nodes': node_dir,
        'metrics': metrics
    }

def main():
    """Основная функция для обработки всех кейсов"""
    # Создаем директорию для выходных данных
    output_dir = "outputs/processed_data"
    os.makedirs(output_dir, exist_ok=True)
    
    all_cases_data = {}
    
    # Сбор всех метрик
    all_lut = []
    all_dsp = []
    all_cp_synth = []
    all_cp_impl = []
    
    case_index = find_available_cases()
    for case_num in case_index:
        try:
            case_data = process_case(case_num)
            if case_data is not None:
                all_cases_data[case_num] = case_data
                
                # Добавление метрик в общие списки
                if 'metrics' in case_data:
                    metrics = case_data['metrics']
                    all_lut.extend(metrics.get('LUT', []))
                    all_dsp.extend(metrics.get('DSP', []))
                    all_cp_synth.extend(metrics.get('CP_synthesis', []))
                    all_cp_impl.extend(metrics.get('CP_implementation', []))
        except Exception as e:
            print(f"Error processing case {case_num}: {str(e)}")
            continue
    
    # Сохранение общих данных
    with open(os.path.join(output_dir, 'all_cases_data.pkl'), 'wb') as f:
        pickle.dump(all_cases_data, f)
        
    # Сохранение общих метрик в отдельные файлы
    pd.DataFrame({'LUT': all_lut}).to_csv(os.path.join(output_dir, 'graph_target_lut.csv'), index=False)
    pd.DataFrame({'DSP': all_dsp}).to_csv(os.path.join(output_dir, 'graph_target_dsp.csv'), index=False)
    pd.DataFrame({'CP': all_cp_synth}).to_csv(os.path.join(output_dir, 'graph_target_cp_synthesis.csv'), index=False)
    pd.DataFrame({'CP': all_cp_impl}).to_csv(os.path.join(output_dir, 'graph_target_cp_implementation.csv'), index=False)

    print(f"\nSaved all files to {output_dir}/")
    print(f"Metrics statistics:")
    print(f"LUT values: {len(all_lut)}")
    print(f"DSP values: {len(all_dsp)}")
    print(f"CP synthesis values: {len(all_cp_synth)}")
    print(f"CP implementation values: {len(all_cp_impl)}")

if __name__ == "__main__":
    main() 