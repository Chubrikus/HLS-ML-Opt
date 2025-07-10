import pandas as pd
import numpy as np
import pickle
import os
import glob
from collections import defaultdict

def find_available_cases():
    """Автоматическое определение доступных кейсов"""
    case_dirs = glob.glob("all_cases/case_[0-9]*")  # Изменено для поиска всех кейсов
    available_cases = set()
    
    for case_dir in case_dirs:
        try:
            case_num = int(os.path.basename(case_dir).split('_')[1])
            # Проверяем наличие необходимых файлов с правильными именами
            dfg_file = os.path.join(case_dir, "DFG_case_1.txt")
            json_file = os.path.join(case_dir, "case_1_all_data.json")
            # Убираем проверку на cc файл
            if os.path.exists(dfg_file) and os.path.exists(json_file):
                available_cases.add(case_num)
                print(f"Found case {case_num} in {case_dir}")
                print(f"Files found: {os.path.basename(dfg_file)}, {os.path.basename(json_file)}")
        except Exception as e:
            print(f"Error processing directory {case_dir}: {str(e)}")
            continue
    
    available_cases = sorted(list(available_cases))
    print(f"Found {len(available_cases)} unique available cases: {available_cases}")
    return available_cases


class CustomGraph:
    """Простая замена StellarGraph для представления графов"""
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.node_features = nodes.drop('id', axis=1).values.astype(np.float32)
        
        # Создаем отображение меток узлов на индексы
        self.node_mapping = {node_id: idx for idx, node_id in enumerate(nodes['id'])}
        self.edge_index = self._create_edge_index()
        
    def _create_edge_index(self):
        """Создание списка рёбер в формате индексов"""
        edge_list = []
        for _, row in self.edges.iterrows():
            source_idx = self.node_mapping[row['source']]
            target_idx = self.node_mapping[row['target']]
            edge_list.append([source_idx, target_idx])
        return np.array(edge_list).T if edge_list else np.zeros((2, 0))
    
    def get_node_features(self):
        """Получение признаков узлов"""
        return self.node_features
    
    def get_edge_index(self):
        """Получение списка рёбер"""
        return self.edge_index
        
    def get_node_mapping(self):
        """Получение отображения меток узлов на индексы"""
        return self.node_mapping

def load_case_data(case_num):
    """Загрузка данных кейса"""
    processed_dir = os.path.join("all_cases", f"case_{case_num:02d}", "processed")
    
    try:
        # Загрузка всех необходимых файлов
        edges = pd.read_csv(os.path.join(processed_dir, "edges.pkl"))
        nodes = pd.read_csv(os.path.join(processed_dir, "nodes.pkl"))
        
        with open(os.path.join(processed_dir, "node_mapping.pkl"), 'rb') as f:
            node_mapping = pickle.load(f)
            
        with open(os.path.join(processed_dir, "multiplication_nodes.pkl"), 'rb') as f:
            mult_nodes = pickle.load(f)
            
        with open(os.path.join(processed_dir, "metrics.pkl"), 'rb') as f:
            metrics = pickle.load(f)
            
        return edges, nodes, node_mapping, mult_nodes, metrics
    except Exception as e:
        print(f"Error loading data for case {case_num}: {str(e)}")
        return None

def generate_graph_variants(case_num, base_nodes, edges, node_mapping, mult_nodes, metrics):
    """Генерация вариантов графа с разными директивами"""
    graphs = []
    targets = {
        'LUT': [], 
        'DSP': [], 
        'CP_synthesis': [],
        'CP_implementation': []
    }
    
    # Создание вариантов графа для каждой конфигурации
    for i in range(len(metrics['LUT_op'])):
        nodes = base_nodes.copy()
        
        # Обновление признака f10 для узлов умножения
        f10 = [0] * len(nodes)
        for op in metrics['LUT_op'][i]:
            # Преобразуем номер операции в строку вида 'm{op}'
            node_id = f"m{op}"
            node_idx = nodes.index[nodes['id'] == node_id].tolist()
            if node_idx and node_idx[0] in mult_nodes:
                f10[node_idx[0]] = 10
                
        nodes['f10'] = f10
        
        # Создание графа
        graph = CustomGraph(nodes, edges)
        graphs.append(graph)
            
        # Сохранение целевых метрик
        targets['LUT'].append(metrics['LUT'][i])
        targets['DSP'].append(metrics['DSP'][i])
        targets['CP_synthesis'].append(metrics['CP_synthesis'][i])
        targets['CP_implementation'].append(metrics['CP_implementation'][i])
    
    print(f"Successfully created {len(graphs)} graph variants")
    return graphs, targets

def main():
    """Основная функция для создания набора данных графов"""
    # Создаем директорию для выходных данных
    output_dir = "outputs/processed_data"
    os.makedirs(output_dir, exist_ok=True)
    
    case_index = find_available_cases()
    all_graphs = []
    all_targets = {
        'LUT': [], 
        'DSP': [], 
        'CP_synthesis': [],
        'CP_implementation': []
    }
    
    for case_num in case_index:
        print(f"Processing case {case_num}")
        
        try:
            # Загрузка данных кейса
            result = load_case_data(case_num)
            if result is None:
                continue
                
            edges, nodes, node_mapping, mult_nodes, metrics = result
            
            # Генерация вариантов графа
            case_graphs, case_targets = generate_graph_variants(
                case_num, nodes, edges, node_mapping, mult_nodes, metrics
            )
            
            all_graphs.extend(case_graphs)
            all_targets['LUT'].extend(case_targets['LUT'])
            all_targets['DSP'].extend(case_targets['DSP'])
            all_targets['CP_synthesis'].extend(case_targets['CP_synthesis'])
            all_targets['CP_implementation'].extend(case_targets['CP_implementation'])
            
            print(f"Added {len(case_graphs)} graphs from case {case_num}")
        except Exception as e:
            print(f"Error processing case {case_num}: {str(e)}")
            continue
    
    print(f"Total number of graphs generated: {len(all_graphs)}")
    
    # Сохранение набора данных графов
    with open(os.path.join(output_dir, 'graph_dataset.pkl'), 'wb') as f:
        pickle.dump(all_graphs, f)
        
    # Сохранение целевых метрик в отдельные файлы
    pd.DataFrame({'LUT': all_targets['LUT']}).to_csv(os.path.join(output_dir, 'graph_target_lut.csv'), index=False)
    pd.DataFrame({'DSP': all_targets['DSP']}).to_csv(os.path.join(output_dir, 'graph_target_dsp.csv'), index=False)
    pd.DataFrame({'CP': all_targets['CP_synthesis']}).to_csv(os.path.join(output_dir, 'graph_target_cp_synthesis.csv'), index=False)
    pd.DataFrame({'CP': all_targets['CP_implementation']}).to_csv(os.path.join(output_dir, 'graph_target_cp_implementation.csv'), index=False)
    
    print(f"\nSaved all files to {output_dir}/")
    print(f"Target metrics statistics:")
    print(f"LUT values: {len(all_targets['LUT'])}")
    print(f"DSP values: {len(all_targets['DSP'])}")
    print(f"CP synthesis values: {len(all_targets['CP_synthesis'])}")
    print(f"CP implementation values: {len(all_targets['CP_implementation'])}")

if __name__ == "__main__":
    main() 