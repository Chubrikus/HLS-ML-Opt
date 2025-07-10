import pickle
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class CustomGraph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.node_features = nodes.drop('id', axis=1).values.astype(np.float32)

def visualize_graph(graph_idx=0):
    # Загрузка данных
    with open('outputs/processed_data/graph_dataset.pkl', 'rb') as f:
        graphs = pickle.load(f)
    
    # Получение графа
    graph = graphs[graph_idx]
    
    # Создание графа для визуализации
    G = nx.DiGraph()
    
    # Добавление узлов
    for idx, row in graph.nodes.iterrows():
        node_id = row['id']
        node_type = 'input' if node_id.startswith('in') else 'output' if node_id.startswith('o') else 'mult'
        G.add_node(node_id, node_type=node_type)
    
    # Добавление рёбер
    for _, row in graph.edges.iterrows():
        G.add_edge(row['source'], row['target'])
    
    # Визуализация
    plt.figure(figsize=(15, 10))
    
    # Позиции узлов
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Цвета узлов
    node_colors = []
    for node in G.nodes():
        if node.startswith('in'):
            node_colors.append('green')  # Входы - зеленый
        elif node.startswith('o'):
            node_colors.append('red')    # Выходы - красный
        else:  # Промежуточные узлы
            # Проверяем тип операции
            if graph.nodes.loc[graph.nodes['id'] == node, 'f1'].iloc[0] == 1:
                node_colors.append('lightblue')  # Сложение - голубой
            else:
                node_colors.append('blue')       # Умножение - синий
    
    # Рисование графа
    nx.draw(G, pos, 
            with_labels=True, 
            node_color=node_colors,
            node_size=500,
            font_size=8,
            font_weight='bold',
            arrows=True,
            arrowsize=20)
    
    # Легенда
    plt.title(f'Граф {graph_idx}\nЗеленый - входы, Голубой - сложения, Синий - умножения, Красный - выходы')
    
    # Сохранение
    plt.savefig(f'outputs/data_analysis/graph_{graph_idx}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Вывод информации о графе
    print(f"\nИнформация о графе {graph_idx}:")
    print(f"Количество узлов: {len(G.nodes)}")
    print(f"Количество рёбер: {len(G.edges)}")
    print("\nТипы узлов:")
    print(f"Входы: {sum(1 for node in G.nodes if node.startswith('in'))}")
    print(f"Умножения: {sum(1 for node in G.nodes if node.startswith('m'))}")
    print(f"Выходы: {sum(1 for node in G.nodes if node.startswith('o'))}")
    
    # Добавляем информацию о типах операций
    mult_nodes = [node for node in G.nodes if node.startswith('m')]
    add_ops = sum(1 for node in mult_nodes if graph.nodes.loc[graph.nodes['id'] == node, 'f1'].iloc[0] == 1)
    mul_ops = sum(1 for node in mult_nodes if graph.nodes.loc[graph.nodes['id'] == node, 'f2'].iloc[0] == 1)
    
    print("\nТипы операций:")
    print(f"Сложение: {add_ops}")
    print(f"Умножение: {mul_ops}")
    
    # Загрузка целевых значений
    lut_df = pd.read_csv('outputs/processed_data/graph_target_lut.csv')
    dsp_df = pd.read_csv('outputs/processed_data/graph_target_dsp.csv')
    cp_syn_df = pd.read_csv('outputs/processed_data/graph_target_cp_synthesis.csv')
    cp_impl_df = pd.read_csv('outputs/processed_data/graph_target_cp_implementation.csv')
    
    print("\nЦелевые значения:")
    print(f"LUT: {lut_df.iloc[graph_idx]['LUT']}")
    print(f"DSP: {dsp_df.iloc[graph_idx]['DSP']}")
    print(f"CP_syn: {cp_syn_df.iloc[graph_idx]['CP']}")
    print(f"CP_impl: {cp_impl_df.iloc[graph_idx]['CP']}")

    print("\nСтруктура соединений:")
    # Подсчет входных соединений для промежуточных узлов
    input_connections = {}
    for _, row in graph.edges.iterrows():
        target = row['target']
        if target.startswith('m'):
            if target not in input_connections:
                input_connections[target] = 0
            input_connections[target] += 1
    
    # Подсчет выходных соединений для промежуточных узлов
    output_connections = {}
    for _, row in graph.edges.iterrows():
        source = row['source']
        if source.startswith('m'):
            if source not in output_connections:
                output_connections[source] = 0
            output_connections[source] += 1
    
    print("\nСтатистика соединений:")
    print(f"Максимальное количество входов у узла: {max(input_connections.values())}")
    print(f"Среднее количество входов: {sum(input_connections.values()) / len(input_connections):.2f}")
    print(f"Максимальное количество выходов у узла: {max(output_connections.values())}")
    print(f"Среднее количество выходов: {sum(output_connections.values()) / len(output_connections):.2f}")

def print_detailed_info(graph_idx=0):
    # Загрузка данных
    with open('outputs/processed_data/graph_dataset.pkl', 'rb') as f:
        graphs = pickle.load(f)
    
    # Получение графа
    graph = graphs[graph_idx]
    
    print("\nСтруктура узлов (первые 5):")
    print(graph.nodes.head())
    
    print("\nСтруктура рёбер (первые 5):")
    print(graph.edges.head())
    
    print("\nПризнаки узлов (размерность):")
    print(f"Форма: {graph.node_features.shape}")

def visualize_unraveled_graph(graph_idx=0):
    # Загрузка данных
    with open('outputs/processed_data/graph_dataset.pkl', 'rb') as f:
        graphs = pickle.load(f)
    
    # Получение графа
    graph = graphs[graph_idx]
    
    # Создание графа для визуализации
    G = nx.DiGraph()
    
    # Добавление узлов
    for idx, row in graph.nodes.iterrows():
        node_id = row['id']
        node_type = 'input' if node_id.startswith('in') else 'output' if node_id.startswith('o') else 'mult'
        G.add_node(node_id, node_type=node_type)
    
    # Добавление рёбер
    for _, row in graph.edges.iterrows():
        G.add_edge(row['source'], row['target'])
    
    # Определяем слои для каждого узла
    node_layers = {}
    max_layer = 0
    
    for node in G.nodes():
        if node.startswith('in'):
            node_layers[node] = 0
        else:
            # Находим максимальный слой среди предшественников
            predecessors = list(G.predecessors(node))
            if predecessors:
                max_pred_layer = max([node_layers.get(pred, 0) for pred in predecessors])
                node_layers[node] = max_pred_layer + 1
            else:
                node_layers[node] = 0
            max_layer = max(max_layer, node_layers[node])
    
    # Размещаем узлы по слоям с учетом их типа
    pos = {}
    layer_height = 2.0  # Расстояние между слоями
    node_spacing = 1.0  # Расстояние между узлами в слое
    
    # Сначала размещаем входные узлы
    input_nodes = [n for n in G.nodes() if n.startswith('in')]
    for i, node in enumerate(input_nodes):
        pos[node] = (0, -len(input_nodes)/2 + i)
    
    # Затем размещаем промежуточные и выходные узлы
    for layer in range(1, max_layer + 2):  # +2 для выходных узлов
        nodes_in_layer = [n for n, l in node_layers.items() if l == layer]
        
        # Разделяем узлы по типам операций
        add_nodes = [n for n in nodes_in_layer if n.startswith('m') and 
                    graph.nodes.loc[graph.nodes['id'] == n, 'f1'].iloc[0] == 1]
        mul_nodes = [n for n in nodes_in_layer if n.startswith('m') and 
                    graph.nodes.loc[graph.nodes['id'] == n, 'f2'].iloc[0] == 1]
        out_nodes = [n for n in nodes_in_layer if n.startswith('o')]
        
        # Размещаем узлы сложения
        for i, node in enumerate(add_nodes):
            pos[node] = (layer * layer_height, -len(nodes_in_layer)/2 + i)
        
        # Размещаем узлы умножения
        for i, node in enumerate(mul_nodes):
            pos[node] = (layer * layer_height, -len(nodes_in_layer)/2 + len(add_nodes) + i)
        
        # Размещаем выходные узлы
        for i, node in enumerate(out_nodes):
            pos[node] = (layer * layer_height, -len(nodes_in_layer)/2 + len(add_nodes) + len(mul_nodes) + i)
    
    # Увеличиваем размер фигуры
    plt.figure(figsize=(30, 15))
    
    # Цвета узлов
    node_colors = []
    for node in G.nodes():
        if node.startswith('in'):
            node_colors.append('green')  # Входы - зеленый
        elif node.startswith('o'):
            node_colors.append('red')    # Выходы - красный
        else:  # Промежуточные узлы
            if graph.nodes.loc[graph.nodes['id'] == node, 'f1'].iloc[0] == 1:
                node_colors.append('lightblue')  # Сложение - голубой
            else:
                node_colors.append('blue')       # Умножение - синий
    
    # Рисование графа
    nx.draw(G, pos, 
            with_labels=True, 
            node_color=node_colors,
            node_size=500,
            font_size=8,
            font_weight='bold',
            arrows=True,
            arrowsize=20)
    
    # Легенда
    plt.title(f'Распутанный граф {graph_idx}\nЗеленый - входы, Голубой - сложения, Синий - умножения, Красный - выходы')
    
    # Сохранение
    plt.savefig(f'outputs/data_analysis/graph_{graph_idx}_unraveled.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    visualize_graph(0)  # Визуализация первого графа
    visualize_unraveled_graph(0)  # Визуализация распутанного графа
    print_detailed_info(0)  # Подробная информация о первом графе 