import numpy as np
import torch
import pickle
import pandas as pd
import sys
from GNN.graph_model_lut import LUTModel, CustomGraph
from GNN.graph_model_dsp import DSPModel
from GNN.graph_cp_implementation import CPModel
import torch.nn.functional as F

# Добавляем CustomGraph в глобальное пространство имен
sys.modules['__main__'].CustomGraph = CustomGraph

class RLEnv:
    def __init__(self, alpha=0.5, lambda0=0.5, graphs=None, target_dsp=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.alpha = alpha
        self.lambda0 = lambda0
        self.dsp_margin = 2.6  # Допустимое отклонение от целевого DSP

        # Загружаем модели и скейлеры
        lut_ckpt = torch.load('outputs/lut_implementation/models/lut_model.pt', map_location=self.device)
        dsp_ckpt = torch.load('outputs/dsp_implementation/models/dsp_model.pt', map_location=self.device)
        cp_ckpt  = torch.load('outputs/cp_implementation/models/cp_model.pt', map_location=self.device)

        self.lut_model = LUTModel(input_dim=11, hidden_dim=128, emb_dim=128, output_dim=1).to(self.device)
        self.dsp_model = DSPModel(input_dim=11, hidden_dim=128, emb_dim=128, output_dim=1).to(self.device)
        self.cp_model  = CPModel(input_dim=11, hidden_dim=128, emb_dim=128, output_dim=1).to(self.device)

        self.lut_model.load_state_dict(lut_ckpt['model_state_dict'])
        self.dsp_model.load_state_dict(dsp_ckpt['model_state_dict'])
        self.cp_model.load_state_dict(cp_ckpt['model_state_dict'])

        self.lut_model.eval()
        self.dsp_model.eval()
        self.cp_model.eval()

        # Загружаем скейлеры
        self.lut_scaler = lut_ckpt['scaler']
        self.dsp_scaler = dsp_ckpt['scaler']
        self.cp_scaler  = cp_ckpt['scaler']

        # Для нормировки (если нужно)
        self.lut_mean = getattr(self.lut_scaler, 'mean_', 0)
        self.lut_std  = getattr(self.lut_scaler, 'scale_', 1)
        self.cp_mean  = getattr(self.cp_scaler, 'mean_', 0)
        self.cp_std   = getattr(self.cp_scaler, 'scale_', 1)
        self.dsp_std  = getattr(self.dsp_scaler, 'scale_', 1)

        # Загружаем данные
        if graphs is not None and target_dsp is not None:
            self.graphs = graphs
            self.target_dsp = target_dsp
        else:
            with open('outputs/processed_data/graph_dataset.pkl', 'rb') as fp:
                self.graphs = pickle.load(fp)
            self.target_dsp = pd.read_csv('outputs/processed_data/graph_target_dsp.csv')
        self.target_lut = pd.read_csv('outputs/processed_data/graph_target_lut.csv')
        self.target_cp = pd.read_csv('outputs/processed_data/graph_target_cp.csv')
        self.current_graph = None
        self.current_nodes = None
        self.current_edges = None
        self.target_dsp_value = None
        self.step_count = 0
        self.multiplication_nodes = []
        self.dsp_assignments = {}
        self.current_mult_index = 0
        
    def _find_multiplication_nodes(self):
        """Находит все узлы с операциями умножения в текущем графе"""
        mult_nodes = []
        for idx, node in self.current_nodes.iterrows():
            # Проверяем признак операции умножения (f2) и тип реализации (f10)
            if node['f2'] > 0:  # f2 > 0 означает операцию умножения
                mult_nodes.append(idx)
        return mult_nodes
        
    def reset(self, graph_index, target_dsp):
        """Сброс окружения для нового графа"""
        self.current_graph = self.graphs[graph_index]
        self.current_nodes = self.current_graph.nodes.copy()
        self.current_edges = self.current_graph.edges.copy()
        self.target_dsp_value = target_dsp
        self.step_count = 0

        # Устанавливаем f10 = 10 для всех операций умножения
        mult_mask = self.current_nodes['f2'] > 0
        self.current_nodes.loc[mult_mask, 'f10'] = 10
        
        # Находим все операции умножения
        self.multiplication_nodes = self._find_multiplication_nodes()
        self.max_steps = len(self.multiplication_nodes)  # Максимум шагов = количеству операций
        self.current_mult_index = 0
        
        # Инициализируем словарь назначений
        self.dsp_assignments = {node_idx: False for node_idx in self.multiplication_nodes}
        
        # Добавляем атрибуты для отслеживания статистики
        self.mult_ops = len(self.multiplication_nodes)
        self.dsp_assigned = 0
        
        # Получаем начальное состояние
        state = self._get_state()
        return state
        
    def step(self, action):
        """Выполнение действия и получение награды"""
        self.step_count += 1
        
        # Получаем текущий узел умножения
        if self.current_mult_index < len(self.multiplication_nodes):
            current_node = self.multiplication_nodes[self.current_mult_index]
            
            # Применяем действие к конкретному узлу
            if action == 1:
                self.dsp_assignments[current_node] = True
                # Обновляем признак типа реализации
                self.current_nodes.at[current_node, 'f10'] = 0  # f10 = 0 для DSP
                self.dsp_assigned += 1
            else:
                self.current_nodes.at[current_node, 'f10'] = 10  # f10 = 10 для LUT
            
            self.current_mult_index += 1
        
        # Получаем новое состояние
        state = self._get_state()
        
        # Проверяем завершение эпизода
        done = self.current_mult_index >= len(self.multiplication_nodes)
        
        # Вычисляем награду и текущие значения
        reward, lut, dsp, cp = self._calculate_reward(done)
        
        return state, reward, done, lut, dsp, cp
        
    def _get_state(self):
        node_features = torch.FloatTensor(self.current_nodes[['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10']].values).to(self.device)
        edge_index = torch.LongTensor(self.current_graph.get_edge_index()).to(self.device)
        batch = torch.zeros(len(self.current_nodes), dtype=torch.long, device=self.device)
        with torch.no_grad():
            lut_emb, _ = self.lut_model(node_features, edge_index, batch)
            dsp_emb, _ = self.dsp_model(node_features, edge_index, batch)
            cp_emb, _ = self.cp_model(node_features, edge_index, batch)
            lut_emb = lut_emb.cpu().numpy().flatten()
            dsp_emb = dsp_emb.cpu().numpy().flatten()
            cp_emb = cp_emb.cpu().numpy().flatten()
        normalized_dsp = self.target_dsp_value / 100.0
        progress = self.current_mult_index / len(self.multiplication_nodes) if len(self.multiplication_nodes) > 0 else 0.0
        # Метаданные графа
        num_nodes = float(len(self.current_nodes))
        num_mult = float((self.current_nodes['f2'] > 0).sum())
        state = np.concatenate([
            lut_emb,
            dsp_emb,
            cp_emb,
            [num_nodes],
            [num_mult],
            [normalized_dsp],
            [progress]
        ]).astype(np.float32)
        return state

    def _calculate_reward(self, done=False):
        """Вычисление награды и текущих значений"""
        with torch.no_grad():
            node_features = torch.FloatTensor(self.current_nodes[['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10']].values).to(self.device)
            edge_index = torch.LongTensor(self.current_graph.get_edge_index()).to(self.device)
            batch = torch.zeros(len(self.current_nodes), dtype=torch.long, device=self.device)
            
            _, current_lut = self.lut_model(node_features, edge_index, batch)
            _, current_dsp = self.dsp_model(node_features, edge_index, batch)
            _, current_cp = self.cp_model(node_features, edge_index, batch)
            current_lut = current_lut.item()
            current_dsp = current_dsp.item()
            current_cp = current_cp.item()
        
        # Преобразуем значения обратно
        current_lut = self.lut_scaler.inverse_transform(np.array([[current_lut]]))[0][0]
        current_dsp = self.dsp_scaler.inverse_transform(np.array([[current_dsp]]))[0][0]
        current_cp = self.cp_scaler.inverse_transform(np.array([[current_cp]]))[0][0]
        
        alpha = self.alpha
        lambda0 = self.lambda0
        reward = 0.0
        if done:
            # Если мы попали в целевой DSP, даем большой бонус и штрафуем только за LUT и CP
            if abs(current_dsp - self.target_dsp_value) <= self.dsp_margin:
                #bonus = 5.0  # Большой бонус за попадание в цель
                print(f"Попадание в погрешность 2.6")
                reward = -alpha * current_lut - lambda0 * current_cp
            else:
                # Асимметричный штраф за DSP
                if current_dsp < self.target_dsp_value:
                    dsp_penalty = (self.target_dsp_value - current_dsp)/1.5
                else:
                    dsp_penalty = (current_dsp - self.target_dsp_value)/1.5
                print(f"  [DEBUG] Награды: LUT={-alpha * current_lut}, DSP={-dsp_penalty}, CP={-lambda0 * current_cp}")
                reward = -alpha * current_lut - dsp_penalty - lambda0 * current_cp
            # Дополнительный бонус, если DSP в окрестности 3 от целевого
            if abs(current_dsp - self.target_dsp_value) <= 1:
                reward += 20.0
            elif abs(current_dsp - self.target_dsp_value) <= 3:
                reward += 12.0
            elif abs(current_dsp - self.target_dsp_value) <= 5:
                reward += 7.0
        
            print(f"  [DEBUG] Суммарно: {reward}")
        else:
            reward = 0.0
                
        return reward/10, current_lut, current_dsp, current_cp
