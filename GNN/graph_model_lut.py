import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from torch.utils.data import Dataset, DataLoader
import pickle
import argparse
import pprint as pp
from sklearn import model_selection
from sklearn.preprocessing import RobustScaler, QuantileTransformer, StandardScaler
import os
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns
from scipy import stats
import gc
import matplotlib.pyplot as plt

class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-6):  # Увеличиваем patience и уменьшаем min_delta
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
            self.counter = 0

def collate_fn(batch):
    """Функция для создания батча"""
    # Распаковываем батч
    xs, edge_indices, ys = zip(*batch)
    
    # Создаем список для хранения смещений узлов
    cumsum = 0
    batch_idx = []
    edge_indices_list = []
    
    # Обрабатываем каждый граф
    for i, (x, edge_index) in enumerate(zip(xs, edge_indices)):
        num_nodes = x.size(0)
        batch_idx.extend([i] * num_nodes)
        
        # Обновляем индексы рёбер
        if edge_index.numel() > 0:
            edge_indices_list.append(edge_index + cumsum)
        
        cumsum += num_nodes
    
    # Объединяем все данные
    x = torch.cat(xs, dim=0)
    edge_index = torch.cat(edge_indices_list, dim=1)
    batch = torch.tensor(batch_idx, dtype=torch.long)
    y = torch.stack(ys)
    
    return x, edge_index, y, batch

class CustomGraph:
    """Простая замена для представления графов"""
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        # Выбираем только признаки f0-f10
        self.node_features = nodes[['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10']].values.astype(np.float32)
        
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
        return torch.FloatTensor(self.node_features)
    
    def get_edge_index(self):
        """Получение списка рёбер"""
        return torch.LongTensor(self.edge_index)
        
    def get_node_mapping(self):
        """Получение отображения меток узлов на индексы"""
        return self.node_mapping

class GraphDataset(Dataset):
    def __init__(self, graphs, targets):
        self.graphs = graphs
        self.targets = torch.tensor(targets, dtype=torch.float32)  # Сразу создаем тензор
        
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        graph = self.graphs[idx]
        
        # Преобразуем данные в тензоры PyTorch
        x = torch.FloatTensor(graph.get_node_features())
        edge_index = torch.LongTensor(graph.get_edge_index())
        y = self.targets[idx]
        
        return x, edge_index, y

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        self.scale = np.sqrt(in_channels)
        
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        attention = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / self.scale, dim=-1)
        out = torch.matmul(attention, v)
        return out + x  # Residual connection

class ResGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)
        self.attention = AttentionModule(out_channels)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        
        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = nn.Identity()
            
        self.activation = nn.SiLU()  # More advanced activation function
        
    def forward(self, x, edge_index):
        identity = self.shortcut(x)
        
        out = self.conv1(x, edge_index)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.conv2(out, edge_index)
        out = self.norm2(out)
        out = self.attention(out)
        out = self.activation(out + identity)  # Residual connection
        return self.dropout(out)

class GCN(nn.Module):
    def __init__(self, num_features, dropout_rate=0.2):
        super().__init__()
        
        # Увеличиваем размерность сети
        self.input_transform = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Глубокие ResNet блоки с постепенным увеличением размерности
        self.block1 = ResGCNBlock(256, 512, dropout_rate)
        self.block2 = ResGCNBlock(512, 768, dropout_rate)
        self.block3 = ResGCNBlock(768, 1024, dropout_rate)
        self.block4 = ResGCNBlock(1024, 1536, dropout_rate)
        
        # Выходные слои с skip-connections
        self.global_attention = AttentionModule(1536)
        
        self.output_layers = nn.Sequential(
            nn.Linear(1536, 768),
            nn.LayerNorm(768),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(768, 384),
            nn.LayerNorm(384),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(384, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(192, 1)
        )
        
        # Инициализация весов
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x, edge_index, batch):
        # Input transformation
        x = self.input_transform(x)
        
        # ResNet blocks with attention
        x = self.block1(x, edge_index)
        x = self.block2(x, edge_index)
        x = self.block3(x, edge_index)
        x = self.block4(x, edge_index)
        
        # Global attention
        x = self.global_attention(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Output layers
        x = self.output_layers(x)
        
        return x

class CombinedLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.delta = delta
        
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        huber_loss = F.smooth_l1_loss(pred, target, beta=self.delta)
        return 0.5 * mse_loss + 0.5 * huber_loss

class DataPreprocessor:
    def __init__(self):
        self.quantile_transformer = QuantileTransformer(output_distribution='normal')
        self.eps = 1e-8
        
    def fit_transform(self, data):
        # Log transform for large values
        data_log = np.sign(data) * np.log1p(np.abs(data) + self.eps)
        
        # Quantile transform for better distribution
        data_transformed = self.quantile_transformer.fit_transform(data_log.reshape(-1, 1)).reshape(-1)
        
        return data_transformed
    
    def transform(self, data):
        data_log = np.sign(data) * np.log1p(np.abs(data) + self.eps)
        return self.quantile_transformer.transform(data_log.reshape(-1, 1)).reshape(-1)
    
    def inverse_transform(self, data):
        data_log = self.quantile_transformer.inverse_transform(data.reshape(-1, 1)).reshape(-1)
        return np.sign(data_log) * (np.exp(np.abs(data_log)) - 1 - self.eps)

class LUTModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, emb_dim=128, output_dim=1):
        super(LUTModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_dim, emb_dim)
        self.fc2 = nn.Linear(emb_dim, output_dim)
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        if batch is not None:
            batch = batch.long()
        x = global_mean_pool(x, batch)
        emb = F.relu(self.fc1(x))  # (batch, 128)
        out = self.fc2(emb)        # (batch, 1)
        return emb, out

def train_model(model, train_loader, val_loader, device, num_epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_model_state = None
    patience = 20
    counter = 0
    
    # Сохраняем историю обучения
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Очистка кэша CUDA перед каждой эпохой
        torch.cuda.empty_cache()
        gc.collect()
        
        for batch_data in train_loader:
            # Распаковываем и переносим данные на устройство
            x, edge_index, y, batch = [b.to(device) for b in batch_data]
            
            optimizer.zero_grad()
            emb, out = model(x, edge_index, batch)
            loss = criterion(out, y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            del emb, out, loss, x, edge_index, y, batch
            torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Валидация
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_data in val_loader:
                # Распаковываем и переносим данные на устройство
                x, edge_index, y, batch = [b.to(device) for b in batch_data]
                
                emb, out = model(x, edge_index, batch)
                val_loss += criterion(out, y).item()
                
                del emb, out, x, edge_index, y, batch
                torch.cuda.empty_cache()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        
        # Сохраняем лучшую модель
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1
        
        # Ранняя остановка
        if counter >= patience:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break
    
    # Загружаем лучшую модель
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

def evaluate_model(model, test_loader, device):
    model.eval()
    total_mae = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            x, edge_index, y, mask = [b.to(device) for b in batch_data]
            
            # Проверяем входные данные
            if torch.isnan(x).any():
                print("Warning: NaN detected in input features")
                continue
                
            emb, out = model(x, edge_index, mask)
            
            # Проверяем выходные данные на NaN
            if torch.isnan(out).any():
                print("Warning: NaN detected in model output")
                print(f"Input features stats: min={x.min()}, max={x.max()}, mean={x.mean()}")
                continue
                
            mae = F.l1_loss(out, y)
            total_mae += mae.item()
            
            predictions.extend(out.cpu().numpy())
            targets.extend(y.cpu().numpy())
    
    return total_mae / len(test_loader), np.array(predictions), np.array(targets)

def calculate_accuracy_metrics(y_true, y_pred):
    """Рассчитывает различные метрики точности"""
    # Точность с разными порогами погрешности
    accuracy_5 = np.mean(np.abs((y_true - y_pred) / y_true) <= 0.05) * 100
    accuracy_10 = np.mean(np.abs((y_true - y_pred) / y_true) <= 0.10) * 100
    accuracy_20 = np.mean(np.abs((y_true - y_pred) / y_true) <= 0.20) * 100
    
    # Точность по диапазонам значений
    ranges = [
        (0, 1000, "0-1K"),
        (1000, 5000, "1K-5K"),
        (5000, 10000, "5K-10K"),
        (10000, float('inf'), ">10K")
    ]
    
    range_accuracies = {}
    for start, end, label in ranges:
        mask = (y_true >= start) & (y_true < end)
        if np.sum(mask) > 0:  # если есть значения в этом диапазоне
            acc_5 = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) <= 0.05) * 100
            acc_10 = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) <= 0.10) * 100
            acc_20 = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) <= 0.20) * 100
            range_accuracies[label] = {
                "count": int(np.sum(mask)),
                "accuracy_5": float(acc_5),
                "accuracy_10": float(acc_10),
                "accuracy_20": float(acc_20)
            }
    
    return {
        "overall_accuracy_5": float(accuracy_5),
        "overall_accuracy_10": float(accuracy_10),
        "overall_accuracy_20": float(accuracy_20),
        "range_accuracies": range_accuracies
    }

def plot_training_history(train_losses, val_losses, save_path='outputs/lut_implementation/plots'):
    """Визуализация истории обучения"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Создаем директорию если её нет
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'training_history.png'))
    plt.close()

def plot_predictions_vs_targets(y_true, y_pred, save_path='outputs/lut_implementation/plots'):
    """Визуализация предсказаний vs реальных значений"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title('Predictions vs True Values')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.grid(True)
    
    # Добавляем R² в график
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes)
    
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'predictions_vs_targets.png'))
    plt.close()

def plot_error_distribution(y_true, y_pred, save_path='outputs/lut_implementation/plots'):
    """Визуализация распределения ошибок"""
    errors = y_pred - y_true
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.grid(True)
    
    # Добавляем статистику
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    plt.text(0.05, 0.95, f'Mean Error: {mean_error:.4f}\nStd Error: {std_error:.4f}', 
             transform=plt.gca().transAxes)
    
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'error_distribution.png'))
    plt.close()

def plot_metrics_by_range(y_true, y_pred, save_path='outputs/lut_implementation/plots'):
    """Визуализация метрик по диапазонам значений"""
    # Определяем диапазоны на основе реальных данных
    min_val = np.min(y_true)
    max_val = np.max(y_true)
    step = (max_val - min_val) / 4  # Разделим на 4 диапазона
    
    ranges = [
        (min_val, min_val + step, f"{int(min_val)}-{int(min_val + step)}"),
        (min_val + step, min_val + 2*step, f"{int(min_val + step)}-{int(min_val + 2*step)}"),
        (min_val + 2*step, min_val + 3*step, f"{int(min_val + 2*step)}-{int(min_val + 3*step)}"),
        (min_val + 3*step, max_val, f"{int(min_val + 3*step)}-{int(max_val)}")
    ]
    
    metrics = []
    for start, end, label in ranges:
        mask = (y_true >= start) & (y_true < end)
        if np.sum(mask) > 0:
            mae = mean_absolute_error(y_true[mask], y_pred[mask])
            mse = mean_squared_error(y_true[mask], y_pred[mask])
            r2 = r2_score(y_true[mask], y_pred[mask]) if len(np.unique(y_true[mask])) > 1 else 0
            metrics.append({
                'range': label,
                'mae': mae,
                'mse': mse,
                'r2': r2,
                'count': int(np.sum(mask))
            })
    
    # Создаем график
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Metrics by Value Range')
    
    # MAE
    axes[0,0].bar([m['range'] for m in metrics], [m['mae'] for m in metrics])
    axes[0,0].set_title('MAE by Range')
    axes[0,0].set_ylabel('MAE')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # MSE
    axes[0,1].bar([m['range'] for m in metrics], [m['mse'] for m in metrics])
    axes[0,1].set_title('MSE by Range')
    axes[0,1].set_ylabel('MSE')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # R²
    axes[1,0].bar([m['range'] for m in metrics], [m['r2'] for m in metrics])
    axes[1,0].set_title('R² by Range')
    axes[1,0].set_ylabel('R²')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Count
    axes[1,1].bar([m['range'] for m in metrics], [m['count'] for m in metrics])
    axes[1,1].set_title('Sample Count by Range')
    axes[1,1].set_ylabel('Count')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'metrics_by_range.png'))
    plt.close()

def plot_error_matrix(y_true, y_pred, save_path='outputs/lut_implementation/plots'):
    """Создание матрицы ошибок и анализа относительных ошибок"""
    # Рассчитываем ошибки
    abs_errors = np.abs(y_pred - y_true)
    rel_errors = (y_pred - y_true) / y_true * 100
    
    # Обработка выбросов для визуализации
    rel_errors = np.clip(rel_errors, -100, 100)  # Ограничиваем относительные ошибки до ±100%
    
    # Создаем DataFrame для удобного анализа
    df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred,
        'Abs_Error': abs_errors,
        'Rel_Error_%': rel_errors
    })
    
    # Сортируем по абсолютной ошибке
    df = df.sort_values('Abs_Error', ascending=False)
    
    # Создаем график
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Error Analysis Matrix')
    
    # 1. Scatter plot предсказаний
    axes[0,0].scatter(y_true, y_pred, alpha=0.5)
    axes[0,0].plot([y_true.min(), y_true.max()], 
                   [y_true.min(), y_true.max()], 'r--')
    axes[0,0].set_title('Predictions vs Actual')
    axes[0,0].set_xlabel('Actual Values (LUT)')
    axes[0,0].set_ylabel('Predicted Values (LUT)')
    axes[0,0].grid(True)
    
    # 2. Гистограмма относительных ошибок
    axes[0,1].hist(rel_errors, bins=50, edgecolor='black')
    axes[0,1].set_title('Distribution of Relative Errors')
    axes[0,1].set_xlabel('Relative Error (%)')
    axes[0,1].set_ylabel('Count')
    axes[0,1].grid(True)
    
    # 3. Box plot относительных ошибок
    axes[1,0].boxplot(rel_errors)
    axes[1,0].set_title('Relative Error Box Plot')
    axes[1,0].set_ylabel('Relative Error (%)')
    axes[1,0].grid(True)
    
    # 4. Статистика ошибок
    stats_text = f"""Error Statistics:
    Mean Abs Error: {np.mean(abs_errors):.2f} LUT
    Median Abs Error: {np.median(abs_errors):.2f} LUT
    Mean Rel Error: {np.mean(rel_errors):.2f}%
    Median Rel Error: {np.median(rel_errors):.2f}%
    Max Rel Error: {np.max(np.abs(rel_errors)):.2f}%
    Min Rel Error: {np.min(rel_errors):.2f}%
    
    90th percentile Abs Error: {np.percentile(abs_errors, 90):.2f} LUT
    95th percentile Abs Error: {np.percentile(abs_errors, 95):.2f} LUT
    """
    axes[1,1].text(0.1, 0.1, stats_text, fontsize=10)
    axes[1,1].axis('off')
    
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'error_matrix.png'))
    plt.close()
    
    # Сохраняем детальную статистику в CSV
    error_stats = pd.DataFrame({
        'Metric': [
            'Mean Absolute Error (LUT)', 
            'Median Absolute Error (LUT)',
            '90th percentile Absolute Error (LUT)',
            '95th percentile Absolute Error (LUT)',
            'Mean Relative Error (%)',
            'Median Relative Error (%)',
            '90th percentile Relative Error (%)',
            '95th percentile Relative Error (%)',
            'Max Relative Error (%)',
            'Min Relative Error (%)'
        ],
        'Value': [
            np.mean(abs_errors),
            np.median(abs_errors),
            np.percentile(abs_errors, 90),
            np.percentile(abs_errors, 95),
            np.mean(rel_errors),
            np.median(rel_errors),
            np.percentile(rel_errors, 90),
            np.percentile(rel_errors, 95),
            np.max(np.abs(rel_errors)),
            np.min(rel_errors)
        ]
    })
    error_stats.to_csv(os.path.join(save_path, 'error_statistics.csv'), index=False)
    
    # Сохраняем детальный анализ по каждому предсказанию
    df.to_csv(os.path.join(save_path, 'detailed_errors.csv'), index=False)

def plot_confusion_matrix(y_true, y_pred, save_path='outputs/lut_implementation/plots'):
    """Создание confusion matrix для диапазонов LUT"""
    # Определяем диапазоны
    ranges = [
        (0, 500, "0-500"),
        (501, 1000, "501-1000"),
        (1001, 2000, "1001-2000"),
        (2001, 3000, "2001-3000"),
        (3001, 4000, "3001-4000"),
        (4001, 5324, "4001-5324")
    ]
    
    # Создаем метки для истинных и предсказанных значений
    def get_range_label(value):
        for start, end, label in ranges:
            if start <= value <= end:
                return label
        return ranges[-1][2]  # Возвращаем последний диапазон для значений выше максимума
    
    y_true_labels = [get_range_label(val) for val in y_true]
    y_pred_labels = [get_range_label(val) for val in y_pred]
    
    # Создаем confusion matrix
    unique_labels = [r[2] for r in ranges]
    matrix = np.zeros((len(ranges), len(ranges)))
    
    for true_label, pred_label in zip(y_true_labels, y_pred_labels):
        i = unique_labels.index(true_label)
        j = unique_labels.index(pred_label)
        matrix[i, j] += 1
    
    # Создаем график
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues',
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('Confusion Matrix (LUT Ranges)')
    plt.xlabel('Predicted Range')
    plt.ylabel('True Range')
    
    # Поворачиваем метки для лучшей читаемости
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # Сохраняем график
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()

def generate_all_plots(train_losses, val_losses, y_true, y_pred):
    """Генерация всех графиков в отдельные файлы"""
    # История обучения
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('outputs/lut_implementation/plots/training_history.png')
    plt.close()
    
    # Предсказания vs реальные значения
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title('Predictions vs True Values')
    plt.xlabel('True Values (LUT)')
    plt.ylabel('Predictions (LUT)')
    plt.grid(True)
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes)
    plt.savefig('outputs/lut_implementation/plots/predictions_vs_targets.png')
    plt.close()
    
    # Распределение ошибок
    errors = y_pred - y_true
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error (LUT)')
    plt.ylabel('Count')
    plt.grid(True)
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    plt.text(0.05, 0.95, f'Mean Error: {mean_error:.4f}\nStd Error: {std_error:.4f}', 
             transform=plt.gca().transAxes)
    plt.savefig('outputs/lut_implementation/plots/error_distribution.png')
    plt.close()
    
    # Метрики по диапазонам
    plot_metrics_by_range(y_true, y_pred)
    
    # Матрица ошибок
    plot_confusion_matrix(y_true, y_pred)
    
    # Матрица анализа ошибок
    plot_error_matrix(y_true, y_pred)

def main(args):
    # Загружаем данные
    with open('outputs/processed_data/graph_dataset.pkl', 'rb') as fp:
        graphs = pickle.load(fp)
    
    graph_labels_lut = pd.read_csv('outputs/processed_data/graph_target_lut.csv')
    
    # Проверяем данные на NaN
    print("Checking input data...")
    print(f"Number of graphs: {len(graphs)}")
    print(f"Number of labels: {len(graph_labels_lut)}")
    print(f"Labels contain NaN: {graph_labels_lut.isna().any().any()}")
    
    # Нормализация целевой переменной
    scaler = StandardScaler()
    graph_labels_lut_scaled = scaler.fit_transform(graph_labels_lut.values)
    
    # Проверяем нормализованные данные
    print("Checking normalized data...")
    print(f"Contains NaN: {np.isnan(graph_labels_lut_scaled).any()}")
    print(f"Min value: {np.min(graph_labels_lut_scaled)}")
    print(f"Max value: {np.max(graph_labels_lut_scaled)}")
    
    # Определяем устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Получаем размерность признаков узлов
    num_features = graphs[0].get_node_features().shape[1]
    
    # Разделяем данные на train/val/test
    indices = np.arange(len(graphs))
    train_idx, temp_idx = model_selection.train_test_split(indices, train_size=0.7, random_state=42)
    val_idx, test_idx = model_selection.train_test_split(temp_idx, train_size=0.5, random_state=42)
    
    # Создаем наборы данных
    train_dataset = GraphDataset(
        [graphs[i] for i in train_idx],
        graph_labels_lut_scaled[train_idx]
    )
    val_dataset = GraphDataset(
        [graphs[i] for i in val_idx],
        graph_labels_lut_scaled[val_idx]
    )
    test_dataset = GraphDataset(
        [graphs[i] for i in test_idx],
        graph_labels_lut_scaled[test_idx]
    )
    
    # Создаем загрузчики данных
    train_loader = DataLoader(
        train_dataset, 
        batch_size=int(args['batch_size']),
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=int(args['batch_size']),
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=int(args['batch_size']),
        collate_fn=collate_fn
    )
    
    # Создаем модель
    model = LUTModel(num_features, hidden_dim=128, emb_dim=128, output_dim=1).to(device)
    
    # Обучаем модель
    model, train_losses, val_losses = train_model(model, train_loader, val_loader, device, int(args['epoch']))
    
    # Оцениваем на тестовых данных
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            x, edge_index, y, batch = [b.to(device) for b in batch_data]
            emb, out = model(x, edge_index, batch)
            
            # Сохраняем предсказания и целевые значения
            predictions.extend(out.cpu().numpy())
            targets.extend(y.cpu().numpy())
    
    # Преобразуем обратно
    predictions = np.array(predictions).reshape(-1, 1)
    targets = np.array(targets).reshape(-1, 1)
    
    # Обратное преобразование
    predictions_orig = scaler.inverse_transform(predictions).flatten()
    targets_orig = scaler.inverse_transform(targets).flatten()
    
    # Вычисляем метрики на оригинальных значениях
    mae = mean_absolute_error(targets_orig, predictions_orig)
    mse = mean_squared_error(targets_orig, predictions_orig)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets_orig, predictions_orig)
    
    # Сохраняем метрики
    metrics = {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2)
    }
    
    # Создаем директорию для сохранения, если её нет
    os.makedirs('outputs/lut_implementation/metrics', exist_ok=True)
    
    # Сохраняем метрики
    with open('outputs/lut_implementation/metrics/test_metrics_lut.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Сохраняем модель и данные
    os.makedirs('outputs/lut_implementation/models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'metrics': metrics,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'predictions': predictions_orig,  # Сохраняем оригинальные значения
        'targets': targets_orig  # Сохраняем оригинальные значения
    }, 'outputs/lut_implementation/models/lut_model.pt')

    # Сохраняем полную модель для Netron
    torch.save(model, 'outputs/lut_implementation/models/lut_model_full.pt')

    print("\nTest Metrics:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    
    # Генерируем графики
    generate_all_plots(train_losses, val_losses, targets_orig, predictions_orig)
    
    # --- СОХРАНЕНИЕ ЭМБЕДДИНГОВ ---
    def get_lut_embeddings(model, loader, device):
        model.eval()
        embeddings = []
        with torch.no_grad():
            for batch_data in loader:
                x, edge_index, y, batch = [b.to(device) for b in batch_data]
                emb, out = model(x, edge_index, batch)
                embeddings.append(emb.cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    os.makedirs('outputs/lut_implementation/embeddings', exist_ok=True)
    test_embeddings = get_lut_embeddings(model, test_loader, device)
    np.save('outputs/lut_implementation/embeddings/test_embeddings.npy', test_embeddings)

    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='provide arguments for the graph embedding model with LUT predictions'
    )
    
    parser.add_argument('--epoch', help='number of epochs', default=150)
    parser.add_argument('--batch-size', help='size of batch', default=64)
    parser.add_argument('--dropout-rate', help='dropout rate', default=0.4)
    parser.add_argument('--lr', help='learning rate', default=0.0005)
    parser.add_argument('--weight-decay', help='weight decay for L2 regularization', default=0.02)
    parser.add_argument('--random-seed', help='random seed', default=42)
    
    args = vars(parser.parse_args())
    pp.pprint(args)
    main(args)
