import torch
import torch.nn as nn
import torch.nn.functional as F
from actor_critic import ActorCritic
from rl_env import RLEnv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
import seaborn as sns
import time
from datetime import datetime
from collections import defaultdict
import random
import numpy as np

# class EarlyStoppingCallback:
#     def __init__(self, patience=1000, min_delta=0.0005, min_episodes=500): 
#         self.patience = patience
#         self.min_delta = min_delta
#         self.min_episodes = min_episodes
#         self.best_reward = float('-inf')
#         self.wait = 0
#         self.best_weights = None
#         self.stop_training = False
#         self.training_time = 0
#         self.start_time = None
#         
#     def on_training_begin(self):
#         self.start_time = time.time()
#         
#     def on_episode_end(self, model, episode, metrics):
#         if episode < self.min_episodes:
#             return False
#             
#         current_reward = metrics.avg_rewards[-1]
#         
#         if current_reward > self.best_reward + self.min_delta:
#             self.best_reward = current_reward
#             self.wait = 0
#             # Сохраняем лучшие веса
#             self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
#         else:
#             self.wait += 1
#             if self.wait >= self.patience:
#                 self.stop_training = True
#                 self.training_time = time.time() - self.start_time
#                 print(f"\nРанняя остановка на эпизоде {episode}")
#                 print(f"Лучшая средняя награда: {self.best_reward:.3f}")
#                 print(f"Время обучения: {self.training_time:.2f} секунд")
#                 return True
#         return False

# class EarlyStoppingOnGrowth:
#     def __init__(self, growth_window=150, growth_threshold=10, min_episodes=450):
#         """
#         growth_window: размер окна (количество эпизодов) для анализа среднего прироста награды
#         growth_threshold: минимальный прирост средней награды между окнами, чтобы считать рост резким
#         min_episodes: минимальное количество эпизодов до начала проверки ранней остановки
#         """
#         self.growth_window = growth_window
#         self.growth_threshold = growth_threshold
#         self.min_episodes = min_episodes
#         self.stop_training = False
#         self.training_time = 0
#         self.start_time = None
#         self.best_weights = None
# 
#     def on_training_begin(self):
#         self.start_time = time.time()
# 
#     def on_episode_end(self, model, episode, metrics):
#         if episode < self.min_episodes + 2 * self.growth_window:
#             return False
# 
#         avg_rewards = metrics.avg_rewards
#         prev_window = np.mean(avg_rewards[-2*self.growth_window:-self.growth_window])
#         curr_window = np.mean(avg_rewards[-self.growth_window:])
# 
#         if (curr_window - prev_window) >= self.growth_threshold:
#             self.stop_training = True
#             self.training_time = time.time() - self.start_time
#             # Сохраняем лучшие веса
#             self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
#             print(f"\nРанняя остановка: резкий рост награды на {episode} эпизоде!")
#             print(f"Средняя награда выросла с {prev_window:.3f} до {curr_window:.3f}")
#             print(f"Время обучения: {self.training_time:.2f} секунд")
#             return True
#         return False

class TrainingMetrics:
    def __init__(self, window_size=100):
        self.episode_rewards = []
        self.avg_rewards = []
        self.lut_values = []
        self.dsp_values = []
        self.cp_values = []
        self.dsp_accuracy = []  # Точность попадания в целевой DSP
        self.zero_dsp_ratio = []  # Отношение случаев с DSP=0
        self.window_size = window_size
        
        # Новые метрики для отслеживания операций умножения
        self.mult_ops_count = []  # Количество операций умножения
        self.dsp_assigned_count = []  # Количество назначенных DSP
        self.dsp_efficiency = []  # Эффективность использования DSP
        
    def add_episode(self, reward, lut, dsp, cp, target_dsp, mult_ops, dsp_assigned):
        self.episode_rewards.append(reward)
        self.lut_values.append(lut)
        self.dsp_values.append(dsp)
        self.cp_values.append(cp)
        
        # Считаем точность DSP (в пределах ±3)
        dsp_accuracy = 1.0 if abs(dsp - target_dsp) <= 4 else 0.0
        self.dsp_accuracy.append(dsp_accuracy)
        
        # Считаем случаи с DSP=0
        self.zero_dsp_ratio.append(1.0 if dsp == 0 else 0.0)
        
        # Новые метрики
        self.mult_ops_count.append(mult_ops)
        self.dsp_assigned_count.append(dsp_assigned)
        self.dsp_efficiency.append(dsp_assigned / mult_ops if mult_ops > 0 else 0)
        
        # Считаем среднюю награду
        if len(self.episode_rewards) >= self.window_size:
            avg_reward = np.mean(self.episode_rewards[-self.window_size:])
        else:
            avg_reward = np.mean(self.episode_rewards)
        self.avg_rewards.append(avg_reward)
    
    def plot_metrics(self, save_path='training_metrics.png'):
        fig, axes = plt.subplots(4, 2, figsize=(15, 16))
        fig.suptitle('Метрики обучения')
        
        # График наград
        axes[0,0].plot(self.episode_rewards, alpha=0.3, label='Награда за эпизод')
        axes[0,0].plot(self.avg_rewards, label='Средняя награда')
        axes[0,0].set_title('Награды')
        axes[0,0].legend()
        
        # График LUT, DSP, CP
        axes[0,1].plot(self.lut_values, label='LUT', alpha=0.5)
        axes[0,1].set_title('LUT значения')
        axes[1,0].plot(self.dsp_values, label='DSP', alpha=0.5)
        axes[1,0].set_title('DSP значения')
        axes[1,1].plot(self.cp_values, label='CP', alpha=0.5)
        axes[1,1].set_title('CP значения')
        
        # График точности DSP и отношения нулевых DSP
        window = 100
        dsp_acc_moving_avg = pd.Series(self.dsp_accuracy).rolling(window).mean()
        zero_dsp_moving_avg = pd.Series(self.zero_dsp_ratio).rolling(window).mean()
        
        axes[2,0].plot(dsp_acc_moving_avg, label='Точность DSP')
        axes[2,0].set_title('Точность попадания в целевой DSP')
        axes[2,1].plot(zero_dsp_moving_avg, label='DSP=0 ratio')
        axes[2,1].set_title('Отношение случаев с DSP=0')
        
        # Новые графики для операций умножения
        axes[3,0].plot(self.mult_ops_count, label='Операции умножения')
        axes[3,0].plot(self.dsp_assigned_count, label='Назначенные DSP')
        axes[3,0].set_title('Количество операций и DSP')
        axes[3,0].legend()
        
        axes[3,1].plot(self.dsp_efficiency, label='Эффективность DSP')
        axes[3,1].set_title('Эффективность использования DSP')
        axes[3,1].set_ylabel('DSP/MUL ratio')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    def save_to_csv(self, save_path='training_metrics.csv'):
        df = pd.DataFrame({
            'episode': range(len(self.episode_rewards)),
            'reward': self.episode_rewards,
            'avg_reward': self.avg_rewards,
            'lut': self.lut_values,
            'dsp': self.dsp_values,
            'cp': self.cp_values,
            'dsp_accuracy': self.dsp_accuracy,
            'zero_dsp_ratio': self.zero_dsp_ratio,
            'mult_ops_count': self.mult_ops_count,
            'dsp_assigned_count': self.dsp_assigned_count,
            'dsp_efficiency': self.dsp_efficiency
        })
        df.to_csv(save_path, index=False)

def select_graph(env, episode, metrics):
    """Всегда выбираем случайный граф"""
    total_graphs = len(env.graphs)
    graph_index = np.random.randint(0, total_graphs)
    target_dsp = env.target_dsp.iloc[graph_index].values[0]
    return graph_index, target_dsp

def balance_graphs_by_mult_count(graphs, target_dsp, max_per_group=100):
    """
    Балансировка графов: не более max_per_group графов на каждое уникальное количество операций умножения.
    Если в группе меньше max_per_group — берём все.
    """
    # Считаем количество операций умножения для каждого графа
    mult_counts = np.array([(g.nodes['f2'] > 0).sum() for g in graphs])
    groups = defaultdict(list)
    for i, count in enumerate(mult_counts):
        groups[count].append(i)

    balanced_indices = []
    for count, idxs in groups.items():
        if len(idxs) > max_per_group:
            chosen = np.random.choice(idxs, max_per_group, replace=False)
            balanced_indices.extend(chosen)
        else:
            balanced_indices.extend(idxs)

    np.random.shuffle(balanced_indices)
    balanced_graphs = [graphs[i] for i in balanced_indices]
    balanced_target_dsp = target_dsp.iloc[balanced_indices].reset_index(drop=True)
    print(f"Балансировка: {len(groups)} групп, максимум {max_per_group} графов на группу, всего {len(balanced_graphs)} графов.")
    print(f"n графов после балансировки: {len(balanced_graphs)}")
    return balanced_graphs, balanced_target_dsp

def balance_graphs_by_target_dsp_bins(graphs, target_dsp, bins, target_per_bin=278):
    # bins — список границ диапазонов, например: [0, 3, 7, 12, 20, 40, 100]
    target_dsps = np.array([target_dsp.iloc[i].values[0] for i in range(len(graphs))])
    bins_ids = np.digitize(target_dsps, bins) - 1

    from collections import defaultdict
    bin_to_indices = defaultdict(list)
    for idx, bin_id in enumerate(bins_ids):
        bin_to_indices[bin_id].append(idx)

    balanced_indices = []
    for bin_id in range(len(bins)-1):
        idxs = bin_to_indices[bin_id]
        if len(idxs) >= target_per_bin:
            # undersampling
            balanced_indices.extend(random.sample(idxs, target_per_bin))
        else:
            # oversampling (с дублированием)
            if len(idxs) > 0:
                balanced_indices.extend(np.random.choice(idxs, target_per_bin, replace=True))
    random.shuffle(balanced_indices)
    balanced_graphs = [graphs[i] for i in balanced_indices]
    balanced_target_dsp = target_dsp.iloc[balanced_indices].reset_index(drop=True)

    # Выводим распределение после балансировки
    print("Распределение target DSP по диапазонам после балансировки:")
    for bin_id in range(len(bins)-1):
        print(f"[{bins[bin_id]}, {bins[bin_id+1]}): {target_per_bin}")
    print(f"Всего графов после балансировки: {len(balanced_graphs)}")
    return balanced_graphs, balanced_target_dsp

def train_rl(num_episodes=10000, 
             learning_rate=0.001,
             gamma=0.99,
             alpha=0.3,
             lambda0=0.3,
             save_interval=10,
             device=None,
             patience=50,
             min_delta=0.001,
             min_episodes=200):
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    # Создаем окружение и модель
    env = RLEnv(alpha=alpha, lambda0=lambda0)
    input_dim = 388  # 128*3 + 2 (num_nodes, num_mult) + 1 + 1
    model = ActorCritic(input_dim).to(device)
    
    # --- Используем сбалансированные графы и target_dsp по диапазонам DSP ---
    bins = [0, 3, 7, 12, 20, 40, 100]
    balanced_graphs, balanced_target_dsp = balance_graphs_by_target_dsp_bins(env.graphs, env.target_dsp, bins, target_per_bin=278)
    # Передаём сбалансированные графы в среду
    env = RLEnv(graphs=balanced_graphs, target_dsp=balanced_target_dsp, alpha=alpha, lambda0=lambda0)
    pairs = []
    for idx in range(len(balanced_graphs)):
        target = balanced_target_dsp.iloc[idx].values[0]
        pairs.append((idx, target))
    all_pairs = pairs * 8
    total_pairs = len(all_pairs)
    print(f'Всего пар (graph, target_dsp) с дублированием: {total_pairs}')
    random.shuffle(all_pairs)
    
    # Оптимизатор и функция потерь
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    huber_loss = nn.HuberLoss()
    
    # Метрики для отслеживания прогресса
    metrics = TrainingMetrics()
    # early_stopping = EarlyStoppingOnGrowth(growth_window=100, growth_threshold=2.0, min_episodes=500)
    
    print("\nНачинаем обучение...")
    print("Эпизод |   LUT   |   DSP   |   CP    | Награда | Ср.Награда | DSP точность | Время")
    print("-" * 90)
    
    # early_stopping.on_training_begin()
    training_start = time.time()
    
    best_avg_reward = float('-inf')
    best_episode = -1
    best_combo_score = float('-inf')
    
    for episode in range(num_episodes):
        episode_start = time.time()

        # --- Берём следующую пару из перемешанного списка ---
        graph_index, target_dsp = all_pairs[episode % total_pairs]
        
        # Сбрасываем окружение
        state = env.reset(graph_index, target_dsp)
        episode_reward = 0
        
        # Сохраняем историю для обучения
        states = []
        actions = []
        rewards = []
        values = []
        action_probs = []
        
        # Для отладки: счётчик шагов внутри эпизода
        step_idx = 0
        graph_id = getattr(env.current_graph, 'name', graph_index)  # если есть имя, иначе индекс
        mult_ops_total = env.mult_ops
        while True:
            # Преобразуем состояние в тензор
            state_tensor = torch.FloatTensor(state).to(device)
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            # Получаем действие от модели
            with torch.no_grad():
                action_prob, value = model(state_tensor)
                action = torch.multinomial(action_prob, 1).item()
            
            # Сохраняем для обучения
            states.append(state_tensor)
            actions.append(action)
            values.append(value)
            action_probs.append(action_prob[0, action].item())
            
            # Выполняем действие
            next_state, reward, done, current_lut, current_dsp, current_cp = env.step(action)
            #print(f"[STEP {step_idx}] reward={reward}, done={done}")
            step_idx += 1
            rewards.append(reward)
            
            # Обновляем состояние и награду
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Обновляем метрики
        metrics.add_episode(episode_reward, current_lut, current_dsp, current_cp, target_dsp, env.mult_ops, env.dsp_assigned)
        mean_reward = metrics.avg_rewards[-1]
        dsp_accuracy = metrics.dsp_accuracy[-1]
        
        # Обучаем на собранном опыте
        optimizer.zero_grad()
        
        # Вычисляем возвраты
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        
        # Нормализуем возвраты
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        # Пересчитываем действия и значения с градиентами
        policy_losses = []
        value_losses = []
        for state, action, R in zip(states, actions, returns):
            action_prob, value = model(state)
            log_prob = torch.log(action_prob[0, action])
            advantage = R - value.item()
            
            policy_losses.append(-log_prob * advantage)
            value_losses.append(huber_loss(value, R.reshape(1, 1)))
        
        # Суммарная потеря
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        
        # Обновляем веса
        loss.backward()
        optimizer.step()
        
        # Вычисляем время эпизода
        episode_time = time.time() - episode_start
        total_time = time.time() - training_start
        
        # Приводим episode_reward к float, если это массив или тензор
        if isinstance(episode_reward, (np.ndarray, torch.Tensor)):
            episode_reward = float(episode_reward.squeeze())
        
        # Выводим результаты
        print(f"Эпизод {episode+1:4d}: LUT={float(current_lut):8.2f}, DSP={float(current_dsp):7.2f}, CP={float(current_cp):7.2f}, R={float(episode_reward):7.2f}, СрR={float(mean_reward):7.2f}, DSP_acc={float(dsp_accuracy):.2f}, Время={float(episode_time):.2f}с")
        #print(f"  [DEBUG] Финальные значения: LUT={current_lut}, DSP={current_dsp}, CP={current_cp}")

        # Подробный разбор награды 
        '''
        dsp_error = abs(current_dsp - target_dsp)
        if current_dsp > target_dsp - 3:
            dsp_penalty = -alpha * current_lut - 10 * abs(target_dsp - current_dsp - 3) - lambda0 * current_cp
        else:
            dsp_penalty = -alpha * current_lut - 5 * abs(target_dsp - current_dsp - 3) - lambda0 * current_cp
            if abs(target_dsp - current_dsp) < 1e-3:
                dsp_penalty += 15
        final_reward = dsp_penalty / 10
        print(f"  [DEBUG] LUT={current_lut:.2f}, DSP={current_dsp:.2f}, CP={current_cp:.2f}, Суммарно: {float(np.asarray(final_reward).squeeze()):.2f}")
        '''

        # Периодически сохраняем чекпойнт и метрики
        if (episode + 1) % save_interval == 0:
            torch.save({
                'episode': episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'reward': mean_reward,
                'training_time': total_time,
                'metrics': metrics,
            }, 'actor_critic_checkpoint.pt')
            
            metrics.plot_metrics()
            metrics.save_to_csv()
        
        # Сохраняем лучшую модель по средней награде (после 100 эпизодов)
        if episode >= 100 and mean_reward > best_avg_reward:
            best_avg_reward = mean_reward
            best_episode = episode
            torch.save({
                'episode': episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'reward': mean_reward,
                'training_time': total_time,
                'metrics': metrics,
            }, 'best_actor_critic_model.pt')
            print(f"==> Сохранён новый best_model на эпизоде {episode+1} со средней наградой {mean_reward:.4f}")

        # После обновления метрик
        if episode >= 100:
            dsp_acc = np.mean(metrics.dsp_accuracy[-100:])
            mean_lut = np.mean(metrics.lut_values[-100:])
            mean_cp = np.mean(metrics.cp_values[-100:])
            combo_score = dsp_acc - 0.0001 * mean_lut - 0.01 * mean_cp
            if combo_score > best_combo_score:
                best_combo_score = combo_score
                torch.save({
                    'episode': episode,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'reward': mean_reward,
                    'dsp_acc': dsp_acc,
                    'mean_lut': mean_lut,
                    'mean_cp': mean_cp,
                    'combo_score': combo_score,
                    'training_time': total_time,
                    'metrics': metrics,
                }, 'best_actor_critic_model_combo.pt')
                print(f"==> [BEST MODEL] Эпизод {episode+1}: Новый лучший combo_score={combo_score:.4f} (DSP_acc={dsp_acc:.4f}, LUT={mean_lut:.2f}, CP={mean_cp:.2f}) — модель сохранена!")

        # Сохраняем промежуточные чекпоинты каждые 1000 эпизодов
        if (episode + 1) % 1000 == 0:
            torch.save({
                'episode': episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'reward': mean_reward,
                'training_time': total_time,
                'metrics': metrics,
            }, f'actor_critic_checkpoint_ep{episode+1}.pt')
            metrics.plot_metrics(f'training_metrics_ep{episode+1}.png')
            metrics.save_to_csv(f'training_metrics_ep{episode+1}.csv')
            print(f"==> Сохранён чекпоинт на эпизоде {episode+1}")

        # После обновления метрик
        if (episode + 1) % 100 == 0:
            last_episodes = min(100, len(metrics.dsp_values))
            print("Распределение target DSP за последние 100 эпизодов:", [all_pairs[i][1] for i in range(episode-last_episodes+1, episode+1)])
            print("Ошибки DSP за последние 100 эпизодов:", [abs(metrics.dsp_values[i] - all_pairs[i][1]) for i in range(episode-last_episodes+1, episode+1)])

if __name__ == '__main__':
    # Параметры обучения
    params = {
        'num_episodes': 10000,
        'learning_rate': 0.001,
        'gamma': 0.99,
        'alpha': 0.001,
        'lambda0': 0.5,
        'save_interval': 10,
        'patience': 50,
        'min_delta': 0.001,
        'min_episodes': 200
    }
    
    train_rl(**params) 