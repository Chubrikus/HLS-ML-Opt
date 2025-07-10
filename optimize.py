import torch
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os
import pickle
import json
import sys
import argparse
import torch.nn as nn

from train_rl import ActorCritic, TrainingMetrics
from rl_env import RLEnv

# Добавляем пути для импорта
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
gpp9_dir = os.path.join(os.path.dirname(parent_dir), 'gpp9')
gnn_dir = os.path.join(gpp9_dir, 'GNN')

# Добавляем все необходимые пути
sys.path.extend([
    parent_dir,  # new_opt
    gpp9_dir,    # gpp9
    gnn_dir,     # gpp9/GNN
    os.path.join(parent_dir, 'GNN')  # new_opt/GNN
])

def optimize_fir_filter(case_folder, model_path, num_iterations=20, device=None):
    """Оптимизация FIR фильтра или другого кейса с помощью обученной модели"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")

    # Создаем директорию для результатов
    output_dir = os.path.join(case_folder, "opt_res")
    os.makedirs(output_dir, exist_ok=True)

    print("Загрузка чекпойнта...")
    checkpoint = torch.load(model_path, map_location=device)
    print("Чекпойнт загружен.")

    print("Инициализация модели...")
    model = ActorCritic(input_dim=633).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Модель инициализирована и переведена в режим оценки.")

    print("Инициализация окружения...")
    # --- Поиск graph.pkl ---
    graph_path = os.path.join(case_folder, "processed", "graph.pkl")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Файл {graph_path} не найден!")
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    env = RLEnv()
    env.graphs = [graph]
    print("Окружение инициализировано.")

    # --- Поиск .json с метриками ---
    metrics_json = None
    for fname in os.listdir(case_folder):
        if fname.endswith('.json'):
            metrics_json = os.path.join(case_folder, fname)
            break
    if metrics_json is None:
        raise FileNotFoundError(f"В папке {case_folder} не найдено ни одного .json файла с метриками!")
    with open(metrics_json, 'r') as f:
        metrics = json.load(f)

    results = {
        'iteration': [], 'RL_CP': [], 'Target_CP': [], 'Delta_CP': [],
        'RL_DSP': [], 'Target_DSP': [], 'Delta_DSP': [],
        'RL_LUT': [], 'Target_LUT': [], 'Delta_LUT': [],
        'LUT_nodes': [], 'DSP_nodes': []
    }

    # --- Получаем целевые значения из метрик (ищем первые попавшиеся ключи) ---
    first_key = next(iter(metrics))
    target_cp = metrics[first_key].get('CP_post_synthesis', 0)
    target_dsp = metrics[first_key].get('DSP', 0)
    target_lut = metrics[first_key].get('LUT', 0)
    print(f"Целевые значения: CP={target_cp}, DSP={target_dsp}, LUT={target_lut}")

    for i in range(num_iterations):
        print(f"\nИтерация {i+1}/{num_iterations}")
        state = env.reset(0, target_dsp)
        step = 0
        while True:
            step += 1
            state_tensor = torch.FloatTensor(state).to(device)
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)
            with torch.no_grad():
                action_prob, _ = model(state_tensor)
                action = torch.multinomial(action_prob, 1).item()
            next_state, reward, done, current_lut, current_dsp, current_cp = env.step(action)
            state = next_state
            if done:
                print(f"Итерация завершена. LUT={current_lut:.2f}, DSP={current_dsp:.2f}, CP={current_cp:.2f}")
                lut_nodes = [node for node, is_dsp in env.dsp_assignments.items() if not is_dsp]
                dsp_nodes = [node for node, is_dsp in env.dsp_assignments.items() if is_dsp]
                print("\nРаспределение узлов:")
                print("LUT узлы:", lut_nodes)
                print("DSP узлы:", dsp_nodes)
                break
        results['iteration'].append(i+1)
        results['RL_CP'].append(current_cp)
        results['Target_CP'].append(target_cp)
        results['Delta_CP'].append(current_cp - target_cp)
        results['RL_DSP'].append(current_dsp)
        results['Target_DSP'].append(target_dsp)
        results['Delta_DSP'].append(current_dsp - target_dsp)
        results['RL_LUT'].append(current_lut)
        results['Target_LUT'].append(target_lut)
        results['Delta_LUT'].append(current_lut - target_lut)
        results['LUT_nodes'].append(lut_nodes)
        results['DSP_nodes'].append(dsp_nodes)

    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("\nСохранение результатов в CSV...")
    df.to_csv(os.path.join(output_dir, f'optimization_results_{timestamp}.csv'), index=False)
    print(f"Результаты сохранены в {output_dir}/optimization_results_{timestamp}.csv")

    best_iteration = df['Delta_CP'].abs() + df['Delta_DSP'].abs() + df['Delta_LUT'].abs()
    best_iteration = best_iteration.idxmin()
    print("\nЛучшее распределение узлов (итерация", best_iteration + 1, "):")
    print("LUT узлы:", df['LUT_nodes'].iloc[best_iteration])
    print("DSP узлы:", df['DSP_nodes'].iloc[best_iteration])
    print("\nМетрики для лучшего распределения:")
    print(f"CP: {df['RL_CP'].iloc[best_iteration]:.2f} (целевое: {target_cp:.2f})")
    print(f"DSP: {df['RL_DSP'].iloc[best_iteration]:.2f} (целевое: {target_dsp:.2f})")
    print(f"LUT: {df['RL_LUT'].iloc[best_iteration]:.2f} (целевое: {target_lut:.2f})")
    return df

def optimize_with_learning():
    """Оптимизация с дообучением модели на конкретном графе"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")

    # Спрашиваем у пользователя папку кейса
    case_folder = input("Введите путь к папке с кейсом: ")
    if not os.path.isdir(case_folder):
        print(f"Папка {case_folder} не найдена! Проверьте путь.")
        return

    # Спрашиваем у пользователя целевой DSP
    target_dsp = float(input("Введите целевой DSP: "))

    # Фиксированные значения для alpha и lambda0
    alpha = 0.001  # Вес для LUT
    lambda0 = 0.25  # Вес для CP
    print(f"Используются веса: alpha={alpha}, lambda0={lambda0}")

    # Создаем директорию для результатов
    output_dir = os.path.join(case_folder, "opt_res")
    os.makedirs(output_dir, exist_ok=True)

    print("Инициализация новой модели...")
    model = ActorCritic(input_dim=388).to(device)
    
    # Создаем оптимизатор с фиксированной скоростью обучения
    initial_lr = 0.001  # Фиксированная скорость обучения
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    print(f"Инициализирован оптимизатор с фиксированной скоростью обучения {initial_lr}")
    print("Модель инициализирована.")

    print("Инициализация окружения...")
    graph_path = os.path.join(case_folder, "processed", "graph.pkl")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Файл {graph_path} не найден!")
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    env = RLEnv(alpha=alpha, lambda0=lambda0)
    env.graphs = [graph]

    # Загрузка метаданных
    metadata_df = pd.read_csv('optimizer/case_40/processed/metadata.csv')

    # Передача метаданных в среду
    env.metadata = metadata_df.iloc[0].values  # Предполагается, что метаданные соответствуют графу
    print("Окружение инициализировано.")

    results = {
        'iteration': [], 'RL_CP': [], 'Target_CP': [], 'Delta_CP': [],
        'RL_DSP': [], 'Target_DSP': [], 'Delta_DSP': [],
        'RL_LUT': [], 'Target_LUT': [], 'Delta_LUT': [],
        'LUT_nodes': [], 'DSP_nodes': [], 'Reward': []
    }

    best_solutions = []  # Список для хранения топ 5 решений
    metrics = TrainingMetrics()

    # Параметры для ранней остановки
    max_iterations = 9000  # Максимальное количество итераций
    window_size = 50  # Увеличиваем размер окна для скользящего среднего
    min_improvement = 0.001  # Уменьшаем требуемое улучшение
    patience = 200  # Увеличиваем терпение
    best_reward = float('-inf')
    no_improvement_count = 0

    # Создаем фигуру для сохранения графиков
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # График наград
    reward_line, = ax1.plot([], [], 'b-', label='Награда')
    ax1.set_title('Награда в процессе обучения')
    ax1.set_xlabel('Итерация')
    ax1.set_ylabel('Награда')
    ax1.legend()
    ax1.grid(True)

    # График DSP
    dsp_line, = ax2.plot([], [], 'r-', label='DSP')
    ax2.axhline(y=target_dsp, color='r', linestyle='--', label='Целевой DSP')
    ax2.set_title('DSP в процессе обучения')
    ax2.set_xlabel('Итерация')
    ax2.set_ylabel('DSP')
    ax2.legend()
    ax2.grid(True)

    # График LUT
    lut_line, = ax3.plot([], [], 'g-', label='LUT')
    ax3.set_title('LUT в процессе обучения')
    ax3.set_xlabel('Итерация')
    ax3.set_ylabel('LUT')
    ax3.legend()
    ax3.grid(True)

    # График CP
    cp_line, = ax4.plot([], [], 'm-', label='CP')
    ax4.set_title('CP в процессе обучения')
    ax4.set_xlabel('Итерация')
    ax4.set_ylabel('CP')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()

    # Списки для хранения данных для графиков
    iterations = []
    rewards = []
    dsps = []
    luts = []
    cps = []
    moving_averages = []  # Список для хранения скользящих средних

    for episode in range(max_iterations):
        model.train()
        state = env.reset(0, target_dsp)
        episode_rewards = []
        episode_log_probs = []
        episode_values = []

        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_probs, value = model(state_tensor)
            action = torch.multinomial(action_probs, 1).item()
            log_prob = torch.log(action_probs[0, action])

            # Логируем распределение действий
            #print(f"Эпизод {episode+1}, шаг {env.current_mult_index}: action={action} (DSP={action==1}, LUT={action==0}), prob_DSP={action_probs[0,1].item():.3f}, prob_LUT={action_probs[0,0].item():.3f}")

            next_state, reward, done, current_lut, current_dsp, current_cp = env.step(action)

            episode_rewards.append(reward)
            episode_log_probs.append(log_prob)
            episode_values.append(value)

            if done:
                break
            state = next_state

        # Обновляем метрики
        total_reward = sum(episode_rewards)
        metrics.add_episode(total_reward, current_lut, current_dsp, current_cp, target_dsp, env.mult_ops, env.dsp_assigned)
        
        # Обучаем модель на собранном опыте
        returns = compute_returns(episode_rewards)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        
        '''
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)
            print(f"returns: {returns}")  
'''
        optimizer.zero_grad()
        policy_losses = []
        value_losses = []
        huber_loss = nn.HuberLoss()
        
        for log_prob, value, R in zip(episode_log_probs, episode_values, returns):
            advantage = R - value.item()
            policy_losses.append(-log_prob * advantage)
            value_losses.append(huber_loss(value, R.reshape(1, 1)))
        
        # Энтропийный бонус для стимулирования исследования
        entropy = -(action_probs * action_probs.log()).sum()
        entropy_coef = 0.86 #max(0.01, 1 * (0.995**episode))
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum() - entropy_coef * entropy
        loss.backward()
        optimizer.step()

        # Сохраняем результаты
        results['iteration'].append(episode+1)
        results['RL_CP'].append(current_cp)
        results['Target_CP'].append(env.target_cp)
        results['Delta_CP'].append(current_cp - env.target_cp)
        results['RL_DSP'].append(current_dsp)
        results['Target_DSP'].append(target_dsp)
        results['Delta_DSP'].append(current_dsp - target_dsp)
        results['RL_LUT'].append(current_lut)
        results['Target_LUT'].append(env.target_lut)
        results['Delta_LUT'].append(current_lut - env.target_lut)
        results['LUT_nodes'].append([node for node, is_dsp in env.dsp_assignments.items() if not is_dsp])
        results['DSP_nodes'].append([node for node, is_dsp in env.dsp_assignments.items() if is_dsp])
        results['Reward'].append(total_reward)

        # Обновляем данные для графиков
        iterations.append(episode + 1)
        rewards.append(total_reward)
        dsps.append(current_dsp)
        luts.append(current_lut)
        cps.append(current_cp)

        # Вычисляем скользящее среднее
        if len(rewards) >= window_size:
            current_ma = sum(rewards[-window_size:]) / window_size
            moving_averages.append(current_ma)
            
            # Проверяем улучшение скользящего среднего
            if len(moving_averages) > 1:
                improvement = (current_ma - moving_averages[-2]) / abs(moving_averages[-2] + 1e-6)
                if improvement < min_improvement:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0
                    best_reward = current_ma

        # Обновляем графики
        reward_line.set_data(iterations, rewards)
        dsp_line.set_data(iterations, dsps)
        lut_line.set_data(iterations, luts)
        cp_line.set_data(iterations, cps)

        # Обновляем масштаб для всех графиков
        for ax in [ax1, ax2, ax3, ax4]:
            ax.relim()
            ax.autoscale_view()

        # Сохраняем графики каждые 10 итераций
        if (episode + 1) % 10 == 0:
            plt.savefig(os.path.join(output_dir, f'training_metrics_current.png'))

        # Сохраняем решение если оно лучше предыдущего
        if total_reward > best_reward:
            best_solutions.append({
                'reward': total_reward,
                'dsp_nodes': [node for node, is_dsp in env.dsp_assignments.items() if is_dsp],
                'lut_nodes': [node for node, is_dsp in env.dsp_assignments.items() if not is_dsp],
                'dsp_value': current_dsp,
                'lut_value': current_lut,
                'cp_value': current_cp
            })
            # Сортируем и оставляем топ 15
            best_solutions.sort(key=lambda x: x['reward'], reverse=True)
            best_solutions = best_solutions[:15]
            # Сохраняем топ-15 решений на лету
            best_solutions_df = pd.DataFrame(best_solutions)
            best_solutions_df.to_csv(os.path.join(output_dir, 'best_solutions_live.csv'), index=False)

        print(f"Эпизод {episode+1}: Награда = {total_reward:.2f}, "
              f"Loss = {loss.item():.4f}, "
              f"DSP = {current_dsp:.2f}, "
              f"LUT = {current_lut:.2f}, "
              f"CP = {current_cp:.2f}")

        # Проверяем условие остановки
        if len(moving_averages) > 0 and no_improvement_count >= patience:
            print(f"\nРанняя остановка: средняя награда не улучшается в течение {patience} итераций")
            print(f"Последнее среднее значение награды: {moving_averages[-1]:.2f}")
            break

    # Сохраняем финальные графики
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(output_dir, f'training_metrics_{timestamp}.png'))
    plt.close()

    # Сохраняем все результаты
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, f'optimization_results_{timestamp}.csv'), index=False)

    # Сохраняем графики метрик
    metrics.plot_metrics(os.path.join(output_dir, f'training_metrics_{timestamp}.png'))
    metrics.save_to_csv(os.path.join(output_dir, f'training_metrics_{timestamp}.csv'))

    return df

def compute_returns(rewards, gamma=0.99):
    """Вычисление дисконтированных возвратов."""
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def update_model(model, optimizer, log_probs, values, returns):
    """Обновление модели на основе собранного опыта."""
    optimizer.zero_grad()
    policy_losses = []
    value_losses = []
    huber_loss = nn.HuberLoss()
    for log_prob, value, R in zip(log_probs, values, returns):
        advantage = R - value.item()
        policy_losses.append(-log_prob * advantage)
        value_losses.append(huber_loss(value, torch.tensor(R).reshape(1, 1).to(value.device)))
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    return loss.item(), sum(value_losses).item()

if __name__ == '__main__':
    optimize_with_learning()
