import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rl_env import RLEnv

class ActorCritic(nn.Module):
    def __init__(self, input_dim=388, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # Общие слои с увеличенной размерностью
        self.common = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh()
        )
        
        # Actor (политика) - принимает решение о назначении DSP для текущей операции
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim // 2, 2),
            nn.Softmax(dim=-1)
        )
        
        # Critic (функция ценности) - оценивает качество текущего состояния
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Инициализация весов
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Случайная инициализация bias в actor-голове
        for m in self.actor.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.normal_(m.bias, mean=0.0, std=0.1)
        
    def forward(self, x):
        # Преобразуем входные данные в тензор и добавляем размерность батча если нужно
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(x.device if isinstance(x, torch.Tensor) else 'cuda' if torch.cuda.is_available() else 'cpu')
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        # Проверяем размерность
        if x.shape[1] != 388:
            print(f"Warning: Expected input dimension 388, got {x.shape[1]}")
            # Добавляем или обрезаем до нужной размерности
            if x.shape[1] < 388:
                padding = torch.zeros(x.shape[0], 388 - x.shape[1], device=x.device)
                x = torch.cat([x, padding], dim=1)
            else:
                x = x[:, :388]
        
        # Общие признаки
        x = self.common(x)
        
        # Получаем вероятности действий и значение
        action_prob = self.actor(x)
        value = self.critic(x)
        
        return action_prob, value
'''
def train_actor_critic(env, model, num_episodes=1000, gamma=0.99, lr=0.0005):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    huber_loss = nn.HuberLoss()
    
    for episode in range(num_episodes):
        # Выбираем случайный граф и целевое значение DSP
        graph_index = np.random.randint(0, len(env.graphs))
        target_dsp = env.target_dsp.iloc[graph_index].values[0]
        
        # Сбрасываем окружение
        state = env.reset(graph_index, target_dsp)
        
        episode_reward = 0
        done = False
        
        while not done:
            # Получаем вероятности действий и оценку ценности
            action_probs, value = model(state)
            
            # Выбираем действие
            action = torch.multinomial(action_probs, 1).item()
            
            # Выполняем действие
            next_state, reward, done, lut, dsp, cp = env.step(action)
            
            # Получаем следующую оценку ценности
            _, next_value = model(next_state)
            
            # Вычисляем TD-ошибку
            td_target = reward + gamma * next_value * (1 - done)
            td_error = td_target - value
            
            # Вычисляем потери
            actor_loss = -torch.log(action_probs[0, action]) * td_error.detach()
            critic_loss = huber_loss(value, td_target.detach())
            loss = actor_loss + critic_loss
            
            # Обновляем веса
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Обновляем состояние
            state = next_state
            episode_reward += reward
            
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}, LUT: {lut:.2f}, DSP: {dsp:.2f}, CP: {cp:.2f}")
            '''

'''
def main():
    # Создаем окружение
    env = RLEnv(alpha=0.5, lambda0=0.5)
    
    # Создаем модель
    input_dim = 388  # Размерность состояния (128 * 3 + 2 + 1)
    model = ActorCritic(input_dim)
    
    # Обучаем модель
    train_actor_critic(env, model)
    
    # Сохраняем модель
    torch.save(model.state_dict(), 'actor_critic_model.pt')

if __name__ == '__main__':
    main()
    '''
