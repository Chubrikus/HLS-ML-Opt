from Run_one_case import run_one_case
import os
import sys
import argparse

def run_all_cases(num_cases=10, samples=10, start_solution=1, case_start=1):
    """
    Запуск синтеза для всех сгенерированных кейсов
    
    Args:
        num_cases: Количество кейсов для обработки
        samples: Количество комбинаций директив для каждого кейса
        start_solution: Начальный номер решения
        case_start: Начальный номер кейса
    """
    total_success = 0
    
    # Вычисляем базовую директорию с кейсами
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gen_cases")
    
    print(f"Запуск {num_cases} кейсов с {samples} вариантами директив для каждого")
    print(f"Начальный кейс: {case_start}, начальное решение: {start_solution}")
    
    for case_id in range(case_start, case_start + num_cases):
        case_dir = os.path.join(base_dir, f"case_{case_id}")
        
        # Проверяем существование директории кейса
        if not os.path.exists(case_dir):
            print(f"Предупреждение: Директория для кейса {case_id} не найдена: {case_dir}")
            continue
        
        print(f"\n{'='*50}")
        print(f"Обработка кейса {case_id}")
        print(f"{'='*50}")
        
        try:
            success = run_one_case(
                case_id=case_id,
                case_dir=case_dir,
                samp_num=samples,
                sol_start=start_solution
            )
            
            if success:
                total_success += 1
                print(f"Кейс {case_id} успешно обработан")
            else:
                print(f"Ошибка при обработке кейса {case_id}")
                
        except Exception as e:
            print(f"Исключение при обработке кейса {case_id}: {e}")
    
    print(f"\n{'='*50}")
    print(f"Обработка завершена")
    print(f"Успешно обработано кейсов: {total_success} из {num_cases}")
    print(f"{'='*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск синтеза для всех кейсов")
    parser.add_argument('--num_cases', type=int, default=10, help='Количество кейсов для обработки')
    parser.add_argument('--samples', type=int, default=10, help='Количество комбинаций директив для каждого кейса')
    parser.add_argument('--start_solution', type=int, default=1, help='Начальный номер решения')
    parser.add_argument('--case_start', type=int, default=1, help='Начальный номер кейса')
    
    args = parser.parse_args()
    
    run_all_cases(
        num_cases=args.num_cases,
        samples=args.samples,
        start_solution=args.start_solution,
        case_start=args.case_start
    ) 