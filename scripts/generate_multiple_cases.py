from Gen_one_case import Gen_one_case
import os
import sys

def generate_multiple_cases(start_case=1, num_cases=10, max_prim_in=30, max_op_cnt=200):
    """
    Генерирует несколько кейсов последовательно
    
    Args:
        start_case: Начальный номер кейса
        num_cases: Количество кейсов для генерации
        max_prim_in: Максимальное количество первичных входов
        max_op_cnt: Максимальное количество операций
    """
    # Создаем папку для кейсов в корневой директории проекта
    cases_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gen_cases")
    if not os.path.exists(cases_dir):
        os.makedirs(cases_dir)
    
    for case_id in range(start_case, start_case + num_cases):
        print(f"\n{'='*50}")
        print(f"Генерация кейса {case_id}")
        print(f"{'='*50}")
        
        # Генерируем кейс
        Gen_one_case(case_id=case_id, max_prim_in=max_prim_in, max_op_cnt=max_op_cnt)
        
        # Перемещаем файлы в папку кейса
        case_dir = os.path.join(cases_dir, f"case_{case_id}")
        if not os.path.exists(case_dir):
            os.makedirs(case_dir)
        
        # Перемещаем все сгенерированные файлы
        files_to_move = [
            f"DFG_case_{case_id}.txt",
            f"case_{case_id}.cc",
            "directive.tcl",
            "script.tcl"
        ]
        
        for file in files_to_move:
            if os.path.exists(file):
                try:
                    os.rename(file, os.path.join(case_dir, file))
                    print(f"Файл {file} перемещен в {case_dir}")
                except Exception as e:
                    print(f"Ошибка при перемещении файла {file}: {e}")
        
        print(f"Кейс {case_id} успешно сгенерирован!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        start_case = int(sys.argv[1])
    else:
        start_case = 1  # По умолчанию начинаем с case_1
    
    if len(sys.argv) > 2:
        num_cases = int(sys.argv[2])
    else:
        num_cases = 10  # По умолчанию генерируем 10 кейсов
    
    if len(sys.argv) > 3:
        max_prim_in = int(sys.argv[3])
    else:
        max_prim_in = 30
    
    if len(sys.argv) > 4:
        max_op_cnt = int(sys.argv[4])
    else:
        max_op_cnt = 200
    
    print(f"Генерация {num_cases} кейсов, начиная с case_{start_case}")
    print(f"Параметры: max_prim_in={max_prim_in}, max_op_cnt={max_op_cnt}")
    
    generate_multiple_cases(start_case, num_cases, max_prim_in, max_op_cnt)
    
    print("\nВсе кейсы успешно сгенерированы!") 