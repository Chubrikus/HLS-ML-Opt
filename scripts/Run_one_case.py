import os
import subprocess
import sys
import json
import random
import re
import shutil
import time
import threading
import signal

def get_LUT_op_list(directives):
    """
    Извлекает список операторов, использующих LUT из директив
    """
    LUT_op_list = []
    for direct in directives:
        if direct.startswith('#'):
            continue
        res = re.findall(r'\d+', direct)
        if len(res) >= 2:
            LUT_op_list.append(int(res[1]))
    return LUT_op_list

def extract_cp_from_verilog(case_dir):
    """
    Извлекает значение CP из RTL файлов (case_1.v или case_1.vhd)
    
    Args:
        case_dir: Директория кейса
    
    Returns:
        float: Значение CP, или 0 если не удалось найти
    """
    # Пути к возможным файлам
    paths_to_check = [
        os.path.join(case_dir, "project_tmp", "solution_tmp", "syn", "verilog"),
        os.path.join(case_dir, "project_tmp", "solution_tmp", "syn", "vhdl"),
        os.path.join(case_dir, "project_tmp", "solution_tmp", "impl", "verilog"),
        os.path.join(case_dir, "project_tmp", "solution_tmp", "impl", "vhdl"),
        os.path.join(case_dir, "project_tmp", "solution_tmp", "impl", "ip", "hdl", "verilog"),
        os.path.join(case_dir, "project_tmp", "solution_tmp", "impl", "ip", "hdl", "vhdl")
    ]
    
    # Ищем файлы .v или .vhd
    for path in paths_to_check:
        if not os.path.exists(path):
            continue
            
        for file in os.listdir(path):
            if file.endswith(".v") or file.endswith(".vhd"):
                file_path = os.path.join(path, file)
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        # Ищем строку с HLS_SYN_CLOCK
                        match = re.search(r'HLS_SYN_CLOCK=(\d+\.\d+)', content)
                        if match:
                            cp_value = float(match.group(1))
                            print(f"Найдено значение CP в файле {file_path}: {cp_value}")
                            return cp_value
                except Exception as e:
                    print(f"Ошибка при чтении файла {file_path}: {e}")
    
    print("Не удалось найти значение CP в RTL файлах")
    return 0

def run_one_case(case_id, case_dir=None, samp_num=10, sol_start=1):
    """
    Запуск синтеза для указанного кейса с различными комбинациями директив
    
    Args:
        case_id: ID кейса для синтеза
        case_dir: Директория кейса (если None, будет использован путь gen_cases/case_{case_id})
        samp_num: Количество комбинаций директив для проверки
        sol_start: Начальный номер решения
    """
    print(f"Запуск кейса {case_id} с {samp_num} вариантами директив")
    
    # Путь к директории кейса
    if case_dir is None:
        case_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gen_cases", f"case_{case_id}")
    
    if not os.path.exists(case_dir):
        print(f"Ошибка: Директория кейса не найдена: {case_dir}")
        return False
        
    # Проверяем наличие необходимых файлов
    script_file = os.path.join(case_dir, "script.tcl")
    directive_file = os.path.join(case_dir, "directive.tcl")
    
    if not os.path.exists(script_file) or not os.path.exists(directive_file):
        print(f"Ошибка: Необходимые файлы не найдены в {case_dir}")
        return False
    
    # Чтение скрипта и директив
    with open(script_file, 'r') as f:
        script_template = f.read()
    
    with open(directive_file, 'r') as f:
        directive_content = f.read()
    
    directive_list = directive_content.splitlines()
    
    # Генерация комбинаций директив
    all_directive_lists = []
    
    # Базовый вариант - все директивы активны
    all_directive_lists.append(directive_list)
    
    # Генерация случайных комбинаций
    for i in range(1, samp_num):
        direct_list_sampled = []
        for directive in directive_list:
            ri = random.uniform(0, i%17+4)
            if ri <= 3:
                directive = "# " + directive  # Случайно комментируем директивы
            direct_list_sampled.append(directive)
        all_directive_lists.append(direct_list_sampled)
    
    print(f"Сгенерировано {len(all_directive_lists)} комбинаций директив")
    
    # JSON файл для сохранения результатов
    json_file = os.path.join(case_dir, f"case_{case_id}_all_data.json")
    
    # Инициализация JSON, если начинаем с первого решения
    if sol_start == 1:
        all_solutions = {}
        with open(json_file, "w") as f:
            json.dump(all_solutions, f)
    else:
        # Загружаем существующие данные
        try:
            with open(json_file, "r") as f:
                all_solutions = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_solutions = {}
    
    sol = sol_start
    
    # Путь к Vitis HLS
    vitis_hls_path = r"D:\Xilinx\Vitis_HLS\2024.2"
    
    # Проходим по всем комбинациям директив
    for directives in all_directive_lists:
        print(f"Генерация директив для solution_{sol}")
        
        # Создаем временные файлы
        tmp_directive_file = os.path.join(case_dir, "directive_tmp.tcl")
        with open(tmp_directive_file, "w") as f:
            for directive in directives:
                f.write(directive + "\n")
        
        tmp_script_file = os.path.join(case_dir, "script_tmp.tcl")
        
        # Модифицируем скрипт для временного решения
        script_content = script_template
        # Заменяем только важные части, а не все названия, избегаем дублирования -reset
        if 'open_project -reset' in script_content:
            script_content = script_content.replace('open_project -reset', 'open_project -reset')
        else:
            script_content = script_content.replace('open_project', 'open_project -reset')
            
        if 'open_solution -reset' in script_content:
            script_content = script_content.replace('open_solution -reset', 'open_solution -reset')
        else:
            script_content = script_content.replace('open_solution', 'open_solution -reset')
            
        script_content = script_content.replace('solution_1', 'solution_tmp')
        script_content = script_content.replace('project_1', 'project_tmp')
        script_content = script_content.replace('source "./directive.tcl"', 'source "./directive_tmp.tcl"')
        
        with open(tmp_script_file, "w") as f:
            f.write(script_content)
            
        print(f"Создан временный tcl скрипт: {tmp_script_file}")
        print(f"Содержимое скрипта:")
        with open(tmp_script_file, 'r') as f:
            print(f.read())
        
        print(f"Запуск Vitis HLS для solution_{sol}")
        
        # Настройка окружения
        env = os.environ.copy()
        
        # Добавляем путь к tee.exe из Vivado
        vivado_bin = r"D:\Xilinx\Vivado\2024.2\gnuwin\bin"
        env["PATH"] = vivado_bin + os.pathsep + os.path.join(vitis_hls_path, "bin") + os.pathsep + env["PATH"]
        env["XILINX_HLS"] = vitis_hls_path
        
        # Печатаем PATH для отладки
        print(f"PATH: {env['PATH']}")
        
        # Проверяем наличие tee.exe и копируем его в директорию кейса
        tee_path = os.path.join(vivado_bin, "tee.exe")
        if os.path.exists(tee_path):
            print(f"tee.exe найден: {tee_path}")
            # Копируем tee.exe в директорию кейса
            local_tee = os.path.join(case_dir, "tee.exe")
            shutil.copy2(tee_path, local_tee)
            print(f"tee.exe скопирован в: {local_tee}")
            # Добавляем директорию кейса в PATH
            env["PATH"] = case_dir + os.pathsep + env["PATH"]
        else:
            print(f"tee.exe не найден по пути: {tee_path}")
            # Ищем tee.exe в других местах
            for root_dir in [r"D:\Xilinx\Vivado", r"D:\Xilinx\Vitis"]:
                for root, dirs, files in os.walk(root_dir):
                    if "tee.exe" in files:
                        found_tee = os.path.join(root, "tee.exe")
                        print(f"tee.exe найден в: {found_tee}")
                        # Копируем tee.exe в директорию кейса
                        local_tee = os.path.join(case_dir, "tee.exe")
                        shutil.copy2(found_tee, local_tee)
                        print(f"tee.exe скопирован в: {local_tee}")
                        # Добавляем директорию кейса в PATH
                        env["PATH"] = case_dir + os.pathsep + env["PATH"]
                        break
                else:
                    continue
                break
        
        # Создаем bat-файл для запуска HLS
        run_hls_bat = os.path.join(case_dir, "run_hls_tmp.bat")
        with open(run_hls_bat, "w") as f:
            f.write("@echo on\n")  # Включаем вывод всех команд
            f.write(f'cd "{case_dir}"\n')
            f.write('echo Current directory: %CD%\n')
            
            # Устанавливаем PATH с tee.exe
            f.write(f'SET PATH={case_dir};{vivado_bin};{os.path.join(vitis_hls_path, "bin")};%PATH%\n')
            f.write('echo PATH=%PATH%\n')
            
            # Проверяем наличие tee.exe
            f.write('where tee.exe\n')
            
            f.write(f'call "{vitis_hls_path}\\bin\\setupEnv.bat"\n')
            f.write('echo HLS environment setup complete\n')
            f.write(f'if exist script_tmp.tcl echo Found script_tmp.tcl\n')
            f.write(f'if exist directive_tmp.tcl echo Found directive_tmp.tcl\n')
            
            # Запускаем HLS
            f.write(f'call "{vitis_hls_path}\\bin\\vitis_hls.bat" -f script_tmp.tcl\n')
            f.write('echo Vitis HLS exit code: %ERRORLEVEL%\n')
            f.write('exit %ERRORLEVEL%\n')
        
        print(f"Создан bat-файл для запуска HLS: {run_hls_bat}")
        
        # Запуск процесса с выводом вывода в реальном времени
        print(f"Запуск процесса HLS...")
        try:
            # Запускаем процесс с отображением вывода
            process = subprocess.Popen([run_hls_bat], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
            
            # Переменная для отслеживания последнего вывода
            last_output_time = time.time()
            process_terminated = False
            
            # Функция для обновления времени последнего вывода
            def update_last_output_time():
                nonlocal last_output_time
                last_output_time = time.time()
            
            # Функция для проверки таймаута
            def check_timeout():
                nonlocal process_terminated
                while process.poll() is None and not process_terminated:
                    current_time = time.time()
                    if current_time - last_output_time > 300:  # 5 минут = 300 секунд
                        print(f"Таймаут: нет вывода в течение 5 минут. Завершаем процесс...")
                        process_terminated = True
                        if sys.platform == 'win32':
                            # Для Windows
                            try:
                                process.terminate()
                                # Даем процессу 10 секунд на завершение
                                time.sleep(10)
                                if process.poll() is None:
                                    process.kill()
                            except Exception as e:
                                print(f"Ошибка при завершении процесса: {e}")
                        else:
                            # Для Unix/Linux
                            try:
                                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                                # Даем процессу 10 секунд на завершение
                                time.sleep(10)
                                if process.poll() is None:
                                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                            except Exception as e:
                                print(f"Ошибка при завершении процесса: {e}")
                    time.sleep(10)  # Проверяем каждые 10 секунд
            
            # Запускаем поток для проверки таймаута
            timeout_thread = threading.Thread(target=check_timeout)
            timeout_thread.daemon = True
            timeout_thread.start()
            
            # Вывод в реальном времени
            for line in iter(process.stdout.readline, ''):
                print(line.strip())
                update_last_output_time()
            
            process.stdout.close()
            return_code = process.wait()
            process_terminated = True
            
            # Если процесс был завершен по таймауту
            if timeout_thread.is_alive():
                timeout_thread.join(1)  # Даем потоку 1 секунду на завершение
            
            if process_terminated and return_code is None:
                print(f"Процесс был завершен по таймауту")
                sol += 1
                continue
            
            # Проверяем код возврата
            if return_code != 0:
                print(f"Ошибка при выполнении Vitis HLS для solution_{sol}: код {return_code}")
                sol += 1
                continue
                
            # Проверяем результаты без чтения лог-файла
            print(f"Vitis HLS завершил работу для solution_{sol}")
            
            # Проверяем, была ли создана директория проекта
            project_dir = os.path.join(case_dir, "project_tmp")
            if not os.path.exists(project_dir):
                print(f"Ошибка: Директория проекта не создана: {project_dir}")
                sol += 1
                continue
                
            print(f"Директория проекта создана: {project_dir}")
        except Exception as e:
            print(f"Исключение при запуске процесса: {e}")
            sol += 1
            continue
        
        # Анализ отчета о синтезе
        rpt_path = os.path.join(case_dir, "project_tmp", "solution_tmp", "impl", "report", "verilog")
        if not os.path.exists(rpt_path):
            print(f"Директория отчетов не найдена: {rpt_path}")
            sol += 1
            continue
        
        # Поиск файла отчета
        export_rpt = None
        for file in os.listdir(rpt_path):
            if file.endswith("_export.rpt"):
                export_rpt = os.path.join(rpt_path, file)
                break
        
        if not export_rpt:
            print("Файл отчета не найден")
            sol += 1
            continue
        
        # Извлечение метрик из отчета
        SLICE = LUT = FF = DSP = 0
        CP_post_synthesis = 0
        CP_post_implementation = 0
        
        with open(export_rpt, 'r') as f:
            for line in f:
                res = [i for i in line.split() if i.isdigit()]
                if line.startswith('SLICE') and res:
                    SLICE = int(res[0])
                elif line.startswith('LUT') and res:
                    LUT = int(res[0])
                elif line.startswith('FF') and res:
                    FF = int(res[0])
                elif line.startswith('DSP') and res:
                    DSP = int(res[0])
                elif 'CP achieved post-synthesis' in line:
                    try:
                        CP_post_synthesis = float(line.split(':')[1].strip())
                        print(f"Найдено значение CP после синтеза: {CP_post_synthesis}")
                    except (ValueError, IndexError):
                        print(f"Ошибка при извлечении CP из строки: {line}")
                elif 'CP achieved post-implementation' in line:
                    try:
                        CP_post_implementation = float(line.split(':')[1].strip())
                        print(f"Найдено значение CP после реализации: {CP_post_implementation}")
                    except (ValueError, IndexError):
                        print(f"Ошибка при извлечении CP из строки: {line}")
        
        # Если CP не найден в отчете, пробуем извлечь его из RTL файлов
        CP_hls_metadata = 0
        if CP_post_synthesis == 0 and CP_post_implementation == 0:
            print("CP не найден в отчете, пробуем извлечь из RTL файлов")
            CP_hls_metadata = extract_cp_from_verilog(case_dir)
            if CP_hls_metadata > 0:
                print(f"Успешно извлечен CP из RTL файлов: {CP_hls_metadata}")
        
        print(f"Результаты: SLICE={SLICE}, LUT={LUT}, FF={FF}, DSP={DSP}, CP_synthesis={CP_post_synthesis}, CP_implementation={CP_post_implementation}, CP_metadata={CP_hls_metadata}")
        
        # Сохранение результатов в JSON
        all_solutions[f"solution_{sol}"] = {
            'directives': directives,
            'LUT_op': get_LUT_op_list(directives),
            'SLICE': SLICE,
            'LUT': LUT,
            'FF': FF,
            'DSP': DSP,
            'CP_post_synthesis': CP_post_synthesis,
            'CP_post_implementation': CP_post_implementation,
            'CP_hls_metadata': CP_hls_metadata
        }
        
        with open(json_file, "w") as f:
            json.dump(all_solutions, f, indent=2)
        
        print(f"Результаты для solution_{sol} сохранены в {json_file}")
        
        # Увеличиваем счетчик решений
        sol += 1
    
    print(f"Обработка кейса {case_id} завершена. Всего решений: {sol - sol_start}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python Run_one_case.py <case_id> [samples] [start_solution]")
        sys.exit(1)
        
    try:
        case_id = int(sys.argv[1])
        samp_num = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        sol_start = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        
        run_one_case(case_id, samp_num=samp_num, sol_start=sol_start)
    except ValueError:
        print("Ошибка: аргументы должны быть числами")
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1) 