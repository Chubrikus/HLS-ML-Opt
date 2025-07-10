@echo on
cd "D:\Gen\project2\gen_cases\case_1"
echo Current directory: %CD%
SET PATH=D:\Gen\project2\gen_cases\case_1;D:\Xilinx\Vivado\2024.2\gnuwin\bin;D:\Xilinx\Vitis_HLS\2024.2\bin;%PATH%
echo PATH=%PATH%
where tee.exe
call "D:\Xilinx\Vitis_HLS\2024.2\bin\setupEnv.bat"
echo HLS environment setup complete
if exist script_tmp.tcl echo Found script_tmp.tcl
if exist directive_tmp.tcl echo Found directive_tmp.tcl
call "D:\Xilinx\Vitis_HLS\2024.2\bin\vitis_hls.bat" -f script_tmp.tcl
echo Vitis HLS exit code: %ERRORLEVEL%
exit %ERRORLEVEL%
