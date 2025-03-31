addpath(pwd);  % 将当前工作目录（pwd）添加到 MATLAB 的搜索路径中，方便调用当前目录下的函数和脚本。
addpath(fullfile(pwd, 'util'));  % 将当前目录下的 "util" 文件夹添加到搜索路径中。
run vlfeat/toolbox/vl_setup  % 运行 VLFeat 工具箱中的设置脚本，完成 VLFeat 的初始化（如加载 MEX 文件等）
logInfo('done.');  % 输出日志信息 “done.”，表示初始化或路径设置完成
