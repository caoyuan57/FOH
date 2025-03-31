function path = record_diary(expdir, record)
% RECORD_DIARY 日志文件路径生成与管理函数
% 输入参数:
%   expdir - 实验目录路径 (字符串)
%   record - 控制标志 (逻辑值)
%            true : 创建新日志文件并开始记录
%            false: 返回最近日志文件路径
% 输出参数:
%   path   - 生成的日志文件路径 (字符串)

% 定义路径生成匿名函数 (格式: 实验目录/diary_序号.txt)
diary_path = @(i) sprintf('%s/diary_%03d.txt', expdir, i);

ind = 1; % 初始化文件序号

% 循环查找首个可用序号 (防文件覆盖机制)
while exist(diary_path(ind), 'file')
    ind = ind + 1; % 当文件存在时递增序号
end

if record
    % 记录模式: 使用新序号创建日志文件
    path = diary_path(ind);
    diary(path);    % 设置日志文件路径
    diary('on');    % 开启命令窗口记录
else
    % 非记录模式: 返回最近存在的日志文件路径
    path = diary_path(ind-1); % ind-1为最后有效序号
end
end