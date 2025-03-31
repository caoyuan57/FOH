function [recall, precision, info] = vl_pr(labels, scores, varargin)
%VL_PR   精度-召回曲线（Precision-Recall Curve）。
%   [RECALL, PRECISION] = VL_PR(LABELS, SCORES) 计算精度-召回曲线（PR曲线）。LABELS 是 ground truth 标签，
%   大于零表示正样本，小于零表示负样本。SCORES 是样本的得分，得分越大应当对应正样本。
%
%   样本按得分降序排列，从排名1开始。PRECISION(K) 和 RECALL(K) 是当排名小于或等于 K-1 时，
%   样本被预测为正样本，其余为负样本的精度和召回率。
%   例如，PRECISION(3) 是前两名样本中正样本的百分比，PRECISION(1) 是当没有正样本时的精度，默认值为 1。

%   设置为零的标签表示需要在评估中忽略的样本。对于没有检索到的样本，SCORES 设置为 -INF。
%   如果有样本得分为 -INF，则 PR 曲线的最大召回率可能小于 1，除非使用 INCLUDEINF 选项（详见下文）。
%   NUMNEGATIVES 和 NUMPOSITIVES 可用于添加额外的伪样本，并将其得分设置为 -INF。

%   [RECALL, PRECISION, INFO] = VL_PR(...) 返回一个额外的结构体 INFO，其中包含以下字段：

%   info.auc::
%     PR 曲线下的面积（AUC）。如果 INTERPOLATE 选项设置为 FALSE，则使用梯形插值法计算 AUC。
%     如果 INTERPOLATE 选项设置为 TRUE，则曲线为分段常数，不进行其他近似处理，INFO.AUC 与 INFO.AP 相同。

%   info.ap::
%     平均精度（Average Precision, AP），计算时每次召回一个新的正样本时的精度的平均值。

%   info.ap_interp_11::
%     11 点插值的平均精度（平均精度曲线），这是 PASCAL VOC 挑战赛直到 2008 年的标准。

%   info.auc_pa08::
%     已弃用，与 INFO.AP_INTERP_11 相同。

%   VL_PR() 如果没有输出参数，将会绘制 PR 曲线。

%   VL_PR() 支持以下选项：

%   Interpolate:: false
%     如果设置为 true，使用插值的精度。插值的精度定义为从给定召回率及以后最大的精度。

%   NumPositives:: []
%   NumNegatives:: []
%     如果设置为一个数字，假设 LABELS 中包含此数量的正/负标签。附加的正负标签会追加到序列的末尾，得分为 -INF（未检索）。
%     这对大型检索系统有用，仅存储少数前几名结果以提高效率。

%   IncludeInf:: false
%     如果设置为 true，则包含得分为 -INF 的样本，最大召回率为 1，即使存在 -INF 得分的样本。

%   Stable:: false
%     如果设置为 true，则返回的 RECALL 和 PRECISION 将按照 LABELS 和 SCORES 中的顺序返回，而不是按得分降序排序。

%   NormalizePrior:: []
%     如果设置为一个标量，重新加权正负标签，使得正标签的比例等于指定值。

%   参见：VL_ROC(), VL_HELP()。

%   作者：Andrea Vedaldi

% 计算真正例（TP）和假正例（FP）的数量
[tp, fp, p, n, perm, varargin] = vl_tpfp(labels, scores, varargin{:}) ;
opts.stable = false ;   % 默认不使用稳定模式
opts.interpolate = false ;  % 默认不使用插值
opts.normalizePrior = [] ;  % 默认不进行正负标签的加权
opts = vl_argparse(opts,varargin) ;  % 解析传入的选项

% 计算精度和召回率
small = 1e-10 ;  % 防止除以零的极小值
recall = tp / max(p, small) ;  % 计算召回率
if isempty(opts.normalizePrior)
  precision = max(tp, small) ./ max(tp + fp, small) ;  % 计算精度
else
  % 使用 normalizePrior 重新加权精度
  a = opts.normalizePrior ;
  precision = max(tp * a/max(p, small), small) ./ ...
      max(tp * a/max(p, small) + fp * (1 - a)/max(n, small), small) ;
end

% 如果需要插值，进行插值处理
if opts.interpolate
  precision = fliplr(vl_cummax(fliplr(precision))) ;  % 反向计算累计最大值
end

% --------------------------------------------------------------------
%                                                      计算附加信息
% --------------------------------------------------------------------

if nargout > 2 || nargout == 0
  % 使用梯形插值法计算 PR 曲线下的面积（AUC）
  if ~opts.interpolate
    info.auc = 0.5 * sum((precision(1:end-1) + precision(2:end)) .* diff(recall)) ;
  end

  % 计算平均精度（AP）
  sel = find(diff(recall)) + 1 ;
  info.ap = sum(precision(sel)) / p ;
  if opts.interpolate
    info.auc = info.ap ;  % 如果插值，AUC 和 AP 相等
  end

  % 计算 11 点插值的平均精度
  info.ap_interp_11 = 0.0 ;
  for rc = linspace(0, 1, 11)
    pr = max([0, precision(recall >= rc)]) ;
    info.ap_interp_11 = info.ap_interp_11 + pr / 11 ;
  end

  % 兼容定义
  info.auc_pa08 = info.ap_interp_11 ;
end

% --------------------------------------------------------------------
%                                                                 绘图
% --------------------------------------------------------------------

if nargout == 0
  cla ; hold on ;
  plot(recall, precision, 'linewidth', 2) ;  % 绘制 PR 曲线
  if isempty(opts.normalizePrior)
    randomPrecision = p / (p + n) ;  % 计算随机精度
  else
    randomPrecision = opts.normalizePrior ;  % 使用指定的精度
  end
  spline([0 1], [1 1] * randomPrecision, 'r--', 'linewidth', 2) ;  % 绘制随机精度线
  axis square ; grid on ;
  xlim([0 1]) ; xlabel('recall') ;
  ylim([0 1]) ; ylabel('precision') ;
  title(sprintf('PR (AUC: %.2f%%, AP: %.2f%%, AP11: %.2f%%)', ...
                info.auc * 100, ...
                info.ap * 100, ...
                info.ap_interp_11 * 100)) ;  % 显示 AUC 和 AP 等信息
  if opts.interpolate
    legend('PR interp.', 'PR rand.', 'Location', 'SouthEast') ;
  else
    legend('PR', 'PR rand.', 'Location', 'SouthEast') ;
  end
  clear recall precision info ;  % 清除变量
end

% --------------------------------------------------------------------
%                                                       稳定输出
% --------------------------------------------------------------------

if opts.stable
  precision(1) = [] ;  % 移除第一项
  recall(1) = [] ;
  precision_ = precision ;
  recall_ = recall ;
  precision = NaN(size(precision)) ;
  recall = NaN(size(recall)) ;
  precision(perm) = precision_ ;  % 恢复原来的排序
  recall(perm) = recall_ ;
end

% --------------------------------------------------------------------
function h = spline(x, y, spec, varargin)
% 处理线条样式的辅助函数
prop = vl_linespec2prop(spec) ;
h = line(x, y, prop{:}, varargin{:}) ;
