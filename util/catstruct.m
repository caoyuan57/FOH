function A = catstruct(varargin)
% CATSTRUCT   Concatenate or merge structures with different fieldnames
%   X = CATSTRUCT(S1,S2,S3,...) merges the structures S1, S2, S3 ...
%   into one new structure X. X contains all fields present in the various
%   structures. An example:
%
%     A.name = 'Me' ;
%     B.income = 99999 ;
%     X = catstruct(A,B) 
%     % -> X.name = 'Me' ;
%     %    X.income = 99999 ;
%
%   If a fieldname is not unique among structures (i.e., a fieldname is
%   present in more than one structure), only the value from the last
%   structure with this field is used. In this case, the fields are 
%   alphabetically sorted. A warning is issued as well. An axample:
%
%     S1.name = 'Me' ;
%     S2.age  = 20 ; S3.age  = 30 ; S4.age  = 40 ;
%     S5.honest = false ;
%     Y = catstruct(S1,S2,S3,S4,S5) % use value from S4
%
%   The inputs can be array of structures. All structures should have the
%   same size. An example:
%
%     C(1).bb = 1 ; C(2).bb = 2 ;
%     D(1).aa = 3 ; D(2).aa = 4 ;
%     CD = catstruct(C,D) % CD is a 1x2 structure array with fields bb and aa
%
%   The last input can be the string 'sorted'. In this case,
%   CATSTRUCT(S1,S2, ..., 'sorted') will sort the fieldnames alphabetically. 
%   To sort the fieldnames of a structure A, you could use
%   CATSTRUCT(A,'sorted') but I recommend ORDERFIELDS for doing that.
%
%   When there is nothing to concatenate, the result will be an empty
%   struct (0x0 struct array with no fields).
%
%   NOTE: To concatenate similar arrays of structs, you can use simple
%   concatenation: 
%     A = dir('*.mat') ; B = dir('*.m') ; C = [A ; B] ;

%   NOTE: This function relies on unique. Matlab changed the behavior of
%   its set functions since 2013a, so this might cause some backward
%   compatibility issues when dulpicated fieldnames are found.
%
%   See also CAT, STRUCT, FIELDNAMES, STRUCT2CELL, ORDERFIELDS

% version 4.1 (feb 2015), tested in R2014a
% (c) Jos van der Geest
% email: jos@jasen.nl

% History
% Created in 2005
% Revisions
%   2.0 (sep 2007) removed bug when dealing with fields containing cell
%                  arrays (Thanks to Rene Willemink)
%   2.1 (sep 2008) added warning and error identifiers
%   2.2 (oct 2008) fixed error when dealing with empty structs (thanks to
%                  Lars Barring)
%   3.0 (mar 2013) fixed problem when the inputs were array of structures
%                  (thanks to Tor Inge Birkenes).
%                  Rephrased the help section as well.
%   4.0 (dec 2013) fixed problem with unique due to version differences in
%                  ML. Unique(...,'last') is no longer the deafult.
%                  (thanks to Isabel P)
%   4.1 (feb 2015) fixed warning with narginchk

narginchk(1,Inf) ;
N = nargin ;

%% 排序标志检测模块
% 判断最后一个参数是否是'sorted'标志
if ~isstruct(varargin{end}),
    if isequal(varargin{end},'sorted'),
        % 当存在'sorted'标志时调整参数计数
        narginchk(2,Inf) ;
        sorted = 1 ;
        N = N-1 ;
    else
        error('catstruct:InvalidArgument','Last argument should be a structure, or the string "sorted".') ;
    end
else
    sorted = 0 ;
end

%% 结构体预处理模块
sz0 = [] ; % 初始化尺寸记录变量

% 非空输入标记初始化
NonEmptyInputs = false(N,1) ;
NonEmptyInputsN = 0 ;

% 预分配存储空间
FN = cell(N,1) ;  % 存储各结构体字段名
VAL = cell(N,1) ; % 存储各结构体字段值

%% 结构体解析循环
for ii=1:N,
    X = varargin{ii} ;
    % 类型验证：确保输入为结构体
    if ~isstruct(X),
        error('catstruct:InvalidArgument',['Argument #' num2str(ii) ' is not a structure.']) ;
    end

    % 非空结构体处理
    if ~isempty(X),
        %% 尺寸一致性检查
        if ii > 1 && ~isempty(sz0)
            if ~isequal(size(X), sz0)
                error('catstruct:UnequalSizes','All structures should have the same size.') ;
            end
        else
            sz0 = size(X) ; % 记录首个非空结构体尺寸
        end

        %% 信息提取
        NonEmptyInputsN = NonEmptyInputsN + 1 ; % 非空计数器
        NonEmptyInputs(ii) = true ;             % 标记非空
        FN{ii} = fieldnames(X) ;    % 提取字段名
        VAL{ii} = struct2cell(X) ;  % 转换为单元格数组
    end
end

%% 特殊情况处理分支
if NonEmptyInputsN == 0
    % 全空输入处理：返回空结构体
    A = struct([]) ;
elseif NonEmptyInputsN == 1,
    % 单结构体处理：直接返回原结构体
    A = varargin{NonEmptyInputs} ;
    % 按需排序字段
    if sorted,
        A = orderfields(A) ;
    end

%% 常规合并流程
else
    % 垂直拼接所有字段信息
    FN = cat(1,FN{:}) ;
    VAL = cat(1,VAL{:}) ;

    % 维度压缩处理（兼容多维数组）
    FN = squeeze(FN) ;
    VAL = squeeze(VAL) ;

    %% 重复字段处理
    % 获取唯一字段名（保留最后出现的重复项）
    [UFN,ind] = unique(FN, 'last') ;
    % 注意：旧版Matlab可能需要改用unique(FN)

    %% 重复字段警告
    if numel(UFN) ~= numel(FN),
        warning('catstruct:DuplicatesFound','Fieldnames are not unique between structures.') ;
        sorted = 1 ; % 强制启用排序
    end

    %% 字段排序处理
    if sorted,
        VAL = VAL(ind,:) ; % 按索引重组数值
        FN = FN(ind,:) ;   % 按字母顺序排列
    end

    %% 最终结构体生成
    A = cell2struct(VAL, FN); % 转换为结构体
    A = reshape(A, sz0) ;    % 恢复原始数组维度
end



