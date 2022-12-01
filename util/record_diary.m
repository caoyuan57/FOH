function path = record_diary(expdir, record)
diary_path = @(i) sprintf('%s/diary_%03d.txt', expdir, i);
ind = 1;
while exist(diary_path(ind), 'file')
    ind = ind + 1;
end
if record
    path = diary_path(ind);
    diary(path);
    diary('on');
else
    path = diary_path(ind-1);
end
end
