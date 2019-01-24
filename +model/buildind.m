function ind = buildind(sz)
%BUILDIND  Build an index set from sizes stored in a structure.

% Find the total length
len = 0;
for i = fieldnames(sz)'
    f = i{:};
    len = len + sz.(f);
end

% Set the logical indicator for each field
ptr = 0;
for i = fieldnames(sz)'
    f = i{:};
    ind.(f) = false(len,1);
    ind.(f)(ptr + 1 : ptr + sz.(f)) = true;
    ptr = ptr + sz.(f);
end
