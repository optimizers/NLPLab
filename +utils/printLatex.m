function [outLatex] = printLatex(data)
%% Print data stored in struct to LaTeX format for Bcflash & IPOPT

[nProb, nSolv, nMetric] = size(data.pMat);
outLatex = cell(2 * nProb, 1); % each prob for two solvers
% < problem name | solver name | header | exit msg >
HDR_FRMT = [repmat('%s & ', 1, 2 + nMetric), '%s '];
BDY_FRMT = ['multirow{ 2}{*}{%s} & %s & %.2e & %.0e & %.2e & ', ...
    repmat('%d & ', 1, 4), '%s '];

fid = fopen('./output.txt', 'w+');
if fid == -1
    error('Can''t open file.');
end

% Add header to outLatex
outLatex{1} = sprintf(HDR_FRMT, 'Problem', 'Solver', ...
    data.infoHeader{:}, 'Exit Message');

% To output.txt
fprintf(fid, [outLatex{1}, '\\\\ \n']);
% To console
fprintf([outLatex{1}, '\\\\ \n']);

% Parse data
count = 2;
for ind = 1 : nProb
    
    % nMetric x nSolv matrix for the #ind problem
    tempMat = reshape(data.pMat(ind, :, :), nSolv, nMetric);
    
    % Extracting problem name from path
    [~, problemName, ~] = fileparts(data.problems{ind});
    
    % Converting corresponding line to string, splitting by space
    % IPOPT
    msg = data.Ipopt{ind}{2};
    outLatex{count} = sprintf(BDY_FRMT, problemName, 'IPOPT', ...
        tempMat(1, :), msg(1 : end - 2));
    fprintf(fid, ['\\', outLatex{count}, '\\\\ \n']); 
    fprintf(['\\', outLatex{count}, '\\\\ \n']);
    count = count + 1;
    
    % Bcflash
    msg = data.Bcflash{ind}{2};
    outLatex{count} = sprintf(BDY_FRMT, problemName, 'Bcflash', ...
        tempMat(2, :), msg(1 : end - 2));
    fprintf(fid, ['\\', outLatex{count}, '\\\\ \n']);
    fprintf(['\\', outLatex{count}, '\\\\ \n']);
    count = count + 1;
end
end

