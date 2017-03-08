function [problems, notFound] = findProblems(lookInto, problemsFile)
%% FindProblems
% Call UNIX's find command to locate the problems listed in "problemsFile"
% under "lookInto". THIS FUNCTIONS ASSUMES THAT THE FILES ARE IN .NL
% FORMAT! It won't find any other file type!
% Inputs:
%   - lookInto: path under which to check for the .nl problems
%   - problemsFile: file containing the names of the problems
% Outputs:
%   - problems: cell array holding the full path to the .nl problems that
%   were found
%   - notFound: cell array of the problems that were not found

%% Opening the file containing the problems' names
fid = fopen(problemsFile, 'r');
if fid == -1
    error('File cannot be opened');
end

% Read file, check for existence and store in cell array
problems = {};
notFound = {};
while true
    % Parsing the file line by line
    tline = fgetl(fid);
    
    % Looking for the end of the file
    if ~ischar(tline)
        % Quit
        break
    end
    
    % Using UNIX find command to locate the file, case insensitive search
    [status, result] = system(sprintf('find %s -iname "*%s.nl"', ...
        lookInto, tline));
    
    if status ~= 0
        % Check if the shell call returns an error
        error('find caused an error, see the result variable');
    elseif strcmp(result, '')
        % Check if the file wasn't found
        notFound{end + 1} = tline;
    else
        problems{end + 1} = result;
    end
    
end

% Closing the file
fclose(fid);
end