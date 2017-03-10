%% Bcflash test file using AMPL problems
clc;
clear all;
close all;

%% Init - edit accordingly
ROOTDIR = '~/Masters';
% Making sure NlpLab is on the path
addpath(fullfile(ROOTDIR, 'nlplab'));
% Making sure Spot is on the path
addpath(fullfile(ROOTDIR, 'Spot'));
% Making sure logging4matlab is on the path
addpath(fullfile(ROOTDIR, 'logging4matlab'));
% Make sure the ampl-interface is on the path
addpath(ROOTDIR);

%% Getting the problems
% Folder that might contain the .nl problem files
lookInto = '~/Masters/decoded_ampl_models';
% File containing the problems name
problemsFile = './bounds-new.lst';

import utils.findProblems;
[problems, notFound] = utils.findProblems(lookInto, problemsFile);

%% Solve the problems
solverNames = {'Bcflash', 'OldBcflash', 'Tmp2'};
import solvers.BcflashSolver;
import solvers.OldBcflashSolver;
import solvers.Tmp2Solver;
import model.AmplModel;

% Set up the solvers' general parameters
solverOpts = struct('aOptTol', 1e-10, 'aFeasTol', 1e-15, ...
    'maxIter', 1e4, 'verbose', 2, 'useBb', false, 'backtracking', ...
    false, 'maxRT', 10*60, 'maxEval', 1e4);

% Save everything in 'data' struct
data = struct;
data.infoHeader = {'pgNorm', 'fx', 'solveTime', 'iter', ...
    'iterCg', 'nObjFunc', 'nGrad', 'nHess'};
data.solverNames = solverNames;
data.Bcflash = {};
data.Old = {};
data.Tmp2 = {};
data.failed = {}; % Keep track of failures
% Store performance profile data in a 3D matrix
data.pMat = zeros(length(problems), length(solverNames), 7);

nProb = 1;
nSolv = 1;
for problem = problems
    % For each problem
    fprintf('\n\n--- %s ---\n\n', problem{1});
    [~, pname, ~] = fileparts(problem{1});
    for tempSolver = solverNames
        % For each solver
        try
            % Load the problem, sparse = true
            nlp = model.AmplModel(problem{1}, true);
            % Calling the solver
            solver = eval(['solvers.', tempSolver{1}, ...
                'Solver(nlp, solverOpts)']);
            solver.solve();
            
            % Updating the performance profile data matrix
            temp = [solver.pgNorm, solver.fx, solver.solveTime, ...
                solver.iter, solver.nObjFunc, solver.nGrad, solver.nHess];
            data.pMat(nProb, nSolv, 1:7) = temp;
            
            % Storing other stuff
            data.(tempSolver{1}){end + 1} = {solver.x, solver.solved};
        catch ME
            % Some problems return function evaluation failures
            warning('%s\n', ME.message);
            data.failed{end + 1} = problem{1};
            keyboard;
        end
        nSolv = nSolv + 1;
    end
    nProb = nProb + 1;
end

[~, pname, ~] = fileparts(problemsFile);
save([pname, '-run-data'], 'data');
fclose(fid);

%% Build the performance profiles
import utils.perf;
perfOpts = struct('display', true, 'saveFolder', './ampl-runs/', ...
    'prefix', pname, 'logPlot', true);
utils.perf(data.pMat, {data.solverNames, data.infoHeader}, perfOpts);