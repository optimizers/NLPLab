%% Bcflash test file using AMPL problems
clc;
clear all;
close all;

%% Initializing the directories
global ROOTDIR
% Defining the root directory
ROOTDIR = fullfile(getenv('HOME'), 'Masters');
% setPaths is in recUtils
addpath(fullfile(ROOTDIR, 'recUtils'));
% Adding all the other repositories to the path
setPaths; % edit this function accordingly

%% Getting the problems
% Folder that might contain the .nl problem files
lookInto = '~/Masters/decoded_ampl_models';
% File containing the problems name
problemsFile = './bounds-new.lst';

import utils.findProblems;
[problems, notFound] = utils.findProblems(lookInto, problemsFile);

%% Solve the problems
solverNames = {'Bcflash', 'OldBcflash', 'Tmp2', 'Lbfgsb', 'Pnb'};
import solvers.BcflashSolver;
import solvers.OldBcflashSolver;
import solvers.Tmp2Solver;
import solvers.LbfgsbSolver;
import solvers.PnbSolver;
import model.AmplModel;

% Set up the solvers' general parameters
solverOpts = struct('aOptTol', 1e-13, 'aFeasTol', eps, ...
    'maxIter', 1e4, 'verbose', 1, 'maxRT', 10*60, 'maxEval', 1e4);

% Save everything in 'data' struct
data = struct;
data.infoHeader = {'pgNorm', 'solveTime', 'iter', 'nObjFunc', 'nGrad', ...
    'nHess'};
data.solverNames = solverNames;
data.Bcflash = {};
data.OldBcflash = {};
data.Tmp2 = {};
data.Lbfgsb = {};
data.Pnb = {};
data.failed = {}; % Keep track of failures
% Store performance profile data in a 3D matrix
data.pMat = nan(length(problems), length(solverNames), 6);

nProb = 1;
for problem = problems
    % For each problem
    fprintf('\n\n--- %s ---\n\n', problem{1});
    nSolv = 1;
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
            temp = [solver.pgNorm, solver.solveTime, ...
                solver.iter, solver.nObjFunc, solver.nGrad, solver.nHess];
            data.pMat(nProb, nSolv, 1:6) = temp;
            
            % Storing other stuff
            data.(tempSolver{1}){end + 1} = {solver.x, solver.solved};
        catch ME
            % Some problems return function evaluation failures
            warning('%s\n', ME.message);
            data.failed{end + 1} = problem{1};
        end
        nSolv = nSolv + 1;
    end
    nProb = nProb + 1;
end

[~, pname, ~] = fileparts(problemsFile);
save([pname, '-run-data'], 'data');

%% Build the performance profiles
import utils.perf;
perfOpts = struct('display', true, 'saveFolder', './ampl-runs/', ...
    'prefix', pname, 'logPlot', true);
utils.perf(data.pMat, {data.solverNames, data.infoHeader}, perfOpts);