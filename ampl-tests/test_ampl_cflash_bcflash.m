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
import solvers.BcflashSolver;
import solvers.CflashSolver;
import model.AmplModel;
import model.LinIneqProjAmplModel;
import model.BoundProjAmplModel;

% Set up the solvers' general parameters
bcflashOpts = struct('aOptTol', 1e-10, 'aFeasTol', eps, ...
    'maxIter', 1e4, 'verbose', 1, 'maxRT', 5*60, 'maxEval', 1e4);
cflashOpts = struct('aOptTol', 1e-10, 'aFeasTol', eps, ...
    'maxIter', 1e4, 'verbose', 1, 'maxRT', 5*60, 'maxEval', 1e4, ...
    'maxProj', 1e6, 'eqTol', 1e-12, 'maxExtraIter', 1e6);

% Save everything in 'data' struct
data = struct;
data.infoHeader = {'pgNorm', 'solveTime', 'fx', 'iter', 'nObjFunc', ...
    'nGrad', 'nHess'};
data.solverNames = {'Bcflash', 'Cflash'};
np = length(problems);
nn = length(data.solverNames);
nd = length(data.infoHeader);
data.Bcflash = {};
data.Cflash = {};
data.xComp = [];
data.failed = {}; % Keep track of failures
% Store performance profile data in a 3D matrix
data.pMat = nan(np, nn, nd);

nProb = 1;
for problem = problems
    % For each problem
    fprintf('\n\n--- %s ---\n\n', problem{1});
    try
        % Load the problem, sparse = true
        nlp = model.AmplModel(problem{1}, true);
        % Call Bcflash
        bcflash = solvers.BcflashSolver(nlp, bcflashOpts);
        bcflash.solve();
        
        % Call Cflash
        nlp = model.BoundProjAmplModel(problem{1}, true);
        cflash = solvers.CflashSolver(nlp, cflashOpts);
        cflash.solve();
        
    catch ME
        % Some problems return function evaluation failures
        warning('%s\n', ME.message);
        data.failed{end + 1} = problem{1};
        continue
    end
    
    % If both solves are succesful, store data
    temp = [bcflash.pgNorm, bcflash.solveTime, bcflash.fx, ...
        bcflash.iter, bcflash.nObjFunc, bcflash.nGrad, bcflash.nHess];
    data.pMat(nProb, 1, 1 : nd) = temp;
    data.Bcflash{end + 1} = {bcflash.x, bcflash.EXIT_MSG{bcflash.iStop}};
    
    temp = [cflash.pgNorm, cflash.solveTime, cflash.fx, ...
        cflash.iter, cflash.nObjFunc, cflash.nGrad, cflash.nHess];
    data.pMat(nProb, 2, 1 : nd) = temp;
    data.Cflash{end + 1} = {cflash.x, cflash.EXIT_MSG{cflash.iStop}};
    
    data.xComp = [data.xComp; norm(bcflash.x - cflash.x)];
    
    nProb = nProb + 1;
end

[~, pname, ~] = fileparts(problemsFile);
save([pname, '-cflash-vs-bcflash-new'], 'data');

%% Build the performance profiles
import utils.perf;
perfOpts = struct('display', true, 'saveFolder', ...
    './cflash-vs-bcflash-new/', 'prefix', pname, 'logPlot', true);
utils.perf(data.pMat, {data.solverNames, data.infoHeader}, perfOpts);
