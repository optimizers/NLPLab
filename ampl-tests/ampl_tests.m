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
solverNames = {'Bcflash', 'Tmp2', 'Lbfgsb', 'Pnb', 'Ipopt'};
import solvers.BcflashSolver;
import solvers.Tmp2Solver;
import solvers.LbfgsbSolver;
import solvers.PnbSolver;
import solvers.IpoptSolver;
import model.AmplModel;

% Set up the solvers' general parameters
solverOpts = struct('aOptTol', 1e-13, 'aFeasTol', eps, ...
    'maxIter', 1e4, 'verbose', 1, 'maxRT', 10*60, 'maxEval', 1e4, ...
    'aCompTol', 1e-13);
ipoptOpts = solverOpts;

% Save everything in 'data' struct
data = struct;
data.infoHeader = {'pgNorm', 'solveTime', 'fx' 'iter', 'nObjFunc', ...
    'nGrad', 'nHess'};
data.solverNames = solverNames;
data.Bcflash = {};
data.Tmp2 = {};
data.Lbfgsb = {};
data.Pnb = {};
data.Ipopt = {};
data.failed = {}; % Keep track of failures
% Store performance profile data in a 3D matrix
np = length(problems);
ns = length(solverNames);
nm = length(data.infoHeader);
data.pMat = nan(np, ns, nm);

tempOptTol = solverOpts.aOptTol;
tempCompTol = solverOpts.aCompTol;

nProb = 1;
for problem = problems
    % For each problem
    fprintf('\n\n--- %s ---\n\n', problem{1});
    nSolv = 1;
    temp = nan(nm, ns);
    tempData = struct;
    for tempSolver = solverNames
        % For each solver
        try
            % Load the problem, sparse = true
            nlp = model.AmplModel(problem{1}, true);
            
            if strcmp(tempSolver{1}, 'Ipopt')
                
                gNorm = norm(nlp.gobj(nlp.x0));
                ipoptTol = max(gNorm * tempOptTol, eps);
                
                solverOpts.aOptTol = ipoptTol;
                solverOpts.aCompTol = ipoptTol;
            end
            
            % Calling the solver
            solver = eval(['solvers.', tempSolver{1}, ...
                'Solver(nlp, solverOpts)']);
            solver.solve();
            
            if strcmp(tempSolver{1}, 'Lbfgsb')
                solver.nHess = 1;
            end
            
            solverOpts.aOptTol = tempOptTol;
            solverOpts.aCompTol = tempCompTol;
            
            % Updating the performance profile data matrix
            temp(:, nSolv) = [solver.pgNorm; solver.solveTime; ...
                solver.fx; solver.iter; solver.nObjFunc; solver.nGrad; ...
                solver.nHess];
            tempData.(tempSolver{1}) = {solver.x, solver.iStop};
        catch ME
            % Some problems return function evaluation failures
            warning('%s\n', ME.message);
            data.failed{end + 1} = {problem{1}, tempSolver{1}};
            data.pMat(nProb, :) = [];
            break
        end
        nSolv = nSolv + 1;
    end
    data.pMat(nProb, 1 : ns, 1 : nm) = temp'; % (nProb, :, :)
    % Storing other stuff
    for tempSolver = solverNames
        data.(tempSolver{1}){end + 1} = tempData.(tempSolver{1});
    end
    nProb = nProb + 1;
end

[~, pname, ~] = fileparts(problemsFile);
save([pname, '-run-data-new'], 'data');

%% Build the performance profiles
import utils.perf;
perfOpts = struct('display', true, 'saveFolder', './ampl-runs-new/', ...
    'prefix', pname, 'logPlot', true);
utils.perf(data.pMat, {data.solverNames, data.infoHeader}, perfOpts);