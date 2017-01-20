%% Testing MinConf_TMP to solve the projection sub-problem
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
% Folder where the phantom is saved
DATADIR = fullfile(ROOTDIR, 'TestData');
% Folder where the projection matrices are saved
MPDIR = fullfile(DATADIR, 'projMatrices');

%% Creating the phantom data and the sinogram
% --- genData parameters ---
% Resolution reduction factor compared to original (>= 1)
FACTOR = 1;
% --- genCrit parameters ---
critPars.lambda = 1e-00;
critPars.delta = 5e-03;
critPars.penalType = 'PenalGradObj_L2';
critPars.coordType = 'cyl';
critPars.imagPenal = false;
% Chosing set of parameters to use for the phantom
[fant, sino] = genData(DATADIR, FACTOR, ['phant_', num2str(FACTOR)]);

%% Creating the cartesian coordinates objective function object (Critere)
[crit, critPars] = genCrit(fant, sino, DATADIR, MPDIR, critPars);

%% Build the preconditionner
prec = Precond_Factory.create('DiagF', 'crit', crit);

%% Format
HEADER = {'Solver', 'Range', '#iter', '#f', '#g', '#H', '|Pg|', 'RT'};
HEADER_LATEX = {'Solver & ', 'Range & ', '#iter & ', '\#f & ', ...
    '\#g & ', '\#H & ', '|Pg| & ', 'RT \\'};
HEADER_FORMAT = ['%25s', repmat('%10s', 1, 7), '\n'];
BODY_FORMAT = '%25s %10d %10d %10d %10d %10d %10.1e %10.1e\n';
BODY_LATEX = ['%25s %3s %10d %3s %10d %3s %10d %3s %10d %3s %10d %3s', ...
    ' %10.1e %3s %10.1e %5s \n'];
outInfo = {};
outLatex = {};

%% Solve
TOL = 1e-10;
MAX_ITER = 1e5;

import model.ProjModel;

import solvers.TmpSolver;
mcOpts.verbose = 2;
mcOpts.aOptTol = TOL;
mcOpts.maxIter = MAX_ITER;
mcOpts.maxEval = 1e5;
mcOpts.suffDec = 1e-4;
mcOpts.maxIterLS = 50;
mcOpts.corrections = 7;
mcOpts.damped = 0;

import solvers.PnbSolver;
pnbOpts.optTol = TOL;
pnbOpts.maxIter = MAX_ITER;
pnbOpts.verbose = 2;
pnbOpts.exactLS = false;

import solvers.BbSolver;
bbOpts.aOptTol = TOL;
bbOpts.maxIter = MAX_ITER;
bbOpts.maxEval = 1e5;

import solvers.SpgSolver;
spgOpts.aOptTol = TOL;
spgOpts.maxIter = MAX_ITER;
spgOpts.maxEval = 1e5;

import solvers.PqnSolver;
pqnOpts.aOptTol = TOL;
pqnOpts.maxIter = MAX_ITER;
pqnOpts.maxEval = 1e5;
pqnOpts.hess = 'exact';

data = struct;

for r = [75, 50, 25, 0]
    fprintf('\n---------------------- r = %d ----------------------\n', ...
        r);
    RANGE = r;
    
    projModel = ProjModel(prec, crit.J{2}.GeoS);
    
    x0 = 10*ones(projModel.objSize, 1);
    range = round(RANGE/100 * projModel.objSize);
    x0(1 : range) = -10*ones(range, 1); % Fill with -10 according to RANGE
    
    diary(['data/test_projection_data_', num2str(FACTOR), '_', ...
        num2str(RANGE), '.txt']);
    
    %% TMP
    % LSQR (Spot)
    projModel.setPointToProject(x0);
    mcOpts.method = 'lsqr';
    solver = TmpSolver(projModel, mcOpts);
    solver.solve();
    
    name = class(solver);
    name = name(strfind(name, '.') + 1 : end);
    name = [name, mcOpts.method];
    [info, latex] = printInfo(BODY_FORMAT, BODY_LATEX, name, RANGE, ...
        solver);
    
    outInfo{end + 1} = info;
    outLatex{end + 1} = latex;
    if RANGE == 75
        data.(name) = [RANGE, solver.iter, solver.nObjFunc, ...
            solver.nGrad, solver.nHess, solver.pgNorm, solver.solveTime];
    else
        data.(name) = [data.(name); [RANGE, solver.iter, ...
            solver.nObjFunc, solver.nGrad, solver.nHess, solver.pgNorm, ...
            solver.solveTime]];
    end
    
    % LSQR (Spot)
    projModel = ProjModel(prec, crit.J{2}.GeoS);
    projModel.setPointToProject(x0);
    mcOpts.method = 'lsmr';
    solver = TmpSolver(projModel, mcOpts);
    solver.solve();
    
    name = class(solver);
    name = name(strfind(name, '.') + 1 : end);
    name = [name, mcOpts.method];
    [info, latex] = printInfo(BODY_FORMAT, BODY_LATEX, name, RANGE, ...
        solver);
    
    outInfo{end + 1} = info;
    outLatex{end + 1} = latex;
    
    if RANGE == 75
        data.(name) = [RANGE, solver.iter, solver.nObjFunc, ...
            solver.nGrad, solver.nHess, solver.pgNorm, solver.solveTime];
    else
        data.(name) = [data.(name); [RANGE, solver.iter, ...
            solver.nObjFunc, solver.nGrad, solver.nHess, solver.pgNorm, ...
            solver.solveTime]];
    end
    
    % PCG
    projModel = ProjModel(prec, crit.J{2}.GeoS);
    projModel.setPointToProject(x0);
    mcOpts.method = 'pcg';
    solver = TmpSolver(projModel, mcOpts);
    solver.solve();
    
    name = class(solver);
    name = name(strfind(name, '.') + 1 : end);
    name = [name, mcOpts.method];
    [info, latex] = printInfo(BODY_FORMAT, BODY_LATEX, name, RANGE, ...
        solver);
    
    outInfo{end + 1} = info;
    outLatex{end + 1} = latex;
    
    if RANGE == 75
        data.(name) = [RANGE, solver.iter, solver.nObjFunc, ...
            solver.nGrad, solver.nHess, solver.pgNorm, solver.solveTime];
    else
        data.(name) = [data.(name); [RANGE, solver.iter, ...
            solver.nObjFunc, solver.nGrad, solver.nHess, solver.pgNorm, ...
            solver.solveTime]];
    end
    
    % MINRES (Spot)
    projModel = ProjModel(prec, crit.J{2}.GeoS);
    projModel.setPointToProject(x0);
    mcOpts.method = 'minres';
    solver = TmpSolver(projModel, mcOpts);
    solver.solve();
    
    name = class(solver);
    name = name(strfind(name, '.') + 1 : end);
    name = [name, mcOpts.method];
    [info, latex] = printInfo(BODY_FORMAT, BODY_LATEX, name, RANGE, ...
        solver);
    
    outInfo{end + 1} = info;
    outLatex{end + 1} = latex;
    
    if RANGE == 75
        data.(name) = [RANGE, solver.iter, solver.nObjFunc, ...
            solver.nGrad, solver.nHess, solver.pgNorm, solver.solveTime];
    else
        data.(name) = [data.(name); [RANGE, solver.iter, ...
            solver.nObjFunc, solver.nGrad, solver.nHess, solver.pgNorm, ...
            solver.solveTime]];
    end
    
    %% Pnb
    projModel = ProjModel(prec, crit.J{2}.GeoS);
    projModel.setPointToProject(x0);
    solver = PnbSolver(projModel, pnbOpts);
    solver.solve();
    
    name = class(solver);
    name = name(strfind(name, '.') + 1 : end);
    [info, latex] = printInfo(BODY_FORMAT, BODY_LATEX, name, RANGE, ...
        solver);
    
    outInfo{end + 1} = info;
    outLatex{end + 1} = latex;
    
    if RANGE == 75
        data.(name) = [RANGE, solver.iter, solver.nObjFunc, ...
            solver.nGrad, solver.nHess, solver.pgNorm, solver.solveTime];
    else
        data.(name) = [data.(name); [RANGE, solver.iter, ...
            solver.nObjFunc, solver.nGrad, solver.nHess, solver.pgNorm, ...
            solver.solveTime]];
    end
    
    %% Bb
    projModel = ProjModel(prec, crit.J{2}.GeoS);
    projModel.setPointToProject(x0);
    solver = BbSolver(projModel, bbOpts);
    solver.solve();
    
    name = class(solver);
    name = name(strfind(name, '.') + 1 : end);
    [info, latex] = printInfo(BODY_FORMAT, BODY_LATEX, name, RANGE, ...
        solver);
    
    outInfo{end + 1} = info;
    outLatex{end + 1} = latex;
    
    if RANGE == 75
        data.(name) = [RANGE, solver.iter, solver.nObjFunc, ...
            solver.nGrad, solver.nHess, solver.pgNorm, solver.solveTime];
    else
        data.(name) = [data.(name); [RANGE, solver.iter, ...
            solver.nObjFunc, solver.nGrad, solver.nHess, solver.pgNorm, ...
            solver.solveTime]];
    end
    
    %% Spg
    projModel = ProjModel(prec, crit.J{2}.GeoS);
    projModel.setPointToProject(x0);
    solver = SpgSolver(projModel, spgOpts);
    solver.solve();
    
    name = class(solver);
    name = name(strfind(name, '.') + 1 : end);
    [info, latex] = printInfo(BODY_FORMAT, BODY_LATEX, name, RANGE, ...
        solver);
    
    outInfo{end + 1} = info;
    outLatex{end + 1} = latex;
    
    if RANGE == 75
        data.(name) = [RANGE, solver.iter, solver.nObjFunc, ...
            solver.nGrad, solver.nHess, solver.pgNorm, solver.solveTime];
    else
        data.(name) = [data.(name); [RANGE, solver.iter, ...
            solver.nObjFunc, solver.nGrad, solver.nHess, solver.pgNorm, ...
            solver.solveTime]];
    end
    
    %% Pqn
    projModel = ProjModel(prec, crit.J{2}.GeoS);
    projModel.setPointToProject(x0);
    solver = PqnSolver(projModel, pqnOpts);
    solver.solve();
    
    name = class(solver);
    name = name(strfind(name, '.') + 1 : end);
    [info, latex] = printInfo(BODY_FORMAT, BODY_LATEX, name, RANGE, ...
        solver);
    
    outInfo{end + 1} = info;
    outLatex{end + 1} = latex;
    
    if RANGE == 75
        data.(name) = [RANGE, solver.iter, solver.nObjFunc, ...
            solver.nGrad, solver.nHess, solver.pgNorm, solver.solveTime];
    else
        data.(name) = [data.(name); [RANGE, solver.iter, ...
            solver.nObjFunc, solver.nGrad, solver.nHess, solver.pgNorm, ...
            solver.solveTime]];
    end
    
    %% Switch logging off
    diary('off');
end

%% Printing
fid = fopen('data/latex-output.txt', 'w');
fprintf('\n\n\n');
fprintf(HEADER_FORMAT, HEADER{:});
fprintf(fid, HEADER_FORMAT, HEADER_LATEX{:});
for s = outInfo
    fprintf(s{1});
end
for s = outLatex
    fprintf(fid, s{1});
end
fclose(fid);

%% Plot struct data
plotStructData(data, HEADER(2:end));