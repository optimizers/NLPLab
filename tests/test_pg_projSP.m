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
HEADER = {'Solver', 'optTol', '#iter', '#f', '#g', '#H', '|Pg|', 'RT'};
HEADER_LATEX = {'Solver & ', 'optTol & ', '#iter & ', '\#f & ', ...
    '\#g & ', '\#H & ', '|Pg| & ', 'RT \\'};
HEADER_FORMAT = ['%25s', repmat('%10s', 1, 7), '\n'];
BODY_FORMAT = '%25s %10.1e %10d %10d %10d %10d %10.1e %10.1e\n';
BODY_LATEX = ['%25s %3s %10.1e %3s %10d %3s %10d %3s %10d %3s %10d %3s', ...
    ' %10.1e %3s %10.1e %5s \n'];
outInfo = {};
outLatex = {};

%% Solve
MAX_EVAL = 1e5;
MAX_ITER = 1e5;

import model.ProjModel;

import solvers.TmpSolver;
mcOpts.maxIter = MAX_ITER;
mcOpts.maxEval = MAX_EVAL;

import solvers.PnbSolver;
pnbOpts.maxIter = MAX_ITER;
pnbOpts.maxEval = MAX_EVAL;
pnbOpts.exactLS = false;

import solvers.BbSolver;
bbOpts.maxIter = MAX_ITER;
bbOpts.maxEval = MAX_EVAL;

import solvers.SpgSolver;
spgOpts.maxIter = MAX_ITER;
spgOpts.maxEval = MAX_EVAL;

import solvers.PqnSolver;
pqnOpts.maxIter = MAX_ITER;
pqnOpts.maxEval = MAX_EVAL;
pqnOpts.hess = 'exact';

import solvers.LbfgsbSolver;
lbfgsbOpts.maxIter = MAX_ITER;
lbfgsbOpts.maxEval = MAX_EVAL;

import solvers.PnSolver;
pnOpts.maxIter = MAX_ITER;
pnOpts.maxEval = MAX_EVAL;

data = struct;
projModel = ProjModel(prec, crit.J{2}.GeoS);
x0 = -10 * ones(projModel.objSize, 1);

for tol = 10 .^ (-1: -1: -10)
    fprintf('\n%s%.1e%s\n', [repmat('-', 1, 20), ' TOL = '], tol, ...
        [' ', repmat('-', 1, 20)]);
    mcOpts.aOptTol = tol;
    pnbOpts.optTol = tol;
    bbOpts.aOptTol = tol;
    spgOpts.aOptTol = tol;
    pqnOpts.aOptTol = tol;
    lbfgsbOpts.aOptTol = tol;
    pnOpts.optTol = tol;
    
    %% TMP
    % LSQR (Spot)
    projModel = ProjModel(prec, crit.J{2}.GeoS);
    projModel.setPointToProject(-x0);
    mcOpts.method = 'lsqr';
    solver = TmpSolver(projModel, mcOpts);
    solver.solve();
    
    name = class(solver);
    name = name(strfind(name, '.') + 1 : strfind(name, 'Solver') - 1);
    name = [name, mcOpts.method];
    [info, latex] = printInfo(BODY_FORMAT, BODY_LATEX, name, tol, ...
        solver);
    
    outInfo{end + 1} = info;
    outLatex{end + 1} = latex;
    
    try
        data.(name) = [data.(name); [-log10(tol), solver.iter, ...
            solver.nObjFunc, solver.nGrad, solver.nHess, solver.pgNorm, ...
            solver.solveTime]];
    catch
        data.(name) = [-log10(tol), solver.iter, solver.nObjFunc, ...
            solver.nGrad, solver.nHess, solver.pgNorm, solver.solveTime];
    end
    
    % LSMR (Spot)
    projModel = ProjModel(prec, crit.J{2}.GeoS);
    projModel.setPointToProject(-x0);
    mcOpts.method = 'lsmr';
    solver = TmpSolver(projModel, mcOpts);
    solver.solve();
    
    name = class(solver);
    name = name(strfind(name, '.') + 1 : strfind(name, 'Solver') - 1);
    name = [name, mcOpts.method];
    [info, latex] = printInfo(BODY_FORMAT, BODY_LATEX, name, tol, ...
        solver);
    
    outInfo{end + 1} = info;
    outLatex{end + 1} = latex;
    
    try
        data.(name) = [data.(name); [-log10(tol), solver.iter, ...
            solver.nObjFunc, solver.nGrad, solver.nHess, solver.pgNorm, ...
            solver.solveTime]];
    catch
        data.(name) = [-log10(tol), solver.iter, solver.nObjFunc, ...
            solver.nGrad, solver.nHess, solver.pgNorm, solver.solveTime];
    end
    
    % PCG
    projModel = ProjModel(prec, crit.J{2}.GeoS);
    projModel.setPointToProject(-x0);
    mcOpts.method = 'pcg';
    solver = TmpSolver(projModel, mcOpts);
    solver.solve();
    
    name = class(solver);
    name = name(strfind(name, '.') + 1 : strfind(name, 'Solver') - 1);
    name = [name, mcOpts.method];
    [info, latex] = printInfo(BODY_FORMAT, BODY_LATEX, name, tol, ...
        solver);
    
    outInfo{end + 1} = info;
    outLatex{end + 1} = latex;
    
    try
        data.(name) = [data.(name); [-log10(tol), solver.iter, ...
            solver.nObjFunc, solver.nGrad, solver.nHess, solver.pgNorm, ...
            solver.solveTime]];
    catch
        data.(name) = [-log10(tol), solver.iter, solver.nObjFunc, ...
            solver.nGrad, solver.nHess, solver.pgNorm, solver.solveTime];
    end
    
    % MINRES (Spot)
    projModel = ProjModel(prec, crit.J{2}.GeoS);
    projModel.setPointToProject(-x0);
    mcOpts.method = 'minres';
    solver = TmpSolver(projModel, mcOpts);
    solver.solve();
    
    name = class(solver);
    name = name(strfind(name, '.') + 1 : strfind(name, 'Solver') - 1);
    name = [name, mcOpts.method];
    [info, latex] = printInfo(BODY_FORMAT, BODY_LATEX, name, tol, ...
        solver);
    
    outInfo{end + 1} = info;
    outLatex{end + 1} = latex;
    
    try
        data.(name) = [data.(name); [-log10(tol), solver.iter, ...
            solver.nObjFunc, solver.nGrad, solver.nHess, solver.pgNorm, ...
            solver.solveTime]];
    catch
        data.(name) = [-log10(tol), solver.iter, solver.nObjFunc, ...
            solver.nGrad, solver.nHess, solver.pgNorm, solver.solveTime];
    end
    
    %% Pnb
    projModel = ProjModel(prec, crit.J{2}.GeoS);
    projModel.setPointToProject(-x0);
    solver = PnbSolver(projModel, pnbOpts);
    solver.solve();
    
    name = class(solver);
    name = name(strfind(name, '.') + 1 : strfind(name, 'Solver') - 1);
    [info, latex] = printInfo(BODY_FORMAT, BODY_LATEX, name, tol, ...
        solver);
    
    outInfo{end + 1} = info;
    outLatex{end + 1} = latex;
    
    try
        data.(name) = [data.(name); [-log10(tol), solver.iter, ...
            solver.nObjFunc, solver.nGrad, solver.nHess, solver.pgNorm, ...
            solver.solveTime]];
    catch
        data.(name) = [-log10(tol), solver.iter, solver.nObjFunc, ...
            solver.nGrad, solver.nHess, solver.pgNorm, solver.solveTime];
    end
    
    %% Bb
    projModel = ProjModel(prec, crit.J{2}.GeoS);
    projModel.setPointToProject(-x0);
    solver = BbSolver(projModel, bbOpts);
    solver.solve();
    
    name = class(solver);
    name = name(strfind(name, '.') + 1 : strfind(name, 'Solver') - 1);
    [info, latex] = printInfo(BODY_FORMAT, BODY_LATEX, name, tol, ...
        solver);
    
    outInfo{end + 1} = info;
    outLatex{end + 1} = latex;
    
    try
        data.(name) = [data.(name); [-log10(tol), solver.iter, ...
            solver.nObjFunc, solver.nGrad, solver.nHess, solver.pgNorm, ...
            solver.solveTime]];
    catch
        data.(name) = [-log10(tol), solver.iter, solver.nObjFunc, ...
            solver.nGrad, solver.nHess, solver.pgNorm, solver.solveTime];
    end
    
    %% Spg
    projModel = ProjModel(prec, crit.J{2}.GeoS);
    projModel.setPointToProject(-x0);
    solver = SpgSolver(projModel, spgOpts);
    solver.solve();
    
    name = class(solver);
    name = name(strfind(name, '.') + 1 : strfind(name, 'Solver') - 1);
    [info, latex] = printInfo(BODY_FORMAT, BODY_LATEX, name, tol, ...
        solver);
    
    outInfo{end + 1} = info;
    outLatex{end + 1} = latex;
    
    try
        data.(name) = [data.(name); [-log10(tol), solver.iter, ...
            solver.nObjFunc, solver.nGrad, solver.nHess, solver.pgNorm, ...
            solver.solveTime]];
    catch
        data.(name) = [-log10(tol), solver.iter, solver.nObjFunc, ...
            solver.nGrad, solver.nHess, solver.pgNorm, solver.solveTime];
    end
    
    %% Pqn
    projModel = ProjModel(prec, crit.J{2}.GeoS);
    projModel.setPointToProject(-x0);
    solver = PqnSolver(projModel, pqnOpts);
    solver.solve();
    
    name = class(solver);
    name = name(strfind(name, '.') + 1 : strfind(name, 'Solver') - 1);
    [info, latex] = printInfo(BODY_FORMAT, BODY_LATEX, name, tol, ...
        solver);
    
    outInfo{end + 1} = info;
    outLatex{end + 1} = latex;
    
    try
        data.(name) = [data.(name); [-log10(tol), solver.iter, ...
            solver.nObjFunc, solver.nGrad, solver.nHess, solver.pgNorm, ...
            solver.solveTime]];
    catch
        data.(name) = [-log10(tol), solver.iter, solver.nObjFunc, ...
            solver.nGrad, solver.nHess, solver.pgNorm, solver.solveTime];
    end
    
    %% L-BFGS-B
    projModel = ProjModel(prec, crit.J{2}.GeoS);
    projModel.setPointToProject(-x0);
    solver = LbfgsbSolver(projModel, lbfgsbOpts);
    solver.solve();
    
    name = class(solver);
    name = name(strfind(name, '.') + 1 : strfind(name, 'Solver') - 1);
    [info, latex] = printInfo(BODY_FORMAT, BODY_LATEX, name, tol, ...
        solver);
    
    outInfo{end + 1} = info;
    outLatex{end + 1} = latex;
    
    try
        data.(name) = [data.(name); [-log10(tol), solver.iter, ...
            solver.nObjFunc, solver.nGrad, solver.nHess, solver.pgNorm, ...
            solver.solveTime]];
    catch
        data.(name) = [-log10(tol), solver.iter, solver.nObjFunc, ...
            solver.nGrad, solver.nHess, solver.pgNorm, solver.solveTime];
    end
end

%% Printing
fid = fopen(['data/latex-output-fact', num2str(FACTOR), 'txt'], 'w');
save(['data/data_factor', num2str(FACTOR), '.mat'], 'data');
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
HEADER{2} = '-log(tol)';
plotStructData(data, HEADER(2:end), FACTOR);