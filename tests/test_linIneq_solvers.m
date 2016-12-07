clc;
clear all;
close all;

%% Format
HEADER = {'Solver', '#iter', '#f', '#g', '#H', '|Pg|', ...
    '|x*-x|', 'RT'};
HEADER_FORMAT = ['%25s', repmat('%10s ', 1, 7), '\n'];
BODY_FORMAT = '%25s %10d %10d %10d %10d %10.1e %10.1e %10.1e\n';
outInfo = {};

%% Seting MATLAB's path
addpath('~/Masters/NLPlab');
addpath('~/Masters/logging4matlab/');
addpath('~/Masters/Spot');
addpath('~/Masters/optimization/lbfgsb');
addpath('~/Masters/optimization/box_project');

%% Building the model
% Quadratic objective function, upper and lower bounds
m = 20;
n = m;
Q = randi(30, m, n) - randi(30, m, n);
Q = Q' * Q;
c = -round(10 * rand(n, 1));
cL = -1 * ones(n, 1);
cU = 5 * ones(n, 1);
cU(1:3:end) = 1;
x0 = zeros(n, 1);
bL = -inf(n, 1);
bU = inf(n, 1);
A = randi(15, m, n) - randi(15, m, n);
% For the PQN & SPG solvers, we need a projection function
import model.LinIneqQpModel;
liQuadModel = LinIneqQpModel('', x0, A, cL, cU, c, Q);

%% MATLAB's quadprog is the reference solution
xRef = quadprog(Q, c, [A; -A], [cU; -cL]);

%% Solve using SPG (works only if prob is quadratic)
import solvers.SpgSolver;
solver = SpgSolver(liQuadModel, 'aOptTol', 1e-10, 'aFeasTol', 1e-15, ...
    'progTol', 1e-15, 'maxIter', 1e4);
solver = solver.solve();

nrmSol = norm(xRef - solver.x);
outInfo{end + 1} = sprintf(BODY_FORMAT, class(solver), solver.iter, ...
    solver.nObjFunc, solver.nGrad, solver.nHess, solver.pgNorm, nrmSol, ...
    solver.solveTime);

%% Solve using PQN
import solvers.PqnSolver;
solver = PqnSolver(liQuadModel, 'hess', 'exact', 'aOptTol', 1e-10, ...
    'progTol', 1e-15, 'aFeasTol', 1e-15, 'maxIter', 1e4);
solver = solver.solve();

nrmSol = norm(xRef - solver.x);
outInfo{end + 1} = sprintf(BODY_FORMAT, class(solver), solver.iter, ...
    solver.nObjFunc, solver.nGrad, solver.nHess, solver.pgNorm, nrmSol, ...
    solver.solveTime);

%% Solve using Cflash
import solvers.CflashSolver;
solver = CflashSolver(liQuadModel, 'aOptTol', 1e-10, 'aFeasTol', 1e-15, ...
    'maxIter', 1e4);
solver = solver.solve();

nrmSol = norm(xRef - solver.x);
outInfo{end + 1} = sprintf(BODY_FORMAT, class(solver), solver.iter, ...
    solver.nObjFunc, solver.nGrad, solver.nHess, solver.pgNorm, nrmSol, ...
    solver.solveTime);

%% Printing
fprintf('\n\n\n');
fprintf(HEADER_FORMAT, HEADER{:});
for s = outInfo
    fprintf(s{1});
end