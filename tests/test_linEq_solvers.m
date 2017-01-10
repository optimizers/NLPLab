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
addpath('~/Masters/nlplab');
addpath('~/Masters/logging4matlab/');
addpath('~/Masters/Spot');

%% Building the model
% Quadratic objective function, linear equalities
m = 5e2;
n = m;
Q = diag(50 * ones(m - 1, 1), -1) + diag(100 * ones(m, 1), 0) + ...
    diag(50 * ones(m - 1, 1), 1);
c = -randi(1000, n, 1);
bL = -inf(n, 1);
bU = inf(n, 1);
x0 = zeros(n, 1);
cL = 5 * ones(n, 1);
cU = cL;
A = randi(10, m, n);

import model.LinEqProjQpModel;
quadModel = LinEqProjQpModel(Q, c, A, cL, cU, bL, bU, x0, '');

%% MATLAB's quadprog is the reference solution
xRef = quadprog(Q, c, [A; -A], [cU; -cL]);

%% Solve using SPG (works only if prob is quadratic)
import solvers.SpgSolver;
solver = SpgSolver(quadModel, 'aOptTol', 1e-10, 'aFeasTol', 1e-15, ...
    'progTol', 1e-15, 'maxIter', 1e4, 'verbose', 1);
solver = solver.solve();

nrmSol = norm(xRef - solver.x);
outInfo{end + 1} = sprintf(BODY_FORMAT, class(solver), solver.iter, ...
    solver.nObjFunc, solver.nGrad, solver.nHess, solver.pgNorm, nrmSol, ...
    solver.solveTime);

%% Solve using PQN
import solvers.PqnSolver;
solver = PqnSolver(quadModel, 'hess', 'exact', 'aOptTol', 1e-10, ...
    'progTol', 1e-15, 'aFeasTol', 1e-15, 'maxIter', 1e4, 'verbose', 1);
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