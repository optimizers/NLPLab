clc;
clear all;
close all;

%% Seting MATLAB's path
addpath('~/Masters/nlplab');
addpath('~/Masters/logging4matlab/');
addpath('~/Masters/Spot');

%% Format
HEADER = {'Solver', '#iter', '#f', '#g', '#H', '|Pg|', ...
    '|x*-x|', 'RT'};
HEADER_FORMAT = ['%25s', repmat('%10s ', 1, 7), '\n'];
BODY_FORMAT = '%25s %10d %10d %10d %10d %10.1e %10.1e %10.1e\n';
outInfo = {};

%% Building the model
% Quadratic objective function, upper and lower bounds
m = 5e1;
n = m;
Q = diag(1 * ones(m - 1, 1), -1) + diag(10 * ones(m, 1), 0) + ...
    diag(1 * ones(m - 1, 1), 1);
c = -randi(2, n, 1);
bL = -(n : -1 : 1)';
bU = 1e-1 * (1 : n)';
x0 = zeros(n, 1);
cL = -inf(n, 1);
cU = inf(n, 1);
A = [];

%% MATLAB's quadprog is the reference solution
xRef = quadprog(Q, c, [], [], [], [], bL, bU);

%% Solve using Pn
import model.BoundProjQpModel;
quadModel = BoundProjQpModel(Q, c, A, cL, cU, bL, bU, x0, '');
import solvers.PnSolver;
solver = PnSolver(quadModel, 'optTol', 1e-10, 'maxIter', 1e4, ...
    'verbose', 2);
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