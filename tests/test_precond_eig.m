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

%% Saving console log
diary(['data/test_precond_eig', num2str(FACTOR), '.txt']);

%% Checking for the eigenvalues of prec
figure;
semilogy(sort(prec.Mdiag(:).^2), 'r.');
ylabel('\lambda_{CC''}');
axis tight;
a = gca();
a.YLim = [1e-5, 1e0];

% Generating a random restriction on C : B*C <=> C(ind, :)
precSize = prec.Nblks * prec.BlkSiz;
aimFrac = 3e-2;
ind = false(precSize, 1);
ind(unique(randi(precSize, round(aimFrac * precSize, 0), 1))) = true;
realFrac = sum(ind) / precSize;
% Creating opSpot for the restricted preconditionner
CCt = opFunction(precSize, precSize, @(x, mode) prec.AdjointDirect(x));
CCt = CCt(ind, ind);
% Computing the eigen values using a Krylov method (provided by eigs)
opts.isreal = false;
% opts.tol = 1e-6;
% opts.maxit = 1e6;
opts.issym = true;
opts.disp = 1;
% Ask for sum(ind) - 3 eigen values in order for MATLAB to call eigs...
precEigVals = eigs(@(x) CCt * x, sum(ind), sum(ind) - 3, 'LM', opts);

figure;
semilogy(precEigVals, 'b.');
ylabel(['\lambda_{CC''}', sprintf(' (restrict. = %.2f)', realFrac)]);
axis tight;
a = gca();
a.YLim = [1e-5, 1e0];

%% Closing diary
diary off;