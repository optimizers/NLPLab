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
problemsFile = './bounds.lst';

[problems, notFound] = findProblems(lookInto, problemsFile);

%% Solve the problems with Bcflashsolver
% Load Bcflash
import solvers.BcflashSolver;
% Load the AMPL-interface
import ampl.ampl_interface;