function perf(T, header, varargin)
%% Performance profile
% Based on the works of Dolan & Moré and the implementation of Sofi et al.
% Inputs:
%   - T: is a matrix of n_problems x n_solvers for a given performance
%   metric.
%   - header: { {ns x 1}, {nm x 1} } cell array of cell arrays containing
%   the names of the solvers and the metrics used.
%   - varargin: (optional arguments)
%       * logPlot
%       * saveFolder
%       * display
%       * prefix

%% Parsing input arguments
p = inputParser;
p.KeepUnmatched = false;
p.PartialMatching = false;
p.addParameter('logPlot', false);
p.addParameter('saveFolder', '');
p.addParameter('display', true);
p.addParameter('prefix', '');

p.parse(varargin{:});

logPlot = p.Results.logPlot;
saveFolder = p.Results.saveFolder;
display = p.Results.display;
prefix = p.Results.prefix;

colors = ['m', 'b', 'r', 'g', 'c', 'k', 'y'];
markers = ['x', 'x', 'x', 'x', 'v', '^', 'o'];

%% Evaluating ratios
% nProb = # of problems,
% nSolv = # of solvers,
% nMetr = # of metrics.
[nProb, nSolv, nMetr] = size(T);

% Finding the 'best' value for each problem and each metric
minperf = min(T, [], 2);

% Evaluate the performance ratios
R = T ./ repmat(minperf, 1, nSolv);

if logplot
    % Convert to log scale
    R = log2(R);
end

% Handling NaN values
maxRatio = repmat(2 * max(max(R)), nProb, nSolv);
R(isnan(R)) = maxRatio(isnan(R));
R = sort(R);

%% Plotting
for mm = 1 : nMetr
    % For each metric
    h = figure;
    if ~display
        % Don't show figure
        set(h, 'Visible', 'off');
    end
    
    for ss = 1 : nSolv
        % For each solver
        [xs, ys] = stairs(R(:, ss, mm), (1 : nProb) / nProb);
        option = ['-', colors(ss), markers(ss)];
        plot(xs, ys, option, 'MarkerSize', 3);
        hold on;
    end
    
    axis([0.1, 1.1*maxRatio, 0, 1]);
    ylabel(header{2}{mm})
    legend(header{1}{:}, 'location', 'BestOutside');
    
    if ~strcmp(saveFolder, '')
        % Print to eps
        figName = [prefix, header{2}{mm}];
        figName = [saveFolder, figName(isstrprop(figName, ...
            'alphanum')), '.eps'];
        print(h, '-depsc', '-r200', figName);
    end
end

end
