classdef PQNSolver < solvers.NLPSolver
    %% PQNSolver - Calls the MinConf_PQN solver
    % Original documentation follows:
    % ---------------------------------------------------------------------
    % function [x,f,funEvals] = minConf_PQN(funObj, x, funProj, options)
    % function [x,f] = minConf_PQN(funObj, funProj, x, options)
    %
    % Function for using a limited-memory projected quasi-Newton to solve
    %   problems of the form
    %   min funObj(x) s.t. x in C
    %
    % The projected quasi-Newton sub-problems are solved the spectral
    % projected gradient algorithm
    %
    %   @funObj(x): function to minimize (returns gradient as second
    %   argument)
    %   @funProj(x): function that returns projection of x onto C
    %
    %   options:
    %       verbose: level of verbosity (0: no output, 1: final, 2: iter
    %       (default), 3: debug)
    %       optTol: tolerance used to check for optimality (default: 1e-5)
    %       progTol: tolerance used to check for progress (default: 1e-9)
    %       maxIter: maximum number of calls to funObj (default: 500)
    %       maxProject: maximum number of calls to funProj (default:
    %       100000)
    %       numDiff: compute derivatives numerically (0: use user-supplied
    %       derivatives (default), 1: use finite differences, 2: use
    %       complex differentials)
    %       suffDec: sufficient decrease parameter in Armijo condition
    %       (default: 1e-4)
    %       corrections: number of lbfgs corrections to store (default: 10)
    %       adjustStep: use quadratic initialization of line search
    %       (default: 0)
    %       bbInit: initialize sub-problem with Barzilai-Borwein step
    %       (default: 1)
    %       SPGoptTol: optimality tolerance for SPG direction finding
    %       (default: 1e-6)
    %       SPGiters: maximum number of iterations for SPG direction
    %       finding (default: 10)
    
    properties (SetAccess = private, Hidden = false)
        % Subclass of nlp model representing the problem to solve
        % Contains the 'project' function; the projection of x onto C
        nlp;
        % x upon termination
        x;
        % Objective function value at x
        fx;
        % Norm of projected gradient at x
        proj_grad_norm;
        % Iteration counter
        iter;
        % SPG (sub-problem) iteration counter
        SPGiter;
        % Execution time
        time_total;
        % Exit flag
        istop;
        % Solved flag
        solved;
        % Projection calls counter
        nProj;
        % Objective function calls counter
        nObjFunc;
        % optTol relative to the initial gradient norm
        stopTol;
    end
    
    properties (Access = private, Hidden = false)
        % --- PQN parameters ---
        verbose; % 0, 1 or 2
        % Note: optTol := || P[x - g] - x ||
        optTol; % Optimality tolerance in the main problem
        progTol; % Progression tolerance in the main problem
        maxIter; % Maximum number of iterations
        maxEval; % Maximum number of calls to objective function
        maxProj; % Maximum number of calls to project function
        suffDec; % Sufficient decrease coefficient in linesearch
        corrections; % L-BFGS memory updates
        adjustStep; % Quadratic step length
        bbInit; % Use Barzilai-Borwein initialization in sub-problem
        hess; % Choice of hessian in the quadratic approx. (lbfgs || exact)
        lsMaxIter; % Maximal number of iterations in the linesearch
        fid;
        % --- SPG sub-problem parameters ---
        SPGoptTol; % Sub-problem optimality tolerance
        SPGprogTol; % Sub-problem progression tolerance
        SPGtestOpt; % Ensure that sub-problem is solved to optimality
        SPGverbose; % Ouput in MinConf_SPG (0, 1 or 2)
        SPGuseSpectral; % Modified descent (either bbType true or false)
        SPGprojectLS; % Do a projected linesearch
        SPGbbType; % Use Barzilai-Borwein step correction
        SPGmemory; % # previous vals to consider in non-monotone linesearch
    end
    
    properties (Hidden = true, Constant)
        EXIT_MSG = { ...
            ['First-Order Optimality Conditions Below optTol at', ...
            ' Initial Point\n'], ...                                    % 1
            'Directional Derivative below progTol\n', ...               % 2
            'First-Order Optimality Conditions Below optTol\n', ...     % 3
            'Step size below progTol\n', ...                            % 4
            'Function value changing by less than progTol\n', ...       % 5
            'Two consecutive linesearches have failed\n', ...           % 6
            'Function Evaluations exceeds maxIter\n', ...               % 7
            'Number of projections exceeds maxProject\n', ...           % 8
            'Maximal number of iterations reached\n', ...               % 9
            };
        LOG_HEADER = {'Iteration', 'Inner Iter', 'FunEvals', ...
            'Projections', 'Step Length', 'Function Val', 'Opt Cond', ...
            'gtd/|g|*|d|'};
        LOG_FORMAT = '%10s %10s %10s %10s %15s %15s %15s %15s\n';
        LOG_BODY = '%10d %10d %10d %10d %15.5e %15.5e %15.5e %15.5e\n';
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    methods (Access = public)
        
        function self = PQNSolver(nlp, varargin)
            %% Constructor
            % Inputs:
            %   - nlp: a subclass of a nlp model containing the 'obj'
            %   function that returns variable output arguments among the
            %   following: objective function, gradient and hessian at x.
            %   It must also contain the 'project' function that projects a
            %   vector x on the constraint set.
            %   - varargin (optional): the various parameters of the
            %   algorithm
            
            if ~isa(nlp, 'model.nlpmodel')
                error('nlp should be a nlpmodel');
            elseif ~ismethod(nlp, 'project')
                error('nlp doesn''t contain a project method');
            end
            
            self = self@solvers.NLPSolver(nlp, varargin{:});
            
            self.nlp = nlp;
            
            % Gathering optional arguments and setting default values
            p = inputParser;
            % --- PQN parameters ---
            p.addParameter('verbose', 2);
            p.addParameter('optTol', 1e-5);
            p.addParameter('progTol', 1e-9);
            p.addParameter('maxIter', 5e2);
            p.addParameter('maxEval', 5e2);
            p.addParameter('maxProj', 1e5);
            p.addParameter('suffDec', 1e-4);
            p.addParameter('corrections', 10);
            p.addParameter('adjustStep', 0);
            p.addParameter('bbInit', 0);
            p.addParameter('hess', 'lbfgs');
            p.addParameter('fid', 1);
            p.addParameter('lsMaxIter', 50); % Max iters for linesearch
            % --- SPG sub-problem parameters ---
            p.addParameter('SPGoptTol', 1e-5);
            p.addParameter('SPGprogTol', 1e-9);
            p.addParameter('SPGtestOpt', 0);
            p.addParameter('SPGverbose', 0);
            p.addParameter('SPGuseSpectral', 1);
            p.addParameter('SPGprojectLS', 0);
            p.addParameter('SPGbbType', 0);
            p.addParameter('SPGmemory', 1);
            
            p.parse(varargin{:});
            % --- PQN parameters ---
            self.verbose = p.Results.verbose;
            self.optTol = p.Results.optTol;
            self.progTol = p.Results.progTol;
            self.maxIter = p.Results.maxIter;
            self.maxEval = p.Results.maxEval;
            self.maxProj = p.Results.maxProj;
            self.suffDec = p.Results.suffDec;
            self.corrections = p.Results.corrections;
            self.adjustStep = p.Results.adjustStep;
            self.bbInit = p.Results.bbInit;
            self.hess = p.Results.hess;
            self.lsMaxIter = p.Results.lsMaxIter;
            self.fid = p.Results.fid;
            % --- SPG sub-problem parameters ---
            self.SPGoptTol = p.Results.SPGoptTol;
            self.SPGprogTol = p.Results.SPGprogTol;
            self.SPGtestOpt = p.Results.SPGtestOpt;
            self.SPGverbose = p.Results.SPGverbose;
            self.SPGuseSpectral = p.Results.SPGuseSpectral;
            self.SPGprojectLS = p.Results.SPGprojectLS;
            self.SPGbbType = p.Results.SPGbbType;
            self.SPGmemory = p.Results.SPGmemory;
        end
        
        function self = solve(self)
            %% Solve
            
            self.time_total = tic;
            
            if self.verbose == 2
                self.printHeaderFooter('header');
                self.printf(self.LOG_FORMAT, self.LOG_HEADER{:});
            end
            
            % Setting counters
            self.nProj = 0;
            self.nObjFunc = 0;
            self.iter = 1;
            self.SPGiter = 0;
            % Boolean for linesearch failure, a first failure will cause
            % the procedure to reset, whereas two consecutive failures will
            % cause the program to exit.
            failed = false;
            
            % Exit flag set to 0, will exit if not 0
            self.istop = 0;
            % Project initial parameter vector
            x = self.project(self.x0);
            
            % Evaluate initial parameters
            if ~strcmp(self.hess, 'lbfgs')
                [f, g, H] = self.obj(x);
            else
                [f, g] = self.obj(x);
            end
            
            self.stopTol = self.optTol * norm(g);
            pgnrm = norm(self.gpstep(x, g));
            % Check Optimality of Initial Point
            if pgnrm < self.stopTol + self.optTol
                self.istop = 1; % will bypass main loop
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %                      --- Main loop ---
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            while self.istop == 0
                % Compute Step Direction
                if self.iter == 1 || failed
                    % Reset if linesearch has failed
                    p = self.project(x - g);
                    S = zeros(self.nlp.n, 0);
                    Y = zeros(self.nlp.n, 0);
                    Hdiag = 1;
                else
                    y = g - g_old;
                    s = x - x_old;
                    switch self.hess
                        case 'lbfgs'
                            [S, Y, Hdiag] = PQNSolver.lbfgsUpdate(y, s, ...
                                self.corrections, ...
                                S, Y, Hdiag);
                            % Make Compact Representation
                            k = size(Y, 2);
                            L = zeros(k);
                            for j = 1:k
                                L(j + 1: k, j) = S(:, j + 1:k)' * Y(:, j);
                            end
                            N = [S / Hdiag, Y];
                            M = [S' * S / Hdiag L; L' -diag(diag(S' * Y))];
                            HvFunc = @(v) ...
                                PQNSolver.lbfgsHvFunc2(v, Hdiag, N, M);
                        case 'exact'
                            HvFunc = @(v) H * v;
                        otherwise
                            error('Unrecognized method');
                    end
                    
                    xSubInit = x;
                    if self.bbInit
                        % Use Barzilai-Borwein step to init the sub-problem
                        alpha = (s' * s) / (s' * y);
                        if alpha <= 1e-10 || alpha > 1e10  || isnan(alpha)
                            alpha = min(1, 1 / sum(abs(g)));
                        end
                        % Solve Sub-problem
                        xSubInit = x - alpha * g;
                    end
                    
                    % Solve Sub-problem, call MinConf_SPG
                    p = self.solveSubProblem(x, g, HvFunc, xSubInit);
                end
                
                d = p - x;
                g_old = g;
                x_old = x;
                
                % Check that Progress can be made along the direction
                gtd = g' * d;
                
                % We will use normgtd in the log later
                normgtd = gtd / (norm(g) * norm(d));
                if normgtd > -self.progTol
                    self.istop = 2;
                    % Leaving now saves some processing
                    break;
                end
                
                % Select Initial Guess to step length
                if self.iter == 1 || self.adjustStep == 0 || failed
                    t = 1;
                else
                    t = min(1, 2 * (f - f_old) / gtd);
                end
                
                % Bound Step length on first iteration
                if self.iter == 1
                    t = min(1, 1 / sum(abs(g)));
                end
                
                % Evaluate the Objective and Gradient at the Initial Step
                if t == 1
                    x_new = p;
                else
                    x_new = x + t * d;
                end
                [f_new, g_new] = self.obj(x_new);
                
                % Backtracking Line Search
                f_old = f;
                old_failed = failed;
                [x_new, f_new, g_new, t, failed] = self.backtracking(x, ...
                    x_new, f, f_new, g, g_new, d, t);
                
                % Take Step
                x = x_new;
                f = f_new;
                g = g_new;
                
                if strcmp(self.hess, 'exact') && ~failed
                    % If the linesearch computed a new point, re-evaluate
                    % the hessian (not done in backtracking)
                    [~, ~, H] = self.obj(x);
                end
                
                % Optimality value
                pgnrm = norm(self.gpstep(x, g));
                
                % Output Log
                if self.verbose >= 2
                    self.printf(self.LOG_BODY, self.iter, self.SPGiter, ...
                        self.nObjFunc, self.nProj, t, f, pgnrm, normgtd);
                end
                
                % Check optimality conditions
                if pgnrm < self.stopTol + self.optTol
                    self.istop = 3;
                elseif max(abs(t * d)) < self.progTol * norm(d) && ~failed
                    self.istop = 4;
                elseif abs((f - f_old)/max([f_old, f, 1])) ...
                        < self.progTol && ~failed
                    self.istop = 5;
                elseif failed && old_failed
                    % Two consecutive linesearches have failed, exit
                    self.istop = 6;
                elseif self.nObjFunc > self.maxEval
                    self.istop = 7;
                elseif self.nProj > self.maxProj
                    self.istop = 8;
                elseif self.iter >= self.maxIter
                    self.istop = 9;
                end
                
                if self.istop ~= 0
                    break;
                end
                
                self.iter = self.iter + 1;
            end
            self.x = x;
            self.fx = f;
            self.proj_grad_norm = pgnrm;
            
            % -------------------------------------------------------------
            % End of solve
            % -------------------------------------------------------------
            self.solved = ~(self.istop == 7 || self.istop == 8 || ...
                self.istop == 9);
            self.time_total = toc(self.time_total);
            if self.verbose
                self.printf('\nEXIT PQN: %s\nCONVERGENCE: %d\n', ...
                    self.EXIT_MSG{self.istop}, self.solved);
                self.printf('||Pg|| = %8.1e\n', self.proj_grad_norm);
                self.printf('Stop tolerance = %8.1e\n', self.stopTol);
            end
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    methods (Access = private)
        
        function printHeaderFooter(self, msg)
            switch msg
                case 'header'
                    % -----------------------------------------------------
                    % Print header.
                    % -----------------------------------------------------
                    self.printf('\n');
                    self.printf('%s\n', ['*', repmat('-',1,58), '*']);
                    self.printf([repmat('\t', 1, 3), 'MinConf_PQN \n']);
                    self.printf('%s\n\n', ['*', repmat('-',1,58), '*']);
                    self.printf(self.nlp.formatting())
                    self.printf('\nParameters\n----------\n')
                    self.printf('%-15s: %3s %8i', 'maxIter', '', ...
                        self.maxIter);
                    self.printf('\t%-15s: %3s %8.1e\n', ' optTol', '', ...
                        self.optTol);
                    self.printf('%-15s: %3s %8i', 'maxEval', '', ...
                        self.maxEval);
                    self.printf('\t%-15s: %3s %8.1e\n', ' progTol', '', ...
                        self.progTol);
                    self.printf('%-15s: %3s %8i', 'maxProj', '', ...
                        self.maxProj);
                    self.printf('\t%-15s: %3s %8.1e\n', ' suffDec', '', ...
                        self.suffDec);
                    self.printf('%-15s: %3s %8d', 'bbInit', '', ...
                        self.bbInit);
                    self.printf('\t%-15s: %3s %8d\n', ' corrections', ...
                        '', self.corrections);
                    self.printf('%-15s: %3s %8s', 'hess', '', ...
                        self.hess);
                    self.printf('\t%-15s: %3s %8d\n', ' adjustStep', '', ...
                        self.adjustStep);
                    self.printf('\nSPG parameters\n--------------\n')
                    self.printf('%-15s: %3s %8.1e', 'SPGoptTol', '', ...
                        self.SPGoptTol);
                    self.printf('\t%-15s: %3s %8.1e\n', ' SPGprogTol', ...
                        '', self.SPGprogTol);
                    self.printf('%-15s: %3s %8d', 'SPGtestOpt', '', ...
                        self.SPGtestOpt);
                    self.printf('\t%-15s: %3s %8d\n', ' SPGverbose', ...
                        '', self.SPGverbose);
                    self.printf('%-15s: %3s %8d', 'SPGuseSpectral', '', ...
                        self.SPGuseSpectral);
                    self.printf('\t%-15s: %3s %8d\n', ' SPGprojectLS', ...
                        '', self.SPGprojectLS);
                    self.printf('%-15s: %3s %8d', 'SPGbbType', '', ...
                        self.SPGbbType);
                    self.printf('\t%-15s: %3s %8d\n', ' SPGmemory', '', ...
                        self.SPGmemory);
                    self.printf('\n%15s: %3s %8s\n', 'Projection type', ...
                        '', class(self.nlp.projModel));
                case 'footer'
                    % -----------------------------------------------------
                    % Print footer
                    % -----------------------------------------------------
                    self.printf('\n')
                    self.printf(' %-27s  %6i     %-17s  %15.8e\n', ...
                        'No. of iterations', self.iter, ...
                        'Objective value', self.fx);
                    t1 = self.nlp.ncalls_fobj + self.nlp.ncalls_fcon;
                    t2 = self.nlp.ncalls_gobj + self.nlp.ncalls_gcon;
                    self.printf(' %-27s  %6i     %-17s    %6i\n', ...
                        'No. of calls to objective' , t1, ...
                        'No. of calls to gradient', t2);
                    self.printf(' %-27s  %6i \n', ...
                        'No. of Hessian-vector prods', ...
                        self.nlp.ncalls_hvp);
                    self.printf('\n');
                    tt = self.time_total;
                    t1 = self.nlp.time_fobj + self.nlp.time_fcon;
                    t1t = round(100 * t1/tt);
                    t2 = self.nlp.time_gobj + self.nlp.time_gcon;
                    t2t = round(100 * t2/tt);
                    self.printf([' %-24s %6.2f (%3d%%)  %-20s %6.2f', ...
                        '(%3d%%)\n'], 'Time: function evals' , t1, t1t, ...
                        'gradient evals', t2, t2t);
                    t1 = self.nlp.time_hvp; t1t = round(100 * t1/tt);
                    self.printf([' %-24s %6.2f (%3d%%)  %-20s %6.2f', ...
                        '(%3d%%)\n'], 'Time: Hessian-vec prods', t1, ...
                        t1t, 'total solve', tt, 100);
                otherwise
                    error('Unrecognized case in printHeaderFooter');
            end
        end
        
        function printf(self, varargin)
            %% Printf - prints variables arguments to a file
            fprintf(self.fid, varargin{:});
        end
        
        function s = gpstep(self, x, g)
            %% GPStep - computing the projected gradient
            % Inputs:
            %   - x: current point
            %   - g: gradient at x
            % Output:
            %   - s: projected gradient
            % Calling project to increment projection counter
            s = self.project(x - g) - x;
        end
        
        function [f, g, H] = obj(self, x)
            %% Obj - Evaluate objective function, gradient and hessian at x
            % Input:
            %   - x: current point
            % Ouputs (variable):
            %   - f: value of the objective function at x
            %   - g: gradient of the objective function at x
            %   - H: hessian of the objective function at x
            if nargout == 1
                f = self.nlp.obj(x);
            elseif nargout == 2
                [f, g] = self.nlp.obj(x);
            elseif nargout == 3
                [f, g, H] = self.nlp.obj(x);
            end
            self.nObjFunc = self.nObjFunc + 1;
        end
        
        function z = project(self, x)
            %% Project - projecting x on the constraint set
            z = self.nlp.project(x);
            self.nProj = self.nProj + 1;
        end
        
        function p = solveSubProblem(self, x, g, H, x_init)
            %% solveSubProblem - minimize the constrained quad. approx.
            % Calls MinConf_SPG on a quadratic approximation of the
            % objective function at x. Uses the 'project' function defined
            % in projModel to project values on the constraint set.
            % Inputs:
            %   - x: origin from where to compute the minimizing direction
            %   - g: gradient of the objective function at x
            %   - H: hessian of the objective function at x (function
            %   handle).
            %   - x_init: initial point to use in MinConf_SPG
            %   - pgnrm: norm of the projected gradient at the current
            %   iteration
            % Ouput:
            %   - p: computed minimizing direction from x
            
            % Uses SPG to solve for projected quasi-Newton direction,
            % setting parameters
            SPGoptions.verbose = self.SPGverbose;
            % Impose optimality as fraction of the norm of the proj. grad.
            SPGoptions.optTol = self.SPGoptTol;
            SPGoptions.progTol = self.SPGprogTol;
            SPGoptions.testOpt = self.SPGtestOpt;
            SPGoptions.useSpectral = self.SPGuseSpectral;
            SPGoptions.projectLS = self.SPGprojectLS;
            SPGoptions.bbType = self.SPGbbType;
            SPGoptions.memory = self.SPGmemory;
            self.lsMaxIter = 50;
            
            % Building a quadratic approximation
            import model.ShiftedQPModel;
            quadModel = model.ShiftedQPModel('', x_init, x, g, H, ...
                @(p) self.nlp.project(p));
            
            % Solving using MinConf_SPG
            subProblem = MinConf_SPG(quadModel, SPGoptions);
            
            if ~subProblem.solved
                error('MinConf_SPG exited without ''pseudo''-convergence');
            end
            
            % Retrieving solution, number of proj calls & inner iterations
            self.SPGiter = subProblem.iter;
            self.nProj = self.nProj + subProblem.nProj;
            p = subProblem.x;
        end
        
        function [x_new, f_new, g_new, t, failed] = backtracking(self, ...
                x, x_new, f, f_new, g, g_new, d, t)
            
            failed = false;
            lsIter = 1;
            while f_new > f + self.suffDec * g' * (x_new - x)
                
                if self.verbose == 2
                    fprintf('Halving Step Size\n');
                end
                t = t/2;
                
                % Check whether step has become too small
                if sum(abs(t * d)) < self.progTol * norm(d) || ...
                        t == 0 || lsIter > self.lsMaxIter
                    if self.verbose == 2
                        fprintf('Line Search failed\n');
                    end
                    t = 0;
                    failed = true;
                    x_new = x;
                    f_new = f;
                    g_new = g;
                    return;
                end
                
                % Evaluate New Point
                x_new = x + t * d;
                [f_new, g_new] = self.obj(x_new);
                lsIter = lsIter + 1;
            end
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    methods (Static, Access = private)
        function Hv = lbfgsHvFunc2(v, Hdiag, N, M)
            %% Original function from the MinFunc folder
            % L-BFGS hessian-vector product
            Hv = v / Hdiag - N * (M \ (N' * v));
        end
        
        function [old_dirs, old_stps, Hdiag] = lbfgsUpdate(y, s, corr, ...
                old_dirs, old_stps, Hdiag)
            %% Original function from the MinFunc folder
            % Limited memory BFGS hessian update
            ys = y' * s;
            if ys > 1e-10
                numCorrections = size(old_dirs, 2);
                if numCorrections < corr
                    % Full Update
                    old_dirs(:, numCorrections + 1) = s;
                    old_stps(:, numCorrections + 1) = y;
                else
                    % Limited-Memory Update
                    old_dirs = [old_dirs(:, 2:corr), s];
                    old_stps = [old_stps(:, 2:corr), y];
                end
                
                % Update scale of initial Hessian approximation
                Hdiag = ys / (y' * y);
            end
        end
    end
end