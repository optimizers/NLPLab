classdef PqnSolver < solvers.NlpSolver
    %% PqnSolver - Calls the MinConf_PQN solver
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
    %       aOptTol: tolerance used to check for optimality (default: 1e-5)
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
    %       spgaOptTol: optimality tolerance for SPG direction finding
    %       (default: 1e-6)
    %       spgIters: maximum number of iterations for SPG direction
    %       finding (default: 10)
    
    
    properties (SetAccess = private, Hidden = false)
        % Norm of projected gradient at x
        pgNorm;
        % SPG (sub-problem) iteration counter
        spgIter;
        % Exit flag
        iStop;
        % Projection calls counter
        nProj;
    end % gettable private properties
    
    properties (Access = private, Hidden = false)
        % PQN parameters
        verbose; % 0, 1 or 2
        progTol; % Progression tolerance in the main problem
        maxEval; % Maximum number of calls to objective function
        maxProj; % Maximum number of calls to project function
        suffDec; % Sufficient decrease coefficient in linesearch
        corrections; % L-BFGS memory updates
        adjustStep; % Quadratic step length
        bbInit; % Use Barzilai-Borwein initialization in sub-problem
        hess; % Choice of hessian in the quadratic approx. (lbfgs || exact)
        maxIterLS; % Maximal number of iterations in the linesearch
        fid;
        % SPG sub-problem parameters
        spgaOptTol; % Sub-problem optimality tolerance
        spgProgTol; % Sub-problem progression tolerance
        spgTestOpt; % Ensure that sub-problem is solved to optimality
        spgVerbose; % Ouput in MinConf_SPG (0, 1 or 2)
        spgUseSpectral; % Modified descent (either bbType true or false)
        spgProjectLS; % Do a projected linesearch
        spgBbType; % Use Barzilai-Borwein step correction
        spgMemory; % # previous vals to consider in non-monotone linesearch
    end % private properties
    
    properties (Hidden = true, Constant)
        EXIT_MSG = { ...
            ['First-Order Optimality Conditions Below aOptTol at', ...
            ' Initial Point\n'], ...                                    % 1
            'Directional Derivative below progTol\n', ...              % 2
            'First-Order Optimality Conditions Below aOptTol\n', ...   % 3
            'Step size below progTol\n', ...                           % 4
            'Function value changing by less than progTol\n', ...      % 5
            'Two consecutive linesearches have failed\n', ...           % 6
            'Function Evaluations exceeds maxIter\n', ...              % 7
            'Number of projections exceeds maxProject\n', ...          % 8
            'Maximal number of iterations reached\n', ...               % 9
            };
        LOG_HEADER = {'Iteration', 'Inner Iter', 'FunEvals', ...
            'Projections', 'Step Length', 'Function Val', 'Opt Cond', ...
            'gtd/|g|*|d|'};
        LOG_FORMAT = '%10s %10s %10s %10s %15s %15s %15s %15s\n';
        LOG_BODY = '%10d %10d %10d %10d %15.5e %15.5e %15.5e %15.5e\n';
    end % constant properties
    
    
    methods (Access = public)
        
        function self = PqnSolver(nlp, varargin)
            %% Constructor
            % Inputs:
            %   - nlp: a subclass of a nlp model containing the 'obj'
            %   function that returns variable output arguments among the
            %   following: objective function, gradient and hessian at x.
            %   It must also contain the 'project' function that projects a
            %   vector x on the constraint set.
            %   - varargin (optional): the various parameters of the
            %   algorithm
            
            % Specific check for the PQN solver
            if ~ismethod(nlp, 'project')
                error('nlp doesn''t contain a project method');
            end
            
            % Gathering optional arguments and setting default values
            p = inputParser;
            p.KeepUnmatched = true;
            % PQN parameters
            p.addParameter('verbose', 2);
            p.addParameter('progTol', 1e-9);
            p.addParameter('maxEval', 5e2);
            p.addParameter('maxProj', 1e5);
            p.addParameter('suffDec', 1e-4);
            p.addParameter('corrections', 10);
            p.addParameter('adjustStep', 0);
            p.addParameter('bbInit', 0);
            p.addParameter('hess', 'lbfgs');
            p.addParameter('fid', 1);
            p.addParameter('maxIterLS', 50); % Max iters for linesearch
            % SPG sub-problem parameters
            p.addParameter('spgaOptTol', 1e-5);
            p.addParameter('spgProgTol', 1e-9);
            p.addParameter('spgTestOpt', 1);
            p.addParameter('spgVerbose', 0);
            p.addParameter('spgUseSpectral', 1);
            p.addParameter('spgProjectLS', 1);
            p.addParameter('spgBbType', 1);
            p.addParameter('spgMemory', 10);
            
            p.parse(varargin{:});
            
            self = self@solvers.NlpSolver(nlp, p.Unmatched);
            
            % PQN parameters
            self.verbose = p.Results.verbose;
            self.progTol = p.Results.progTol;
            self.maxEval = p.Results.maxEval;
            self.maxProj = p.Results.maxProj;
            self.suffDec = p.Results.suffDec;
            self.corrections = p.Results.corrections;
            self.adjustStep = p.Results.adjustStep;
            self.bbInit = p.Results.bbInit;
            self.hess = p.Results.hess;
            self.maxIterLS = p.Results.maxIterLS;
            self.fid = p.Results.fid;
            % SPG sub-problem parameters
            self.spgaOptTol = p.Results.spgaOptTol;
            self.spgProgTol = p.Results.spgProgTol;
            self.spgTestOpt = p.Results.spgTestOpt;
            self.spgVerbose = p.Results.spgVerbose;
            self.spgUseSpectral = p.Results.spgUseSpectral;
            self.spgProjectLS = p.Results.spgProjectLS;
            self.spgBbType = p.Results.spgBbType;
            self.spgMemory = p.Results.spgMemory;
        end % constructor
        
        function self = solve(self)
            %% Solve
            
            self.solveTime = tic;
            
            if self.verbose >= 2
                self.printHeaderFooter('header');
                self.printf(self.LOG_FORMAT, self.LOG_HEADER{:});
            end
            
            % Setting counters
            self.nProj = 0;
            self.nObjFunc = 0;
            self.iter = 1;
            self.spgIter = 0;
            % Boolean for linesearch failure, a first failure will cause
            % the procedure to reset, whereas two consecutive failures will
            % cause the program to exit.
            failed = false;
            
            % Exit flag set to 0, will exit if not 0
            self.iStop = 0;
            % Project initial parameter vector
            x = self.project(self.nlp.x0);
            
            % Evaluate initial parameters
            if ~strcmp(self.hess, 'lbfgs')
                [f, g, H] = self.obj(x);
            else
                [f, g] = self.obj(x);
            end
            
            self.rOptTol = self.aOptTol * norm(g);
            self.rFeasTol = self.aFeasTol * norm(f);
            pgnrm = norm(self.gpstep(x, g));
            % Check Optimality of Initial Point
            if pgnrm < self.rOptTol + self.aOptTol
                self.iStop = 1; % will bypass main loop
            end
            
            %% Main loop
            while self.iStop == 0
                % Compute Step Direction
                if self.iter == 1 || failed
                    % Reset if linesearch has failed
                    p = self.project(x - g);
                    S = zeros(self.nlp.n, 0);
                    Y = zeros(self.nlp.n, 0);
                    Hdiag = 1;
                else
                    y = g - gOld;
                    s = x - xOld;
                    switch self.hess
                        case 'lbfgs'
                            [S, Y, Hdiag] = ...
                                solvers.PqnSolver.lbfgsUpdate(y, s, ...
                                self.corrections, S, Y, Hdiag);
                            % Make Compact Representation
                            k = size(Y, 2);
                            L = zeros(k);
                            for j = 1:k
                                L(j + 1: k, j) = S(:, j + 1:k)' * Y(:, j);
                            end
                            N = [S / Hdiag, Y];
                            M = [S'*S/Hdiag, L; L', -diag(diag(S' * Y))];
                            HvFunc = @(v) ...
                                solvers.PqnSolver.lbfgsHvFunc2(v, ...
                                Hdiag, N, M);
                        case 'exact'
                            HvFunc = @(v) H * v;
                        otherwise
                            error('Unrecognized method');
                    end
                    
                    xSubInit = x;
                    if self.bbInit
                        % Use Barzilai-Borwein step to init the sub-problem
                        alph = (s' * s) / (s' * y);
                        if alph <= 1e-10 || alph > 1e10  || isnan(alph)
                            alph = min(1, 1 / sum(abs(g)));
                        end
                        % Solve Sub-problem
                        xSubInit = x - alph * g;
                    end
                    
                    % Solve Sub-problem, call MinConf_SPG
                    p = self.solveSubProblem(x, g, HvFunc, xSubInit);
                end
                
                d = p - x;
                gOld = g;
                xOld = x;
                
                % Check that Progress can be made along the direction
                gtd = g' * d;
                
                % We will use normgtd in the log later
                normgtd = gtd / (norm(g) * norm(d));
                if normgtd > -self.progTol || isnan(normgtd)
                    % d might be null
                    self.iStop = 2;
                    % Leaving now saves some processing
                    break;
                end
                
                % Select Initial Guess to step length
                if self.iter == 1 || self.adjustStep == 0 || failed
                    t = 1;
                else
                    t = min(1, 2 * (f - fOld) / gtd);
                end
                
                % Bound Step length on first iteration
                if self.iter == 1
                    t = min(1, 1 / sum(abs(g)));
                end
                
                % Evaluate the Objective and Gradient at the Initial Step
                if t == 1
                    xNew = p;
                else
                    xNew = x + t * d;
                end
                [fNew, gNew] = self.obj(xNew);
                
                % Backtracking Line Search
                fOld = f;
                oldFailed = failed;
                [xNew, fNew, gNew, t, failed] = self.backtracking(x, ...
                    xNew, f, fNew, g, gNew, d, t);
                
                % Take Step
                x = xNew;
                f = fNew;
                g = gNew;
                
                if strcmp(self.hess, 'exact') && ~failed
                    % If the linesearch computed a new point, re-evaluate
                    % the hessian (not done in backtracking)
                    [~, ~, H] = self.obj(x);
                end
                
                % Optimality value
                pgnrm = norm(self.gpstep(x, g));
                
                % Output Log
                if self.verbose >= 2
                    self.printf(self.LOG_BODY, self.iter, self.spgIter, ...
                        self.nObjFunc, self.nProj, t, f, pgnrm, normgtd);
                end
                
                % Check optimality conditions
                if pgnrm < self.rOptTol + self.aOptTol
                    self.iStop = 3;
                elseif max(abs(t * d)) < self.progTol * norm(d) && ~failed
                    self.iStop = 4;
                elseif abs((f - fOld)/max([fOld, f, 1])) ...
                        < self.rFeasTol && ~failed
                    self.iStop = 5;
                elseif failed && oldFailed
                    % Two consecutive linesearches have failed, exit
                    self.iStop = 6;
                elseif self.nObjFunc > self.maxEval
                    self.iStop = 7;
                elseif self.nProj > self.maxProj
                    self.iStop = 8;
                elseif self.iter > self.maxIter
                    self.iStop = 9;
                end
                if self.iStop ~= 0
                    break;
                end
                self.iter = self.iter + 1;
            end % main loop
            self.x = x;
            self.fx = f;
            self.pgNorm = pgnrm;
            
            self.nObjFunc = self.nlp.ncalls_fobj + self.nlp.ncalls_fcon;
            self.nGrad = self.nlp.ncalls_gobj + self.nlp.ncalls_gcon;
            self.nHess = self.nlp.ncalls_hvp + self.nlp.ncalls_hes;
            
            %% End of solve
            self.solved = ~(self.iStop == 7 || self.iStop == 8 || ...
                self.iStop == 9);
            self.solveTime = toc(self.solveTime);
            if self.verbose
                self.printf('\nEXIT PQN: %s\nCONVERGENCE: %d\n', ...
                    self.EXIT_MSG{self.iStop}, self.solved);
                self.printf('||Pg|| = %8.1e\n', self.pgNorm);
                self.printf('Stop tolerance = %8.1e\n', self.rOptTol);
            end
            if self.verbose >= 2
                self.printHeaderFooter('footer');
            end
        end % solve
        
    end % public methods
    
    
    methods (Access = private)
        
        function printHeaderFooter(self, msg)
            switch msg
                case 'header'
                    % Print header
                    self.printf('\n');
                    self.printf('%s\n', ['*', repmat('-',1,58), '*']);
                    self.printf([repmat('\t', 1, 3), 'MinConf_PQN \n']);
                    self.printf('%s\n\n', ['*', repmat('-',1,58), '*']);
                    self.printf(self.nlp.formatting())
                    self.printf('\nParameters\n----------\n')
                    self.printf('%-15s: %3s %8i', 'maxIter', '', ...
                        self.maxIter);
                    self.printf('\t%-15s: %3s %8.1e\n', ' aOptTol', '', ...
                        self.aOptTol);
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
                    self.printf('%-15s: %3s %8.1e', 'spgaOptTol', '', ...
                        self.spgaOptTol);
                    self.printf('\t%-15s: %3s %8.1e\n', ' spgProgTol', ...
                        '', self.spgProgTol);
                    self.printf('%-15s: %3s %8d', 'spgTestOpt', '', ...
                        self.spgTestOpt);
                    self.printf('\t%-15s: %3s %8d\n', ' spgVerbose', ...
                        '', self.spgVerbose);
                    self.printf('%-15s: %3s %8d', 'spgUseSpectral', '', ...
                        self.spgUseSpectral);
                    self.printf('\t%-15s: %3s %8d\n', ' spgProjectLS', ...
                        '', self.spgProjectLS);
                    self.printf('%-15s: %3s %8d', 'spgBbType', '', ...
                        self.spgBbType);
                    self.printf('\t%-15s: %3s %8d\n', ' spgMemory', '', ...
                        self.spgMemory);
                case 'footer'
                    % Print footer
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
                    tt = self.solveTime;
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
            end % switch
        end % printHeaderFooter
        
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
        
        function p = solveSubProblem(self, x, g, H, xInit)
            %% solveSubProblem - minimize the constrained quad. approx.
            % Calls MinConf_SPG on a quadratic approximation of the
            % objective function at x. Uses the 'project' function defined
            % in the nlpmodel to project values on the constraint set.
            % Inputs:
            %   - x: origin from where to compute the minimizing direction
            %   - g: gradient of the objective function at x
            %   - H: hessian of the objective function at x (function
            %   handle).
            %   - xInit: initial point to use in MinConf_SPG
            %   - pgnrm: norm of the projected gradient at the current
            %   iteration
            % Ouput:
            %   - p: computed minimizing direction from x
            
            % Uses SPG to solve for projected quasi-Newton direction,
            % setting parameters
            SpgOptions.verbose = self.spgVerbose;
            % Impose optimality as fraction of the norm of the proj. grad.
            SpgOptions.aOptTol = self.aOptTol; % self.spgaOptTol;
            SpgOptions.progTol = self.progTol; % self.spgProgTol;
            SpgOptions.testOpt = self.spgTestOpt;
            SpgOptions.useSpectral = self.spgUseSpectral;
            SpgOptions.projectLS = self.spgProjectLS;
            SpgOptions.bbType = self.spgBbType;
            SpgOptions.memory = self.spgMemory;
            
            % Building a quadratic approximation
            import model.ShiftedQpModel;
            quadModel = model.ShiftedQpModel('', xInit, x, g, H, ...
                @(p) self.nlp.project(p));
            
            % Solving using MinConf_SPG
            import solvers.SpgSolver
            subProblem = SpgSolver(quadModel, SpgOptions);
            subProblem = subProblem.solve();
            
            if ~subProblem.solved
                error('SpgSolver exited without ''pseudo''-convergence');
            end
            
            % Retrieving solution, number of proj calls & inner iterations
            self.spgIter = subProblem.iter;
            self.nProj = self.nProj + subProblem.nProj;
            p = subProblem.x;
        end % solveSubProblem
        
        function [xNew, fNew, gNew, t, failed] = backtracking(self, ...
                x, xNew, f, fNew, g, gNew, d, t)
            
            failed = false;
            iterLS = 1;
            while fNew > f + self.suffDec * g' * (xNew - x)
                
                if self.verbose == 2
                    fprintf('Halving Step Size\n');
                end
                t = t/2;
                
                % Check whether step has become too small
                if sum(abs(t * d)) < self.progTol * norm(d) || ...
                        t == 0 || iterLS > self.maxIterLS
                    if self.verbose == 2
                        fprintf('Line Search failed\n');
                    end
                    t = 0;
                    failed = true;
                    xNew = x;
                    fNew = f;
                    gNew = g;
                    return;
                end
                
                % Evaluate New Point
                xNew = x + t * d;
                [fNew, gNew] = self.obj(xNew);
                iterLS = iterLS + 1;
            end
        end % backtracking
        
    end % private methods
    
    
    methods (Static, Access = private)
        
        function Hv = lbfgsHvFunc2(v, Hdiag, N, M)
            %% Original function from the MinFunc folder
            % L-BFGS hessian-vector product
            Hv = v / Hdiag - N * (M \ (N' * v));
        end
        
        function [oldDirs, oldStps, Hdiag] = lbfgsUpdate(y, s, corr, ...
                oldDirs, oldStps, Hdiag)
            %% Original function from the MinFunc folder
            % Limited memory BFGS hessian update
            ys = y' * s;
            if ys > 1e-10
                nCorr = size(oldDirs, 2);
                if nCorr < corr
                    % Full Update
                    oldDirs(:, nCorr + 1) = s;
                    oldStps(:, nCorr + 1) = y;
                else
                    % Limited-Memory Update
                    oldDirs = [oldDirs(:, 2:corr), s];
                    oldStps = [oldStps(:, 2:corr), y];
                end
                
                % Update scale of initial Hessian approximation
                Hdiag = ys / (y' * y);
            end
        end
        
    end % static private methods
    
end % class