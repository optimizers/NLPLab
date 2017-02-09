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
        % SPG (sub-problem) iteration counter
        spgIter;
        % Projection calls counter
        nProj;
        maxProj; % Maximum number of calls to project function
        suffDec; % Sufficient decrease coefficient in linesearch
        corrections; % L-BFGS memory updates
        adjustStep; % Quadratic step length
        bbInit; % Use Barzilai-Borwein initialization in sub-problem
        hess; % Choice of hessian in the quadratic approx. (lbfgs || exact)
        maxIterLS; % Maximal number of iterations in the linesearch
        fid;
        % SPG sub-problem parameters
        spgTestOpt; % Ensure that sub-problem is solved to optimality
        spgVerbose; % Ouput in MinConf_SPG (0, 1 or 2)
        spgUseSpectral; % Modified descent (either bbType true or false)
        spgProjectLS; % Do a projected linesearch
        spgBbType; % Use Barzilai-Borwein step correction
        spgMemory; % # previous vals to consider in non-monotone linesearch
    end % private properties
    
    properties (Hidden = true, Constant)
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
            p.PartialMatching = false;
            % PQN parameters
            p.addParameter('maxProj', 1e5);
            p.addParameter('suffDec', 1e-4);
            p.addParameter('corrections', 10);
            p.addParameter('adjustStep', 0);
            p.addParameter('bbInit', 0);
            p.addParameter('hess', 'lbfgs');
            p.addParameter('fid', 1);
            p.addParameter('maxIterLS', 30); % Max iters for linesearch
            % SPG sub-problem parameters
            p.addParameter('spgTestOpt', 1);
            p.addParameter('spgVerbose', 0);
            p.addParameter('spgUseSpectral', 1);
            p.addParameter('spgProjectLS', 0);
            p.addParameter('spgBbType', 1);
            p.addParameter('spgMemory', 1);
            
            p.parse(varargin{:});
            
            self = self@solvers.NlpSolver(nlp, p.Unmatched);
            
            % PQN parameters
            self.maxProj = p.Results.maxProj;
            self.suffDec = p.Results.suffDec;
            self.corrections = p.Results.corrections;
            self.adjustStep = p.Results.adjustStep;
            self.bbInit = p.Results.bbInit;
            self.hess = p.Results.hess;
            self.maxIterLS = p.Results.maxIterLS;
            self.fid = p.Results.fid;
            % SPG sub-problem parameters
            self.spgTestOpt = p.Results.spgTestOpt;
            self.spgVerbose = p.Results.spgVerbose;
            self.spgUseSpectral = p.Results.spgUseSpectral;
            self.spgProjectLS = p.Results.spgProjectLS;
            self.spgBbType = p.Results.spgBbType;
            self.spgMemory = p.Results.spgMemory;
            
            import utils.PrintInfo;
            %             import linesearch.armijo;
        end % constructor
        
        function self = solve(self)
            %% Solve
            
            self.solveTime = tic;
            
            printObj = utils.PrintInfo('Pqn');
            
            if self.verbose >= 2
                extra = containers.Map( ...
                    {'suffDec', 'bbInit', 'corrections', 'hess', ...
                    'adjustStep', 'spgTestOpt', 'spgVerbose', ...
                    'spgUseSpectral', 'spgProjectLS', 'spgBbType', ...
                    'spgMemory', 'maxProj'}, ...
                    {self.suffDec, self.bbInit, self.corrections, ...
                    self.hess, self.adjustStep, self.spgTestOpt, ...
                    self.spgVerbose, self.spgUseSpectral, ...
                    self.spgProjectLS, self.spgBbType, self.spgMemory, ...
                    self.maxProj});
                printObj.header(self, extra);
                self.printf(self.LOG_FORMAT, self.LOG_HEADER{:});
            end
            
            % Setting counters
            self.nProj = 0;
            self.iter = 1;
            self.spgIter = 0;
            
            % Exit flag set to 0, will exit if not 0
            self.iStop = self.EXIT_NONE;
            % Project initial parameter vector
            x = self.project(self.nlp.x0);
            
            % Evaluate initial parameters
            if ~strcmp(self.hess, 'lbfgs')
                [f, g, H] = self.nlp.obj(x);
            else
                [f, g] = self.nlp.obj(x);
            end
            fOld = -inf;
            
            % Relative stopping tolerance
            self.rOptTol = self.aOptTol * norm(g);
            self.rFeasTol = self.aFeasTol * abs(f);
            
            pgnrm = norm(self.gpstep(x, g));
            % Check Optimality of Initial Point
            if pgnrm < self.rOptTol + self.aOptTol
                self.iStop = 1; % will bypass main loop
            end
            
            %% Main loop
            while ~self.iStop % self.iStop == 0
                % Compute Step Direction
                if self.iter == 1
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
                % Descent direction
                d = p - x;
                % Save for L-BFGS updates
                gOld = g;
                xOld = x;
                
                % Check that Progress can be made along the direction
                gtd = g' * d;
                
                % We will use normgtd in the log later
                if gtd > -self.aFeasTol * norm(g) * norm(d) - self.aFeasTol
                    % d might be null
                    self.iStop = self.EXIT_DIR_DERIV;
                    % Leaving now saves some processing
                    break;
                end
                
                % Backtracking Line Search
                [x, fNew, g, t] = self.backtracking(x, f, g, d, fOld, gtd);
                fOld = f;
                f = fNew;
                
                if strcmp(self.hess, 'exact')
                    [~, ~, H] = self.nlp.obj(x);
                end
                
                % Optimality value
                pgnrm = norm(self.gpstep(x, g));
                
                % Output Log
                if self.verbose >= 2
                    self.nObjFunc = self.nlp.ncalls_fobj + ...
                        self.nlp.ncalls_fcon;
                    self.printf(self.LOG_BODY, self.iter, self.spgIter, ...
                        self.nObjFunc, self.nProj, t, f, pgnrm, gtd / ...
                        (norm(g) * norm(d)));
                end
                
                % Check optimality conditions
                if pgnrm < self.rOptTol + self.aOptTol
                    self.iStop = self.EXIT_OPT_TOL;
                elseif max(abs(t * d)) < self.aFeasTol * norm(d) + ...
                        self.aFeasTol
                    self.iStop = self.EXIT_DIR_DERIV;
                elseif abs(f - fOld) < self.rFeasTol + self.aFeasTol
                    self.iStop = self.EXIT_FEAS_TOL;
                elseif self.nObjFunc > self.maxEval
                    self.iStop = self.EXIT_MAX_EVAL;
                elseif self.nProj > self.maxProj
                    self.iStop = self.EXIT_MAX_PROJ;
                elseif self.iter >= self.maxIter
                    self.iStop = self.EXIT_MAX_ITER;
                elseif toc(self.solveTime) >= self.maxRT
                    self.iStop = self.EXIT_MAX_RT;
                end
                
                if self.iStop % self.iStop ~= 0
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
            self.solveTime = toc(self.solveTime);
            % Set solved attribute
            self.isSolved();
            
            printObj.footer(self);
        end % solve
        
        function printf(self, varargin)
            %% Printf - prints variables arguments to a file
            fprintf(self.fid, varargin{:});
        end
        
    end % public methods
    
    
    methods (Access = private)
        
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
        
        function z = project(self, x)
            %% Project - projecting x on the constraint set
            z = self.nlp.project(x);
            if ~self.nlp.solved
                % Propagate throughout the program to exit
                self.iStop = self.EXIT_PROJ_FAILURE;
            end
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
            SpgOptions.aOptTol = self.aOptTol;
            SpgOptions.aFeasTol = self.aFeasTol;
            SpgOptions.testOpt = self.spgTestOpt;
            SpgOptions.useSpectral = self.spgUseSpectral;
            SpgOptions.projectLS = self.spgProjectLS;
            SpgOptions.bbType = self.spgBbType;
            SpgOptions.memory = self.spgMemory;
            SpgOptions.maxIter = self.maxIter;
            % Monotone linesearch
            SpgOptions.memory = 1;
            % Iterations left
            SpgOptions.maxIter = self.maxIter - self.iter;
            % Time left
            SpgOptions.maxRT = self.maxRT - toc(self.solveTime);
            
            % Building a quadratic approximation
            import model.ShiftedQpModel;
            quadModel = model.ShiftedQpModel('', xInit, x, g, H, self.nlp);
            
            % Solving using MinConf_SPG
            import solvers.SpgSolver
            subProblem = SpgSolver(quadModel, SpgOptions);
            subProblem = subProblem.solve();
            
            if ~subProblem.solved
                % SpgSolver exited without 'pseudo'-convergence
                self.iStop = self.EXIT_INNER_FAIL;
            end
            
            % Retrieving solution, number of proj calls & inner iterations
            self.spgIter = subProblem.iter;
            self.iter = self.iter + self.spgIter;
            self.nProj = self.nProj + subProblem.nProj;
            p = subProblem.x;
        end % solveSubProblem
        
        function [xNew, fNew, gNew, t] = backtracking(self, x, f, g, d, ...
                fOld, gtd)
            
            % Select Initial Guess to step length
            if self.iter == 1 || self.adjustStep == 0
                t = 1;
            else
                t = min(1, 2 * (f - fOld) / gtd);
            end
            % Bound Step length on first iteration
            if self.iter == 1
                t = min(1, 1 / sum(abs(g)));
            end
            
            iterLS = 1;
            while true
                
                % Evaluate the Objective and Gradient at the Initial Step
                xNew = x + t * d;
                [fNew, gNew] = self.nlp.obj(xNew);
                
                if fNew <= f + self.suffDec * g' * (xNew - x)
                    % Armijo condition met
                    return
                    % Check whether step has become too small
                elseif sum(abs(t * d)) < self.aFeasTol * norm(d) || ...
                        t == 0
                    self.iStop = self.EXIT_STEP_SIZE;
                    return;
                elseif iterLS >= self.maxIterLS
                    self.iStop = self.EXIT_MAX_ITER_LS;
                    return;
                end
                t = t/2;
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