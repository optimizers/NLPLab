classdef TMPSolver < solvers.NLPSolver
    %% TMPSolver - Calls the MinConf_TMP solver
    % An object oriented framework based on the original minConf solver
    % available at: https://www.cs.ubc.ca/~schmidtm/Software/minConf.html
    %
    % Solves bounded non-linear optimization problems:
    %
    %   min_x { f(x) : bL <= x <= bU
    %
    % The solver has been modified to support a nlp model as primary
    % argument. Note that the function 'obj' has to be defined for the nlp
    % object and must return the objective function value, the gradient and
    % the hessian at x. The number of output arguments must also be
    % variable.
    %
    % The 'method' property determines the way the descent direction is
    % computed. It can either be:
    %
    %   * 'minres': the true hessian of the objective function is used and
    %   must be an ouput argument of nlp's obj function. It must also be a
    %   Spot operator and has to be symmetric as MinRes will be used.
    %
    %   * 'lsqr': the true hessian of the objective function is used and
    %   must be an ouput argument of nlp's obj function. It must also be a
    %   Spot operator and has to be symmetric as LSQR will be used.
    %
    %   * 'newton': the true hessian of the objective function is used an
    %   must be an ouput argument of nlp's obj function. It has to be an
    %   explicit matrix.
    %
    %   * 'lbfgs': the BFGS approximation of the hessian is used, but with
    %   limited memory.
    %
    %   * 'bfgs': the BFGS approximation of the hessian is used.
    %
    %   * 'sd': the direction is simply the steepest descent.
    %
    %   * 'pcg': call MATLAB's pcg() on Hx = -g, where H must be positive
    %   definite.
    %
    % The possibility of using numerical derivatives has been removed.
    % Removed cubic interpolation from linesearch.
    %
    % Original documentation follows:
    % ---------------------------------------------------------------------
    % function [x,f] = minConF_TMP(funObj,x,LB,UB,options)
    %
    % Function for using Two-Metric Projection to solve problems of the
    % form:
    %   min funObj(x)
    %   s.t. LB_i <= x_i <= UB_i
    %
    %   @funObj(x): function to minimize (returns gradient as second
    %   argument)
    %
    %   options:
    %       verbose: level of verbosity (0: no output, 1: final, 2: iter
    %       (default), 3: debug)
    %       optTol: tolerance used to check for progress (default: 1e-7)
    %       maxIter: maximum number of calls to funObj (default: 250)
    %       numDiff: compute derivatives numerically (0: use user-supplied
    %       derivatives (default), 1: use finite differences, 2: use
    %       complex differentials)
    
    properties (SetAccess = private, Hidden = false)
        % Subclass of nlp model representing the problem to solve
        nlp;
        % x upon termination
        x;
        % Objective function value at x
        fx;
        % Norm of the projected gradient at x
        proj_grad_norm;
        % Iteration counter
        iter;
        % Execution time
        time_total;
        % Exit flag
        istop;
        % Convergence flag
        solved;
        % Number of calls to the objective function
        nObjFunc;
        % Tolerance relative to gradient norm
        stopTol;
        
        nf;
        ng;
        nh;
    end
    
    properties (Access = private, Hidden = false)
        % -- Internal parameters --
        verbose; % 0, 1 or 2
        optTol;
        maxIter;
        maxEval; % Maximum number of objective function evaluations
        suffDec;
        lsMaxIter; % Maximal number of iterations during linesearch
        method; % How to compute the descent direction
        corrections; % L-BFGS
        damped; % L-BFGS
        
        descDirTol; % Tolerance on computation of descent direction
        fid;
        krylOpts;
    end
    
    properties (Hidden = true, Constant)
        LOG_HEADER = {'Iteration','FunEvals', 'Step Length', ...
            'Function Val', '||Pg||', 'g''*d'};
        LOG_FORMAT = '%10s %10s %15s %15s %15s %15s\n';
        LOG_BODY = '%10d %10d %15.5e %15.5e %15.5e %15.5e\n';
        EXIT_MSG = { ...
            ['All variables are at their bound and no further', ...
            ' progress is possible at initial point\n'], ...            % 1
            ['All working variables satisfy optimality condition at', ...
            ' initial point\n'], ...                                    % 2
            'Directional derivative below optTol\n', ...                % 3
            ['All variables are at their bound and no further', ...
            ' progress is possible\n'], ...                             % 4
            'All working variables satisfy optimality condition\n', ... % 5
            'Step size below optTol\n', ...                             % 6
            'Function value changing by less than optTol\n', ...        % 7
            'Function Evaluations exceeds maxEval\n', ...              % 8
            'Maximum number of iterations reached\n'                    % 9
            };
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    methods (Access = public)
        function self = TMPSolver(nlp, varargin)
            %% Constructor
            % Inputs:
            %   - nlp: a subclass of a nlp model containing the 'obj'
            %   function that returns variable output arguments among the
            %   following: objective function, gradient and hessian at x.
            %   The hessian can be a Spot operator if it is too expensive
            %   to compute. The method also supports a L-BFGS approximation
            %   of the hessian.
            %   - varargin (optional): the various parameters of the
            %   algorithm
            
            if ~isa(nlp, 'model.nlpmodel')
                error('nlp should be a nlpmodel');
            end
            
            self = self@solvers.NLPSolver(nlp, varargin{:});
            
            % Gathering optional arguments and setting default values
            p = inputParser;
            p.addParameter('verbose', 2);
            p.addParameter('optTol', 1e-6);
            p.addParameter('maxIter', 5e2);
            p.addParameter('maxEval', 5e2);
            p.addParameter('suffDec', 1e-4);
            p.addParameter('method', 'pcg');
            p.addParameter('corrections', 7);
            p.addParameter('damped', 0);
            p.addParameter('fid', 1);
            p.addParameter('lsMaxIter', 50); % Max iters for linesearch
            p.addParameter('descDirTol', 1e-10);
            
            p.parse(varargin{:});
            
            self.verbose = p.Results.verbose;
            self.optTol = p.Results.optTol;
            self.maxIter = p.Results.maxIter;
            self.suffDec = p.Results.suffDec;
            self.lsMaxIter = p.Results.lsMaxIter;
            self.method = p.Results.method;
            self.corrections = p.Results.corrections;
            self.damped = p.Results.damped;
            self.fid = p.Results.fid;
            self.descDirTol = p.Results.descDirTol;
            
            if strcmp(self.method, 'minres') || strcmp(self.method, 'lsqr')
                self.krylOpts.rtol = self.descDirTol;
                self.krylOpts.etol = self.descDirTol;
                self.krylOpts.shift = 0;
                self.krylOpts.show = false;
                self.krylOpts.check = false;
            end
        end
        
        function self = solve(self)
            %% Solve
            % Solves the problem defined in the nlp model that was passed
            % as an argument to the constructor. Computes the descent
            % direction according to the 'method' parameter.
            
            self.time_total = tic;
            
            % Setting counters and exit flag, will exit if not 0
            self.istop = 0;
            self.nObjFunc = 0;
            self.iter = 1;
            
            % Output Log
            if self.verbose >= 2
                self.printHeaderFooter('header');
                self.printf(self.LOG_FORMAT, self.LOG_HEADER{:});
            end
            
            % Make sure initial point is feasible
            x = self.project(self.nlp.x0);
            
            if strcmp(self.method, 'newton') || ...
                    strcmp(self.method, 'minres') || ...
                    strcmp(self.method, 'pcg') || ...
                    strcmp(self.method, 'lsqr')
                [f, g, H] = self.obj(x);
                if strcmp(self.method, 'newton') && ~isnumeric(H)
                    error('Hessian must be explicit if newton is used');
                end
                % Checking if hessian is symmetric
                y = ones(self.nlp.n, 1);
                w = H * y;
                r2 = H * w;
                s = w' * w;
                t = y' * r2;
                z = abs(s - t);
                epsa = (s + eps) * eps^(1/3);
                if z > epsa
                    error(['Can''t use that method because hessian is', ...
                        ' not symmetric']);
                end
                secondOrder = 1;
            elseif strcmp(self.method, 'lbfgs') || strcmp(self.method, ...
                    'bfgs') || strcmp(self.method, 'sd')
                [f, g] = self.obj(x);
                secondOrder = 0;
            else
                error(['Unrecognized method to compute the descent', ...
                    ' direction (self.method)']);
            end
            
            self.stopTol = self.optTol * norm(g);
            
            % Compute working set (inactive constraints)
            working = self.working(x, g);
            
            % Early optimality check - if true won't enter loop
            pgnrm = norm(g(working));
            if isempty(working)
                self.istop = 1;
            elseif pgnrm <= self.stopTol + self.optTol
                self.istop = 2;
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %                       --- Main loop ---
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            while self.istop == 0
                % ---
                % Compute step direction
                % ---
                d = zeros(self.nlp.n, 1);
                switch self.method
                    case 'sd'
                        d(working) = -g(working);
                    case 'newton'
                        [d, H] = self.newtonDescDir(d, g, H, working);
                    case 'lsqr'
                        d(working) = lsqr_spot(H(working, working), ...
                            -g(working), self.krylOpts);
                    case 'minres'
                        d(working) = minres_spot(H(working, working), ...
                            -g(working), self.krylOpts);
                    case 'pcg'
                        % Double argout for pcg disables output message...
                        [d(working), ~] = pcg(H(working, working), ...
                            -g(working), self.descDirTol);
                    case 'lbfgs'
                        if self.iter == 1
                            % First iteration is steepest descent
                            d(working) = -g(working);
                            old_dirs = zeros(self.nlp.n, 0);
                            old_stps = zeros(self.nlp.n, 0);
                            Hdiag = 1;
                        else
                            [d, old_dirs, old_stps, Hdiag] = ...
                                self.lbfgsDescDir(d, x, x_old, g, ...
                                g_old, working, old_dirs, old_stps, Hdiag);
                        end
                        g_old = g;
                        x_old = x;
                    case 'bfgs'
                        if self.iter == 1
                            % First iteration is steepest descent
                            d(working) = -g(working);
                            B = speye(self.nlp.n);
                        else
                            [d, B] = self.bfgsDescDir(d, x, x_old, g, ...
                                g_old, B, working);
                        end
                        g_old = g;
                        x_old = x;
                end
                
                % Check that progress can be made along the direction
                gtd = g' * d;
                if gtd > -self.optTol * norm(g) * norm(d)
                    self.istop = 3;
                    % Leave now
                    break;
                end
                
                % Select initial guess to step length
                t = 1;
                if self.iter == 1 && ~secondOrder
                    t = min(1, 1 / sum(abs(g(working))));
                end
                
                % Evaluate the objective and projected gradient at the
                % initial step
                x_new = self.project(x + t * d);
                [f_new, g_new] = self.obj(x_new);
                
                % ---
                % Check sufficient decrease condition and do a linesearch
                % ---
                f_old = f;
                [x_new, f_new, g_new, t, lsFailed] = self.backtracking( ...
                    x, x_new, f, f_new, g, g_new, t, d);
                
                % Take Step
                x = x_new;
                f = f_new;
                g = g_new;
                
                % Compute Working Set
                working = self.working(x, g);
                
                % If necessary, compute Hessian
                if secondOrder && ~lsFailed
                    [~, ~, H] = self.obj(x);
                end
                
                % Optimality value
                pgnrm = norm(g(working)); % minConf's opt value
                
                % Output log
                if self.verbose >= 2
                    fprintf(self.LOG_BODY, self.iter, self.nObjFunc, ...
                        t, f, pgnrm, gtd);
                end
                
                % Checking various stopping conditions, exit if true
                if isempty(working)
                    self.istop = 4;
                elseif pgnrm <= self.stopTol + self.optTol
                    self.istop = 5;
                    % Check for lack of progress
                elseif sum(abs(t * d)) < self.optTol * norm(d)
                    self.istop = 6;
                elseif abs((f - f_old)/max([f_old, f, 1])) < self.optTol
                    self.istop = 7;
                elseif self.nObjFunc > self.maxEval
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
            self.time_total = toc(self.time_total);
            self.solved = ~(self.istop == 8 || self.istop == 9);
            
            if self.verbose
                self.printf('\nEXIT TMP: %s\nCONVERGENCE: %d\n', ...
                    self.EXIT_MSG{self.istop}, self.solved);
                self.printf('||Pg|| = %8.1e\n', self.proj_grad_norm);
                self.printf('Stop tolerance = %8.1e\n', self.stopTol);
            end
            
            self.nf = self.nlp.ncalls_fobj + self.nlp.ncalls_fcon;
            self.ng = self.nlp.ncalls_gobj + self.nlp.ncalls_gcon;
            self.nh = self.nlp.ncalls_hvp;
            
            if self.verbose >= 2
                self.printHeaderFooter('footer');
            end
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    methods (Access = private)
        
        function printHeaderFooter(self, msg)
            switch msg
                case 'header'
                    % -----------------------------------------------------
                    % Print header
                    % -----------------------------------------------------
                    self.printf('\n');
                    self.printf('%s\n', ['*', repmat('-',1,58), '*']);
                    self.printf([repmat('\t', 1, 3), 'MinConf_TMP \n']);
                    self.printf('%s\n\n', ['*', repmat('-',1,58), '*']);
                    self.printf(self.nlp.formatting())
                    self.printf('\nParameters\n----------\n')
                    self.printf('%-15s: %3s %8i', 'maxIter', '', self.maxIter);
                    
                    self.printf('\t%-15s: %3s %8.1e\n', ' optTol', '', ...
                        self.optTol);
                    self.printf('%-15s: %3s %8.1e', 'suffDec', '', ...
                        self.suffDec);
                    self.printf('\t%-15s: %3s %8d\n', ' lsMaxIter', '', ...
                        self.lsMaxIter);
                    self.printf('%-15s: %3s %8s', 'method', '', self.method);
                    self.printf('\t%-15s: %3s %8d\n', ' corrections', '', ...
                        self.corrections);
                    self.printf('%-15s: %3s %8d', 'damped', '', self.damped);
                    self.printf('\n');
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
                        'No. of Hessian-vector prods', self.nlp.ncalls_hvp);
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
                        '(%3d%%)\n'], 'Time: Hessian-vec prods', t1, t1t, ...
                        'total solve', tt, 100);
                otherwise
                    error('Unrecognized case in printHeaderFooter');
            end
        end
        
        function printf(self, varargin)
            %% Printf - prints variables arguments to a file
            fprintf(self.fid, varargin{:});
        end
        
        function x = project(self, x)
            %% Project - project x on the bounds
            % Upper and lower bounds are defined in nlp model
            x = min(max(x, self.nlp.bL), self.nlp.bU);
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
        
        function x = gpstep(self, x, g)
            %% GPStep - Evaluates the projection of the gradient
            % Used as a stopping criteria (?)
            x = self.project(x - g) - x;
        end
        
        function [x_new, f_new, g_new, t, failed] = backtracking(self, ...
                x, x_new, f, f_new, g, g_new, t, d)
            %% Backtracking Line Search
            % Applies an interpolation procedure (half step) if the new
            % value doesn't improve the value of the objective function.
            
            lsIters = 1;
            failed = false;
            % Check if decrease is sufficient
            while f_new > f + self.suffDec * g' * (x_new - x)
                if self.verbose == 2
                    fprintf('Halving Step Size\n');
                end
                t = 0.5 * t;
                
                % Check whether step has become too small
                if (sum(abs(t * d)) < self.optTol * norm(d)) || ...
                        (lsIters > self.lsMaxIter)
                    if self.verbose == 3
                        fprintf('Line Search failed\n');
                    end
                    % Return original x, f and g values
                    t = 0;
                    x_new = x;
                    f_new = f;
                    g_new = g;
                    failed = true;
                    return;
                end
                
                % Evaluate new point
                x_new = self.project(x + t * d);
                % Evaluate objective function and gradient at new point
                [f_new, g_new] = self.obj(x_new);
                lsIters = lsIters + 1;
            end
        end
        
        function [d, old_dirs, old_stps, Hdiag] = ...
                lbfgsDescDir(self, d, x, x_old, g, g_old, working, ...
                old_dirs, old_stps, Hdiag)
            %% LBFGSDescDir - descent direction computed using L-BFGS
            % Inputs:
            %   - d: descent direction that has to be updated.
            %   - f: current value of the objective function.
            %   - g: current value of the graident.
            %   - H: matrix/opSpot representing the hessian of the
            %   objective function.
            %   - working: logical array representing the current active
            %   constraints.
            % Ouput:
            %   - d: updated descent direction.
            %   - Various L-BFGS variables are passed as output in order to
            %   keep a handle on them:
            %       > old_dirs,
            %       > old_stps,
            %       > Hdiag
            
            % ***
            % All of this is done in MATLAB!
            % ***
            
            if self.damped
                % Updating hessian approximation using damped L-BFGS
                [old_dirs, old_stps, Hdiag] = dampedUpdate( ...
                    g - g_old, x - x_old, self.corrections, ...
                    self.verbose == 2, old_dirs, old_stps, Hdiag);
            else
                % Updating hessian approximation using L-BFGS
                [old_dirs, old_stps, Hdiag] = lbfgsUpdate(g - g_old, ...
                    x - x_old, self.corrections, ...
                    old_dirs, old_stps, Hdiag);
            end
            
            % Curvature criteria must remain greater than 0
            curvSat = sum(old_dirs(working, :) .* ...
                old_stps(working, :)) > 1e-10;
            
            % Computing d = -H^-1 * g, where H is the L-BFGS hessian approx
            d(working) = lbfgs(-g(working), old_dirs(working, ...
                curvSat), old_stps(working, curvSat), Hdiag);
        end
        
        function [d, B] = bfgsDescDir(self, d, x, x_old, ...
                g, g_old, B, working)
            %% BFGSDescDir - descent direction computed using BFGS
            % Inputs:
            %   - d: descent direction that has to be updated.
            %   - x: current point.
            %   - x_old: old point.
            %   - g: current value of the graident.
            %   - g_old: old gradient.
            %   - B: BFGS approximation of the hessian
            %   - working: logical array representing the current active
            %   constraints.
            % Ouput:
            %   - d: updated descent direction.
            %   - B: updated BFGS approximation of the hessian
            
            y = g - g_old;
            s = x - x_old;
            
            ys = y'*s;
            
            if self.iter == 2
                if ys > 1e-10
                    % If curvature condition is met, update hessian approx
                    B = ((y'*y) / (y'*s)) * B;
                end
            end
            if ys > 1e-10
                % If curvature condition is met, update hessian approx
                B = B + (y*y') / (y'*s) - (B*s*(s'*B)) / (s'*B*s);
            else
                if self.verbose == 2
                    fprintf('Skipping Update\n');
                end
            end
            
            % Updating descent direction
            d(working) = -B(working, working) \ g(working);
        end
        
        function [d, H] = newtonDescDir(self, d, g, H, working)
            %% NewtonDescDir - descent direction computed using Newton step
            % Inputs:
            %   - d: descent direction that has to be updated.
            %   - f: current value of the objective function.
            %   - g: current value of the graident.
            %   - H: matrix/opSpot representing the hessian of the
            %   objective function.
            %   - working: logical array representing the current active
            %   constraints.
            % Ouput:
            %   - d: updated descent direction.
            %   - H: updated hessian in case it was not definite positive.
            
            % Cholesky factorization of the hessian
            [R, posDef] = chol(H(working, working));
            
            if posDef == 0
                % If positive definite, solve directly
                d(working) = -R \ (R' \ g(working));
            else
                % If not, add smallest eigen value to the diagonal
                if self.verbose == 2
                    fprintf('Adjusting Hessian\n');
                end
                H(working, working) = H(working, working) + ...
                    speye(sum(working)) * max(0, 1e-12 - ...
                    min(real(eig(H(working, working)))));
                d(working) = -H(working,working) \ g(working);
            end
        end
        
        function working = working(self, x, g)
            %% Working - compute set of 'working' variables
            % true  = variable didn't reach its bound and can be improved
            working = true(self.nlp.n, 1);
            working((x < self.nlp.bL + self.optTol*2) & g >= 0) = false;
            working((x > self.nlp.bU - self.optTol*2) & g <= 0) = false;
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % ---------------------------------------------------------------------
    % Below are found the original L-BFGS update, hessian-vector product,
    % and hessian inverse functions found in the MinFunc folder of the
    % MinConf package.
    % ---------------------------------------------------------------------
    
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
        
        function [old_dirs, old_stps, Hdiag] = dampedUpdate(y, s, corr, ...
                old_dirs, old_stps, Hdiag)
            %% Original function from the MinFunc folder
            % Compute damped update in case curvature condition is not met
            % and hessian is not positive definite.
            
            S = old_dirs(:, 2:end);
            Y = old_stps(:, 2:end);
            k = size(Y, 2);
            L = zeros(k);
            for j = 1:k
                for i = j + 1:k
                    L(i, j) = S(:, i)' * Y(:, j);
                end
            end
            
            D = diag(diag(S' * Y));
            N = [(S / Hdiag), Y];
            M = [(S' * S / Hdiag), L; L', -D];
            
            ys = y' * s;
            Bs = s / Hdiag - N * (M \ (N' * s)); % Product B*s
            sBs = s' * Bs;
            
            eta = .02;
            if ys < eta * sBs
                theta = min(max(0, ((1 - eta) * sBs) / (sBs - ys)), 1);
                y = theta * y + (1 - theta) * Bs;
            end
            
            nCorr = size(old_dirs, 2);
            if nCorr < corr
                % Full Update
                old_dirs(:, nCorr + 1) = s;
                old_stps(:, nCorr + 1) = y;
            else
                % Limited-Memory Update
                old_dirs = [old_dirs(:, 2:corr), s];
                old_stps = [old_stps(:, 2:corr), y];
            end
            % Update scale of initial Hessian approximation
            Hdiag = (y' * s) / (y' * y);
        end
        
        function d = lbfgs(g, s, y, Hdiag)
            %% Original function from the MinFunc folder
            % BFGS Search Direction
            %
            % This function returns the (L-BFGS) approximate inverse
            % Hessian, multiplied by the gradient.
            %
            % If you pass in all previous directions/sizes, it will be the
            % same as full BFGS. If you truncate to the k most recent
            % directions/sizes, it will be L-BFGS.
            %
            % s - previous search directions (p by k)
            % y - previous step sizes (p by k)
            % g - gradient (p by 1)
            % Hdiag - value of initial Hessian diagonal elements (scalar)
            
            [p, k] = size(s);
            ro = zeros(k, 1);
            for i = 1:k
                ro(i,1) = 1 / (y(:, i)' * s(:, i));
            end
            
            q = zeros(p, k + 1);
            r = zeros(p, k + 1);
            al = zeros(k, 1);
            be = zeros(k, 1);
            q(:, k + 1) = g;
            
            for i = k:-1:1
                al(i) = ro(i) * (s(:, i)' * q(:, i + 1));
                q(:, i) = q(:, i+1) - al(i) * y(:, i);
            end
            
            % Multiply by Initial Hessian
            r(:,1) = Hdiag * q(:,1);
            for i = 1:k
                be(i) = ro(i) * (y(:, i)' * r(:, i));
                r(:, i + 1) = r(:, i) + s(:, i) * (al(i) - be(i));
            end
            d = r(:, k + 1);
        end
        
    end
end