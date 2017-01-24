classdef PnSolver < handle
    %% PnSolver - Projected Newton optimization algorithm on custom set
    
    
    properties (Access = public)
        % Execution time
        solveTime;
    end
    
    properties (SetAccess = private, Hidden = false)
        % NlpModel
        nlp;
        % x upon termination
        x;
        % Objective function value at x
        fx;
        % Norm of projected gradient at x
        pgNorm;
        % Iteration counter
        iter;
        % Exit flag
        iStop;
        % Solved flag
        solved;
        % Objective function calls counter
        nObjFunc;
        nGrad;
        nHess;
        % optTol relative to the initial gradient norm
        stopTol;
        relFuncTol;
        memory = 10;
    end
    
    properties (Access = private, Hidden = false)
        verbose; % 0, 1 or 2
        % Note: optTol := || P[x - g] - x ||
        optTol; % Optimality tolerance in the main problem
        funcTol;
        maxIter; % Maximum number of iterations
        maxEval; % Maximum number of calls to objective function
        suffDec; % Sufficient decrease coefficient in line search
        maxIterLS; % Maximal number of iterations in the line search
        fid;
        eqTol;
        lsFunc; % Line search function
        cgIter;
    end
    
    properties (Hidden = true, Constant)
        LOG_HEADER = {'Iteration', 'CG iter', 'FunEvals', 'fObj', ...
            '||Pg||'};
        LOG_FORMAT = '%10s %10s %10s %15s %15s \n';
        LOG_BODY = '%10d %10d %10d %15.5e %15.5e\n';
        EXIT_MSG = { ...
            'All working variables satisfy optimality condition\n', ... % 1
            'Function value changing by less than funcTol\n', ...       % 2
            'Function Evaluations exceeds maxEval\n', ...               % 3
            'Maximum number of iterations reached\n', ...               % 4
            'Maximum number of iterations in line search reached\n', ...% 5
            'Projected CG exit without convergence\n', ...              % 6
            };
        CG_TOL = 1e-10;
    end % constant properties
    
    
    methods (Access = public)
        
        function self = PnSolver(nlp, varargin)
            %% Constructor
            if ~isa(nlp, 'model.NlpModel')
                error('Model must be a NlpModel');
            elseif ~ismethod(nlp, 'project')
                error('nlp doesn''t contain a project method');
            end
            self.nlp = nlp;
            
            % Gathering optional arguments and setting default values
            p = inputParser;
            p.KeepUnmatched = true;
            p.PartialMatching = false;
            p.addParameter('verbose', 2);
            p.addParameter('optTol', 1e-5);
            p.addParameter('maxIter', 5e2);
            p.addParameter('maxEval', 5e2);
            p.addParameter('suffDec', 1e-4);
            p.addParameter('maxIterLS', 50); % Max iters for line search
            p.addParameter('funcTol', eps);
            p.addParameter('exactLS', false);
            p.addParameter('fid', 1);
            
            p.parse(varargin{:});
            self.verbose = p.Results.verbose;
            self.optTol = p.Results.optTol;
            self.maxIter = p.Results.maxIter;
            self.maxEval = p.Results.maxEval;
            self.suffDec = p.Results.suffDec;
            self.maxIterLS = p.Results.maxIterLS;
            self.funcTol = p.Results.funcTol;
            self.fid = p.Results.fid;
        end % constructor
        
        function self = solve(self)
            %% Solve using the Projected Newton for bounds algorithm
            self.solveTime = tic;
            self.iter = 1;
            self.iStop = 0;
            self.cgIter = 0;
            
            % Output Log
            if self.verbose >= 2
                self.printHeaderFooter('header');
                self.printf(self.LOG_FORMAT, self.LOG_HEADER{:});
            end
            
            % Project x0 to make sure it is a feasible point
            x = self.nlp.project(self.nlp.x0);
            
            % Getting obj. func, gradient and hessian at x
            [f, g, H] = self.nlp.obj(x);
            
            fOld = Inf;
            
            % Relative stopping tolerance
            self.stopTol = self.optTol * norm(g);
            self.relFuncTol = self.funcTol * abs(f);
            
            %% Main loop
            while self.iStop == 0
                
                % Stopping criteria is the norm of the 'working' gradient
                pgnrm = norm(self.nlp.project(x - g) - x);
                
                % Output log
                self.nObjFunc = self.nlp.ncalls_fobj + ...
                    self.nlp.ncalls_fcon;
                if self.verbose >= 2
                    fprintf(self.LOG_BODY, self.iter, self.cgIter, ...
                        self.nObjFunc, f, pgnrm);
                end
                
                % Checking various stopping conditions, exit if true
                if pgnrm < self.stopTol + self.optTol
                    self.iStop = 1;
                elseif abs(f - fOld) < self.relFuncTol + self.funcTol
                    self.iStop = 2;
                elseif self.nObjFunc >= self.maxEval
                    self.iStop = 3;
                elseif self.iter >= self.maxIter
                    self.iStop = 4;
                end
                
                if self.iStop ~= 0
                    break
                end
                
                % Compute descent direction
                [p, info] = self.projCg2(H*x - g, H, norm(g), x);
                if info ~= 1
                    self.iStop = 6;
                    break;
                end
                d = x - p;

                % Compute Armijo line search
                x = self.armijo(x, f, g, d);
                
                % Saving old f to check progression
                fOld = f;
                
                % Updating objective function, gradient and hessian
                [f, g, H] = self.nlp.obj(x);
                
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
            self.solved = ~(self.iStop == 8 || self.iStop == 9);
            
            if self.verbose
                self.printf('\nEXIT PN: %s\nCONVERGENCE: %d\n', ...
                    self.EXIT_MSG{self.iStop}, self.solved);
                self.printf('||Pg|| = %8.1e\n', self.pgNorm);
                self.printf('Stop tolerance = %8.1e\n', self.stopTol);
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
                    self.printf([repmat('\t', 1, 3), 'Pn Solver \n']);
                    self.printf('%s\n\n', ['*', repmat('-',1,58), '*']);
                    self.printf(self.nlp.formatting())
                    self.printf('\nParameters\n----------\n')
                    self.printf('%-15s: %3s %8i', 'maxIter', '', ...
                        self.maxIter);
                    self.printf('\t%-15s: %3s %8.1e\n', ' optTol', '', ...
                        self.optTol);
                    self.printf('%-15s: %3s %8.1e', 'suffDec', '', ...
                        self.suffDec);
                    self.printf('\t%-15s: %3s %8d\n', ' maxIterLS', '', ...
                        self.maxIterLS);
                    self.printf('\n');
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
                        self.nlp.ncalls_hvp + self.nlp.ncalls_hes);
                    self.printf('\n');
                    tt = self.solveTime;
                    t1 = self.nlp.time_fobj + self.nlp.time_fcon;
                    t1t = round(100 * t1/tt);
                    t2 = self.nlp.time_gobj + self.nlp.time_gcon;
                    t2t = round(100 * t2/tt);
                    self.printf([' %-24s %6.2f (%3d%%)  %-20s %6.2f', ...
                        '(%3d%%)\n'], 'Time: function evals' , t1, t1t, ...
                        'gradient evals', t2, t2t);
                    t1 = self.nlp.time_hvp + self.nlp.time_hes;
                    t1t = round(100 * t1/tt);
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
        
        function [w, info] = projCg2(self, g, H, gNrm, x)
            %% ProjCg
            %       min_w   q(w) := 0.5*(w - x)'*H*(w - x) + g'*(w - x)
            %       st      w \in X
            %
            % On exit info is set as follows:
            %
            %       info = 1  Convergence in the original variables.
            %       info = 2  Failure to converge within iterMax iterations
            
            % Initialize the iterate w and the residual r.
            w = x;
            % Initialize the residual r of grad q to -g.
            r = g;
            % Initialize the direction p.
            p = -r;
            rTol = max(gNrm * self.CG_TOL, 1e-8);
            iterMax = max(1e4, self.nlp.n);
            for iters = 1:iterMax
                % Initialize rho and the norms of r.
                rho = r'*r;
                % Project on the constraint set
                p = self.nlp.project(p);
                % Compute alph
                Hp = H * p;
                ptHp = p'*Hp;
                alph = rho/ptHp;
                % Update w and the residuals r.
                w = w + alph*p;
                r = r + alph*Hp;
                % Exit if the residual convergence test is satisfied.
                rtr = r'*r;
                if sqrt(rtr) <=  rTol
                    info = 1;
                    self.cgIter = iters;
                    return
                end
                % Compute p = r + beta*p and update rho.
                p = -r + rtr/rho * p;
            end % for loop
            self.cgIters = iters;
            info = 2;
        end % projcg
        
        function [w, info] = projCg(self, g, H, gNrm, x)
            %% ProjCg
            %       min_w   q(w) := 0.5*(w - x)'*H*(w - x) + g'*(w - x)
            %       st      w \in X
            %
            % On exit info is set as follows:
            %
            %       info = 1  Convergence in the original variables.
            %       info = 2  Negative curvature direction generated.
            %       info = 3  Failure to converge within iterMax iterations
            
            % Initialize the iterate w and the residual r.
            w = zeros(self.nlp.n, 1);
            % Initialize the residual r of grad q to -g.
            r = -g;
            % Initialize the direction p.
            p = r;
            % Initialize rho and the norms of r.
            rho = r'*r;
            
            rTol = gNrm * self.CG_TOL;
            
            % Exit if g = 0.
            if sqrt(rho) == 0
                info = 1;
                return
            end
            
            iterMax = max(1e4, self.nlp.n);
            for iters = 1:iterMax
                % Project on the constraint set
                p = self.nlp.project(p);
                % Compute alph
                q = H * p;
                ptq = p'*q;
                if ptq > 0
                    % Positive curvature, continue
                    alph = rho/ptq;
                else
                    % Negative curvature
                    info = 2;
                    return
                end
                % Update w and the residuals r.
                w = w + alph*p;
                r = r - alph*q;
                % Exit if the residual convergence test is satisfied.
                rtr = r'*r;
                rnorm = sqrt(rtr);
                if rnorm <=  rTol
                    info = 1;
                    return
                end
                % Compute p = r + betaFactor*p and update rho.
                betaFactor = rtr/rho;
                p = r + betaFactor * p;
                rho = rtr;
            end % for loop
            info = 3;
        end % projcg
        
        function xNew = armijo(self, x, f, g, d)
            %% Armijo - Armijo line search
            % Perform an Armijo line search on the reduced
            % variables according to 'working'. This function assumes that
            % freeX, freeG and freeH are already reduced. However, xNew
            % must be full-sized since calls will be made to the objective
            % function.
            iterLS = 1;
            t = 1;
            while true
                % Recompute trial step on free variables
                xNew = x + t * d;
                % Update objective function value
                fNew = self.nlp.obj(xNew);
                % Checking exit conditions
                if (f - fNew) >= (self.suffDec * t * g' * d)
                    % Armijo condition satisfied
                    return;
                elseif iterLS >= self.maxIterLS
                    % Maximal number of iterations reached, abort
                    self.iStop = 5;
                    break;
                end
                % Decrease step size
                t = t / 2;
                iterLS = iterLS + 1;
            end
        end
        
    end
    
end