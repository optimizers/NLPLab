classdef PnbSolver < handle
    %% PnbSolver - Projected Newton algorithm for bounded optimization
    
    
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
    end
    
    properties (Hidden = true, Constant)
        LOG_HEADER = {'Iteration', 'FunEvals', 'fObj', '||Pg||', ...
            'nWorking'};
        LOG_FORMAT = '%10s %10s %15s %15s %10s\n';
        LOG_BODY = '%10d %10d %15.5e %15.5e %10d\n';
        EXIT_MSG = { ...
            ['All variables are at their bound and no further', ...
            ' progress is possible\n'], ...                             % 1
            'All working variables satisfy optimality condition\n', ... % 2
            'Function value changing by less than funcTol\n', ...        % 3
            'Function Evaluations exceeds maxEval\n', ...               % 4
            'Maximum number of iterations reached\n', ...               % 5
            'Maximum number of iterations in line search reached\n', ...% 6
            };
    end % constant properties
    
    
    methods (Access = public)
        
        function self = PnbSolver(nlp, varargin)
            %% Constructor
            if ~isa(nlp, 'model.NlpModel')
                error('Model must be a NlpModel');
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
            %% Solve using the PQN-NNLS algorithm
            self.solveTime = tic;
            self.iter = 1;
            self.iStop = 0;
            
            % Output Log
            if self.verbose >= 2
                self.printHeaderFooter('header');
                self.printf(self.LOG_FORMAT, self.LOG_HEADER{:});
            end
            
            % Project x0 to make sure it is a feasible point
            x = self.project(self.nlp.x0);
            
            % Getting obj. func, gradient and hessian at x
            [f, g, H] = self.nlp.obj(x);
            
            fOld = Inf;
            
            % Relative stopping tolerance
            self.stopTol = self.optTol * norm(g);
            self.relFuncTol = self.funcTol * abs(f);
            
            %% Main loop
            while self.iStop == 0
                
                % Get working set of variables
                working = self.getWorkingSet(x, g, H);
                
                % Stopping criteria is the norm of the 'working' gradient
                pgnrm = norm(g(working));
                
                % Output log
                self.nObjFunc = self.nlp.ncalls_fobj + self.nlp.ncalls_fcon;
                if self.verbose >= 2
                    fprintf(self.LOG_BODY, self.iter, self.nObjFunc, f, ...
                        pgnrm, sum(working));
                end
                
                % Checking various stopping conditions, exit if true
                if isempty(working)
                    self.iStop = 1;
                elseif pgnrm < self.stopTol + self.optTol
                    self.iStop = 2;
                elseif abs(f - fOld) < self.relFuncTol + self.funcTol
                    self.iStop = 3;
                elseif self.nObjFunc >= self.maxEval
                    self.iStop = 4;
                elseif self.iter >= self.maxIter
                    self.iStop = 5;
                end
                
                if self.iStop ~= 0
                    break
                end
                
                % Saving fixed variables in xNew, x will be overwritten
                xNew = x;
                
                % We can truncate x, g and H as we are only working on the
                % free variables.
                x = x(working);
                g = g(working);
                H = H(working, working);
                
                % Compute Newton direction for free variables only
                d = self.newtonDir(g, H);
                
                % Trial step P[x + d] on working variables only
                xNew(working) = self.projectSel(x - d, working);
                
                % Compute restricted projected Armijo line search
                xNew = self.restrictedArmijo(xNew, f, x, g, d, working);
                
                % Taking step, restore x to full-sized
                x = xNew;
                
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
                self.printf('\nEXIT PNB: %s\nCONVERGENCE: %d\n', ...
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
                    self.printf([repmat('\t', 1, 3), 'Pnb Solver \n']);
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
        
        function z = project(self, x)
            %% Project
            % Project on the bounds assuming x is full-sized.
            z = min(max(x, self.nlp.bL), self.nlp.bU);
        end
        
        function z = projectSel(self, x, ind)
            %% ProjectSel
            % Project on the bounds for a selected set of indices, assuming
            % x is of reduced size.
            z = min(max(x, self.nlp.bL(ind)), self.nlp.bU(ind));
        end
        
        function working = getWorkingSet(self, x, g, H)
            %% GetWorkingSet - Finds the fixed set for proj. Newton step
            % Inputs:
            %   - x: current point
            %   - g: gradient at current point
            %   - H: hessian at current point
            % Outputs:
            %   - working: bool array of free variables
            
            % Find gradient fixed set
            gFixed = (x == self.nlp.bL & g > 0) | ...
                (x == self.nlp.bU & g < 0);
            
            % Save gradient fixed set
            fixed = gFixed;
            
            % However, we will later compute a Newton direction instead of
            % a steepest descent direction. Therefore, computing a Newton
            % direction on the remaining free variables can help us
            % identify more variables that should be fixed.
            
            % Compute a Newton direction from reduced g & H
            d = self.newtonDir(g(~gFixed), H(~gFixed, ~gFixed));
            
            % We restrict x to the free variables
            x = x(~gFixed);
            
            % Update the gradient fixed set with the Newton fixed set
            % fixed := gradient fixed set | Newton fixed set
            fixed(~gFixed) = (x == self.nlp.bL(~gFixed) & d > 0) | ...
                (x == self.nlp.bU(~gFixed) & d < 0);
            
            % Finally, the working set represents the free variables
            working = ~fixed;
        end
        
        function d = newtonDir(self, g, H)
            %% NewtonDir - computes a Newton descent direction
            % Solves the equation H * d = -g, using the gradient and hessian
            % provided as input arguments, assuming they are of reduced
            % size. This descent direction should only be computed on the
            % free variables.
            
            % Different methods could be used. Using PCG for now.
            [d, ~] = pcg(H, g, max(1e-5 * self.optTol, 1e-12));
        end
        
        function xNew = restrictedArmijo(self, xNew, f, x, g, d, working)
            %% RestrictedArmijo - Armijo line search on the restricted vars
            % Perform a projected Armijo line search on the reduced
            % variables according to 'working'. This function assumes that
            % freeX, freeG and freeH are already reduced. However, xNew
            % must be full-sized since calls will be made to the objective
            % function.
            
            iterLS = 1;
            t = 1;
            fNew = self.nlp.obj(xNew);
            
            while true
                
                if (f - fNew) >= (self.suffDec * t * g' * d)
                    % Armijo condition satisfied
                    return;
                elseif iterLS >= self.maxIterLS
                    % Maximal number of iterations reached, abort
                    self.iStop = 6;
                    return;
                end
                
                % Decrease step size
                t = t / 2;
                % Recompute trial step on free variables
                xNew(working) = self.projectSel(x - t * d, working);
                % Update objective function value
                fNew = self.nlp.obj(xNew);
                
                iterLS = iterLS + 1;
            end
        end
        
    end
    
end