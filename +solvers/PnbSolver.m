classdef PnbSolver < solvers.NlpSolver
    %% PnbSolver - Projected Newton algorithm for bounded optimization
    
    properties (SetAccess = private, Hidden = false)
        suffDec; % Sufficient decrease coefficient in line search
        maxIterLS; % Maximal number of iterations in the line search
        fid;
        lsFunc; % Line search function
        exactLS;
        descDirFunc;
        krylOpts;
    end
    
    properties (Hidden = true, Constant)
        LOG_HEADER = {'Iteration', 'FunEvals', 'fObj', '||Pg||', ...
            'nWorking'};
        LOG_FORMAT = '%10s %10s %15s %15s %10s\n';
        LOG_BODY = '%10d %10d %15.5e %15.5e %10d\n';
    end % constant properties
    
    
    methods (Access = public)
        
        function self = PnbSolver(nlp, varargin)
            %% Constructor
            
            % Gathering optional arguments and setting default values
            p = inputParser;
            p.KeepUnmatched = true;
            p.PartialMatching = false;
            p.addParameter('suffDec', 1e-4);
            p.addParameter('maxIterLS', 50); % Max iters for line search
            p.addParameter('exactLS', false);
            p.addParameter('fid', 1);
            %             p.addParameter('precond', false);
            p.addParameter('method', 'pcg');
            
            p.parse(varargin{:});
            
            self = self@solvers.NlpSolver(nlp, p.Unmatched);
            
            self.suffDec = p.Results.suffDec;
            self.maxIterLS = p.Results.maxIterLS;
            self.exactLS = p.Results.exactLS;
            self.fid = p.Results.fid;
            %             precond = p.Results.precond;
            
            % Setting MinRes, LSQR & LSMR's parameters
            self.krylOpts = struct;
            self.krylOpts.etol = self.aOptTol;
            self.krylOpts.rtol = self.aOptTol;
            self.krylOpts.atol = self.aOptTol;
            self.krylOpts.btol = self.aOptTol;
            self.krylOpts.shift = 0;
            self.krylOpts.show = false;
            self.krylOpts.check = false;
            self.krylOpts.itnlim = self.nlp.n;
            
            % Setting the descent direction computation function
            if strcmp(p.Results.method, 'lsqr')
                % LSQR - NlpModel must be a LeastSquaresModel!
                if  ~isa(nlp, 'model.LeastSquaresModel')
                    error(['nlp must be a model.LeastSquaresModel in', ...
                        ' order to use LSQR.']);
                end
                import krylov.lsqr_spot;
                self.descDirFunc = @(self, x, g, H, working) ...
                    self.callLsqr(x, working);
            elseif strcmp(p.Results.method, 'lsmr')
                % LSMR - NlpModel must be a LeastSquaresModel!
                if  ~isa(nlp, 'model.LeastSquaresModel')
                    error(['nlp must be a model.LeastSquaresModel in', ...
                        ' order to use LSMR.']);
                end
                import krylov.lsmr_spot;
                self.descDirFunc = @(self, x, g, H, working) ...
                    self.callLsmr(x, working);
            elseif strcmp(p.Results.method, 'minres')
                % MinRes
                import krylov.minres_spot;
                self.descDirFunc = @(self, x, g, H, working) ...
                    self.callMinres(g, H);
            else
                % Default to PCG
                self.descDirFunc = @(self, x, g, H, working) ...
                    self.callPcg(g, H);
            end
            
            % Exact line search is only implemented for quadratic or least
            % squares models
            if self.exactLS && ...
                    (isa(self.nlp, 'model.LeastSquaresModel') || ...
                    isa(self.nlp, 'model.QpModel'))
                self.lsFunc = @(xNew, x, f, g, d, H, working) ...
                    self.restrictedExact(xNew, x, f, g, d, H, working);
            else
                % Otherwise, default to Armijo
                self.lsFunc = @(xNew, x, f, g, d, H, working) ...
                    self.restrictedArmijo(xNew, x, f, g, d, H, working);
            end
            
            %             if strcmp(precond, 'precBCCB')
            %                 self.descDirFunc = @(g, H, working) ...
            %                     self.precNewtonDir(g, H, working);
            %             elseif strcmp(precond, 'precD')
            %                 self.descDirFunc = @(g, H, working) ...
            %                     self.precNewtonDir2(g, H, working);
            %             else
            %                 self.descDirFunc = @(g, H, working) ...
            %                     self.newtonDir(g, H);
            %             end
            
            import utils.PrintInfo;
        end % constructor
        
        function self = solve(self)
            %% Solve using the Projected Newton for bounds algorithm
            
            self.solveTime = tic;
            self.iter = 1;
            self.iStop = self.EXIT_NONE;
            self.nlp.resetCounters();
            
            printObj = utils.PrintInfo('Pnb');
            
            % Output Log
            if self.verbose >= 2
                extra = containers.Map( ...
                    {'suffDec', 'maxIterLS', 'exactLS'}, ...
                    {self.suffDec, self.maxIterLS, self.exactLS});
                printObj.header(self, extra);
                self.printf(self.LOG_FORMAT, self.LOG_HEADER{:});
            end
            
            % Project x0 to make sure it is a feasible point
            x = self.project(self.nlp.x0);
            
            % Getting obj. func, gradient and hessian at x
            [f, g, H] = self.nlp.obj(x);
            
            fOld = Inf;
            
            % Relative stopping tolerances
            self.gNorm0 = norm(g);
            rOptTol = self.rOptTol * self.gNorm0;
            rFeasTol = self.rFeasTol * abs(f);
            
            %% Main loop
            while ~self.iStop % self.iStop == 0
                
                % Get working set of variables
                working = self.getWorkingSet(x, g, H);
                
                % Stopping criteria is the norm of the 'working' gradient
                pgnrm = norm(g(working));
                
                % Output log
                self.nObjFunc = self.nlp.ncalls_fobj + ...
                    self.nlp.ncalls_fcon;
                if self.verbose >= 2
                    self.printf(self.LOG_BODY, self.iter, ...
                        self.nObjFunc, f, pgnrm, sum(working));
                end
                
                % Checking various stopping conditions, exit if true
                if ~any(working)
                    self.iStop = self.EXIT_ALL_BOUND;
                elseif pgnrm < rOptTol + self.aOptTol
                    self.iStop = self.EXIT_OPT_TOL;
                elseif abs(f - fOld) < rFeasTol + self.aFeasTol
                    self.iStop = self.EXIT_FEAS_TOL;
                elseif self.nObjFunc >= self.maxEval
                    self.iStop = self.EXIT_MAX_EVAL;
                elseif self.iter >= self.maxIter
                    self.iStop = self.EXIT_MAX_ITER;
                elseif toc(self.solveTime) >= self.maxRT
                    self.iStop = self.EXIT_MAX_RT;
                end
                
                if self.iStop % self.iStop ~= 0
                    break
                end
                
                % Saving fixed variables in xNew, x will be overwritten
                xNew = x;
                
                % We can truncate x, g and H as we are only working on the
                % free variables.
                fullX = x;
                x = x(working);
                g = g(working);
                H = H(working, working);
                
                % Compute Newton direction for free variables only
                d = self.descDirFunc(self, fullX, g, H, working);
                
                % Compute restricted projected Armijo line search
                xNew = self.lsFunc(xNew, x, f, g, d, H, working);
                
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
        
        function x = project(self, x)
            %% Project
            % Project on the bounds assuming x is full-sized.
            x = min(max(x, self.nlp.bL), self.nlp.bU);
        end
        
        function x = projectSel(self, x, ind)
            %% ProjectSel
            % Project on the bounds for a selected set of indices, assuming
            % x is of reduced size.
            x = min(self.nlp.bU(ind), max(x, self.nlp.bL(ind)));
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
            d = self.descDirFunc(self, x, ...
                g(~gFixed), H(~gFixed, ~gFixed), ~gFixed);
            
            x = x(~gFixed);
            fixed(~gFixed) = (x == self.nlp.bL(~gFixed) & d < 0) | ...
                (x == self.nlp.bU(~gFixed) & d > 0);
            working = ~fixed;
        end
        
        function d = callPcg(self, g, H)
            %% CallPcg
            % Solves the equation H * d = -g, using the gradient and
            % hessian provided as input arguments, assuming they are of
            % reduced size. This descent direction should only be computed
            % on the free variables.
            
            % Different methods could be used. Using PCG for now.
            [d, ~] = pcg(H, -g, self.aOptTol + self.rOptTol * self.gNorm0, self.nlp.n);
        end
        
        function d = callMinres(self, g, H)
            %% CallMinres
            % Solves the equation H * d = -g, using the gradient and
            % hessian provided as input arguments, assuming they are of
            % reduced size. This descent direction should only be computed
            % on the free variables.
            d = krylov.minres_spot(H, -g, self.krylOpts);
        end
        
        function d = callLsqr(self, x, working)
            %% CallLsqr
            % Solves the equation H * d = -g, using LSQR. Special functions
            % are defined in the LeastSquaresModel that allow a correct
            % call to LSQR & LSMR. The user must provide A and A*x-b from
            % the objective function ||A*x - b||^2.
            temp = -self.nlp.getResidual(x);
            d = krylov.lsqr_spot(self.nlp.A(:, working), temp, ...
                self.krylOpts);
        end
        
        function d = callLsmr(self, x, working)
            %% CallLsmr
            % Solves the equation H * d = -g, using LSQR. Special functions
            % are defined in the LeastSquaresModel that allow a correct
            % call to LSQR & LSMR. The user must provide A and A*x-b from
            % the objective function ||A*x - b||^2.
            temp = -self.nlp.getResidual(x);
            d = krylov.lsmr_spot(self.nlp.A(:, working), temp, ...
                self.krylOpts);
        end
        
        %         function d = precNewtonDir(self, g, H, working)
        %             %% NewtonDir - computes a Newton descent direction
        %             % Solves the equation H * d = -g, using the gradient and
        %             % hessian provided as input arguments, assuming they are of
        %             % reduced size. This descent direction should only be computed
        %             % on the free variables.
        %
        %             % Handle to hessian preconditionner
        %             precFunc = @(v) self.nlp.hessPrecBCCB(working, v);
        %
        %             % Different methods could be used. Using PCG for now.
        %             [d, ~] = pcg(H, -g, self.aOptTol + self.rOptTol, ...
        %                 max(1e4, self.nlp.n), precFunc);
        %         end
        %
        %         function d = precNewtonDir2(self, g, H, working)
        %             %% NewtonDir - computes a Newton descent direction
        %             % Solves the equation H * d = -g, using the gradient and
        %             % hessian provided as input arguments, assuming they are of
        %             % reduced size. This descent direction should only be computed
        %             % on the free variables.
        %
        %             % Handle to hessian preconditionner
        %             precFunc = @(v) self.nlp.hessPrecD(working, v);
        %
        %             % Different methods could be used. Using PCG for now.
        %             [d, ~] = pcg(H, -g, self.aOptTol + self.rOptTol, ...
        %                 max(1e4, self.nlp.n), precFunc);
        %         end
        
        function xNew = restrictedArmijo(self, xNew, x, f, g, d, ~, ...
                working)
            %% RestrictedArmijo - Armijo line search on the restricted vars
            % Perform a projected Armijo line search on the reduced
            % variables according to 'working'. This function assumes that
            % freeX, freeG and freeH are already reduced. However, xNew
            % must be full-sized since calls will be made to the objective
            % function.
            iterLS = 1;
            t = 1;
            while true
                % Recompute trial step on free variables
                xNew(working) = self.projectSel(x + t * d, working);
                % Update objective function value
                fNew = self.nlp.obj(xNew);
                % Checking exit conditions
                if (f - fNew) >= (self.suffDec * t * g' * d)
                    % Armijo condition satisfied
                    return;
                elseif iterLS >= self.maxIterLS
                    % Maximal number of iterations reached, abort
                    self.iStop = self.EXIT_MAX_ITER_LS;
                    return;
                end
                % Decrease step size
                t = t / 2;
                iterLS = iterLS + 1;
            end
        end
        
        function xNew = restrictedExact(self, xNew, x, ~, g, d, H, working)
            %% RestrictedExact
            % Exact line search restricted to the free variables. xNew is
            % projected at the end to ensure the value remains within the
            % bounds.
            % Step length must stay between 0 and 1
            t = min(max((d' * -g) / (d' * H * d), eps), 1);
            % Take step and project to make sure bounds are satisfied
            xNew(working) = self.projectSel(x + t * d, working);
        end
        
    end
    
end