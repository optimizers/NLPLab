classdef LBFGSSolver < solvers.NlpSolver
    %% LBFGSSolver
    
    
    properties (SetAccess = protected, Hidden = false)
        maxIterCg; % maximum number of CG iters per Newton step
        iterCg; % total number of CG iters
        cgTol;
        fMin;
        fid; % File ID of where to direct log output
        
        % Backtracking parameters
        fSuffDec;
        gSuffDec;
        maxIterLS;
        
        % L-BFGS data
        LbfgsMem;  % Size of the LBFGS history
        LbfgsUpdates; % Number of updates
        LbfgsRejects; % Number of rejected pairs
    end
    
    properties (SetAccess = protected, Hidden = true)
        % Parts of the L-BFGS operator
        
        beg;        % First index of stored pairs
        s;        % Array of s vectors
        y;        % Array of y vectors
        theta;    % Scaling parameter
        l;        % Matrix of the YtS
        sts;      % Matrix of the StS
        J;        % cholesky factorization of the middle matrix
        dd;       % Diagonal of the D matrix containing the ys
        sqrtdd    % Square root of dd
        LD;       %  L * D^(-1/2)
    end
    
    properties (Hidden = true, Constant)
        % Log header and body formats.
        LOG_HEADER_FORMAT = '\n%5s  %13s  %13s  %5s  %6s  %9s %9s\n';
        LOG_BODY_FORMAT = ['%5i  %13.6e  %13.6e  %5i', ...
            '  %6s  %9d %9f\n'];
        LOG_HEADER = {'iter', 'f(x)', '|g(x)|', 'cg', ...
            'status', 'nFree', 'time'};
    end % constant properties
    
    
    methods (Access = public)
        
        function self = LBFGSSolver(nlp, varargin)
            %% Constructor
            
            % Parse input parameters and initialize local variables
            p = inputParser;
            p.KeepUnmatched = true;
            p.PartialMatching = false;
            p.addParameter('maxIterCg', length(nlp.x0));
            p.addParameter('cgTol', 0.1);
            p.addParameter('fMin', -1e32);
            p.addParameter('fid', 1);
            p.addParameter('maxIterLS', 10);
            p.addParameter('fSuffDec', 1e-4);
            p.addParameter('gSuffDec', .5);
            p.addParameter('LbfgsMem', 29);
            
            p.parse(varargin{:});
            
            self = self@solvers.NlpSolver(nlp, p.Unmatched);
            
            % Store various objects and parameters
            self.cgTol     = p.Results.cgTol;
            self.maxIterCg = p.Results.maxIterCg;
            self.fMin      = p.Results.fMin;
            self.fid       = p.Results.fid;
            self.fSuffDec  = p.Results.fSuffDec;
            self.gSuffDec  = p.Results.gSuffDec;
            self.maxIterLS = p.Results.maxIterLS;
            self.LbfgsMem  = p.Results.LbfgsMem;
            
            % Initialize the L-BFGS operator
            self.initLBFGS();
            
            import utils.PrintInfo;
        end % constructor
        
        function self = solve(self)
            %% Solve
            
            self.solveTime = tic;
            self.iter = 1;
            self.iterCg = 1;
            self.iStop = self.EXIT_NONE;
            self.nlp.resetCounters();
            
            printObj = utils.PrintInfo('L-BFGS-B');
            
            if self.verbose >= 2
                extra = containers.Map({'fMin', 'cgTol'}, ...
                    {self.fMin, self.cgTol});
                printObj.header(self, extra);
                self.printf(self.LOG_HEADER_FORMAT, self.LOG_HEADER{:});
            end
            
            % Make sure initial point is feasible
            x = self.project(self.nlp.x0);
            
            % First objective and gradient evaluation.
            [f, g] = self.nlp.obj(x);
            
            % Initialize stopping tolerance and initial TR radius
            gNorm = norm(g);
            self.gNorm0 = gNorm;
            rOptTol = self.rOptTol * gNorm;
            rFeasTol = self.rFeasTol * abs(f);
            
            % Actual and predicted reductions. Initial inf value prevents
            % exits based on related on first iter.
            actRed = inf;
            
            % Miscellaneous iter
            status = '';
            
            %% Main loop
            while ~self.iStop
                % Check stopping conditions
                pgNorm = norm(self.gpstep(x, -1, g));
                now = toc(self.solveTime);
                if pgNorm <= rOptTol + self.aOptTol
                    self.iStop = self.EXIT_OPT_TOL;
                elseif f < self.fMin
                    self.iStop = self.EXIT_UNBOUNDED;
                elseif (abs(actRed) <= (self.aFeasTol + rFeasTol))
                    self.iStop = self.EXIT_FEAS_TOL;
                elseif self.iter >= self.maxIter
                    self.iStop = self.EXIT_MAX_ITER;
                elseif self.nlp.ncalls_fobj + self.nlp.ncalls_fcon >= ...
                        self.maxEval
                    self.iStop = self.EXIT_MAX_EVAL;
                elseif now >= self.maxRT
                    self.iStop = self.EXIT_MAX_RT;
                end
                
                % Print current iter to log
                if self.verbose >= 2
                    [~, nFree] = self.getIndFree(x);
                    self.printf(self.LOG_BODY_FORMAT, self.iter, f, ...
                        pgNorm, self.iterCg, status, nFree, now);
                end
                
                % Act on exit conditions
                if self.iStop
                    self.x = x;
                    self.fx = f;
                    self.pgNorm = pgNorm;
                    break
                end
                
                gOld = g;
                xOld = x;
                
                % Cauchy step
                [xc, c, indFree] = self.cauchy(x, g);
                
                % Subspace minimization
                [d, cgit, ~] = self.subspaceMinimization(x, g, xc, c, indFree);
                self.iterCg = self.iterCg + cgit;
                
                % Line search
                [x, f, g] = strongWolfe(self, x, f, g, d);                
                
                % Update L-BFGS operator
                self.updateLBFGS(x - xOld, g - gOld);
                
                self.iter = self.iter + 1;
            end % main loop
            
            self.nObjFunc = self.nlp.ncalls_fobj + self.nlp.ncalls_fcon;
            self.nGrad = self.nlp.ncalls_gobj + self.nlp.ncalls_gcon;
            self.nHess = self.nlp.ncalls_hvp + self.nlp.ncalls_hes;
            
            %% End of solve
            self.solveTime = toc(self.solveTime);
            % Set solved attribute
            self.isSolved();
            
            printObj.footer(self);
        end % solve
        
        function x = project(self, x, ind)
            %% Project
            % Project a vector onto the box defined by bL, bU.
            if nargin > 2
                x = min(self.nlp.bU(ind), max(x, self.nlp.bL(ind)));
            else
                x = min(self.nlp.bU, max(x, self.nlp.bL));
            end
        end
        
    end % public methods
     
     
    methods (Access = protected)
        
        function [indFree, nFree] = getIndFree(self, x)
            %% GetIndFree
            % Find the free variables
            indFree = (self.nlp.bL < x) & (x < self.nlp.bU);
            if nargout > 1
                nFree = nnz(indFree);
            end
        end

        function s = gpstep(self, x, alph, w, ind)
            %% GpStep - Compute the gradient projection step.
            % Compute the gradient projection step
            %
            % s = P[x + alph*w] - x,
            %
            % where P is the projection onto the box defined by bL, bU.
            if nargin > 4
                s = self.project(x + alph * w, ind) - x;
            else
                s = self.project(x + alph * w) - x;
            end
        end
        
        function breakpoints = brkpt(self, x, w, ind, nFree)
            %% BreakPt - Compute breakpoints along a search path
            %  Return the breakpoints for each coordinate on the piecewise
            %  affine path
            %
            %        x(t) = P(x + t * w)
            %  
            %  The output is a vector t such that for all i
            %
            %        t_i = max{t > 0 | x_i + t*w_i is in [bL_i, bU_i]}
            %
            %
            %  Input:
            %      - x: the starting point of the path
            %      - w: the search direction
            %      - ind: the reduced indices set, in case we use reduced
            %      variables (optional)
            %  Output:
            %      - breakpoints: the vector containing the breakpoint for
            %      each coordinate
            
            if nargin > 3
                bU = self.nlp.bU(ind);
                bL = self.nlp.bL(ind);
                breakpoints = inf(nFree, 1);
            else
                bU = self.nlp.bU;
                bL = self.nlp.bL;
                breakpoints = inf(self.nlp.n, 1);
            end
            
            % Compute gradient sign
            inc = w > 0;
            dec = w < 0;
            
            % Compute breakpoint for each coordinate
            breakpoints(inc) = ( bU(inc) - x(inc) ) ./ w(inc);
            breakpoints(dec) = ( bL(dec) - x(dec) ) ./ w(dec);
            breakpoints = max(breakpoints, 0);
        end
        
        function [breakpoints, d, F, indFree] = cauchyDirection(self, x, g)
            %% CauchyDirection
            %  Compute the breakpoints along the projected gradient path
            %
            %        x(alpha) = P(x - alpha * g)
            %
            %  Return a projected search direction for the Cauchy point and
            %  the indices of the positive breakpoints in this direction
            %  sorted from the smallest breakpoints to the greatest
            %  breakpoints.
            %  
            %  Inputs:
            %      - x: the point where we evaluate the gradient
            %      - g: the gradient of the objective function at x
            %  Outputs:
            %      - breakpoints: vector containing the breakpoints
            %      associated to each coordinate of g
            %      - d: the projected direction on the first segment
            %      - F: the indices of the sorted positive breakpoints
            %      - indFree: the unexposed constraints at x
            
            % Compute breakpoints along the projected gradient path
            breakpoints = self.brkpt(x, -g);
            
            % Compute a direction whose coordinates are zeros if the
            % associated constraints are exposed.
            indFree = (abs(breakpoints) >= eps);
            d = zeros(self.nlp.n, 1);
            d(indFree) = -g(indFree);
            
            % Compute the indices of the breakpoints sorted form the
            % smallest breakpoint to the greatest
            [~, F] = sort(breakpoints);
            F = F(sum(~indFree)+1:end);
        end
        
        function [xc, c, indFree] = cauchy(self, x, g)
            %% Cauchy - Compute the Cauchy Point
            %  Compute a Cauchy point by finding the first local minimum of
            %  the quadratic model
            %
            %        m(x) = 1/2*x'*B*x + g'*x
            %
            %  along the piecewise affine path
            %
            %        x(t) = P(x - tg)
            %
            %  Inputs:
            %        - x: the point where we are at the last iterate
            %        - g: the gradient of the objective function at x
            %  Outputs:
            %        - xc: the Cauchy point !
            %        - c:  a vector that is useful for the subspace
            %        minimization
            
            self.logger.debug('-- Entering Cauchy --');
            
            % Compute breakpoints in the gradient direction
            [breakpoints, d, F, indFree] = self.cauchyDirection(x, g);
            self.logger.debug(sprintf('brptMax = %7.1e', max(breakpoints)));
            self.logger.debug(sprintf('brptMin = %7.1e', min(breakpoints)));
            self.logger.debug(sprintf('brpt(F1) = %7.1e', breakpoints(F(1)) ));
            
            %% Initialization
            xc = x;
            iter = 1;
            
            % Starting point for the subspace minimization
            p = self.Wtimes(d, 2); % p = Wt*d
            c = zeros(size(p));
            
            % Function derivatives
            fp  = -(d.' * d);
            fpp = -self.theta * fp - p.' * self.Mtimes(p);
            
            % Function minimum on the segment
            deltaTMin = -fp / fpp;
            tOld = 0;
            
            % First breakpoint
            b = F(iter);
            t = breakpoints(b);
            deltaT = t;
            
            
            %% Examination of subsequent segments
            while deltaTMin >= deltaT
                iter = iter + 1;
                % Update Cauchy point
                if d(b) > 0
                    xc(b) = self.nlp.bU(b);
                elseif d(b) < 0
                    xc(b) = self.nlp.bL(b);
                end
                
                %Update active constraint
                indFree(b) = false;
                
                % Update c
                c  = c + deltaT * p;
                
                % We leave if all the constraints are active
                if iter > length(F)
                    return;
                end
                
                % Update directional derivatives
                zb = xc(b) - x(b);
                wbt = [self.y(b, self.beg:end), ...
                    self.theta * self.s(b, self.beg:end)];
                gb2 = g(b)^2;
                
                fp = fp + deltaT * fpp + gb2 + self.theta * g(b) * zb ...
                    - g(b) * ( wbt * self.Mtimes(c) );
                fpp = fpp - self.theta * gb2 ...
                    - 2 * g(b) * ( wbt * self.Mtimes(p) ) ...
                    - gb2 * ( wbt * self.Mtimes(wbt.') );
                
                % Update searching direction
                p = p + g(b) * wbt.';
                d(b) = 0;
                
                % Find new minimizer and breakpoint
                deltaTMin = - fp / fpp;
                tOld = t;
                b = F(iter);
                t = breakpoints(b);
                deltaT = t - tOld;
            end
            
            %% Final updates
            deltaTMin = max(deltaTMin, 0);
            tOld = tOld + deltaTMin;
            xc(b) = x(b) + tOld * d(b);
            xc(F) = x(F) + tOld * d(F);
            c = c + deltaTMin * p;
            
            self.logger.debug( sprintf('Iterations : %d', iter) );
            self.logger.debug( sprintf('nFree      : %d', sum(indFree) ));
            self.logger.debug('-- Leaving Cauchy --');
        end

        function [s, iter, info] = subspaceMinimization(self, x, g, xc, c, indFree)
            %% SubspaceMinimization - Minimize the quadratic on the free subspace
            %  Find the solution of the problem
            %
            %        min 1/2*x'*B*x + g'*x  s.t. x(~indFree) = xc(~indFree)
            %
            %  The solution is found by conjugate gradient
            %  Info : 0 - Convergence
            %         1 - Constraints are violated
            %         2 - Maximum number of iteration reached
            
            self.logger.debug('-- Entering Subspace minimization --');
            
            nFree = nnz(indFree);
            iterMax = nFree;
            iter = 0;
            
            % Initialize residual and descent direction
            r = g + self.theta * (x - xc) ...
                - self.Wtimes(self.Mtimes(c), 1);
            r = r(indFree);
            normRc = norm(r);
            self.logger.debug(sprintf('||rc|| = %9.3e', normRc));
            
            p = -r;
            d = zeros(length(r),1);
            rho2 = r.'*r;
            
            while true
                iter = iter + 1;
                
                % Check exit condition
                if sqrt(rho2) < ( min(0.1, sqrt(normRc)) * normRc )
                    info = 0;
                    self.logger.debug(sprintf('||r|| = %9.3e', sqrt(rho2)));
                    self.logger.debug('Exit CG: Convergence');
                    break;
                end
                if iter > iterMax
                    info = 2;
                    self.logger.debug('Exit CG: Max iteration number reached');
                    break;
                end
                    
                % Compute the minimal breakpoint
                alf1 = min( self.brkpt(xc(indFree) + d, p, indFree, nFree) );
                
                % Compute step length
                q = self.Btimes(p, indFree);
                alf2 = rho2 / (p.' * q);
                
                % Exit if we hit the domain boundary
                if alf2 > alf1
                    d = d + alf1 * p;
                    info = 1;
                    self.logger.debug('Exit CG: Constraints violated');
                    break;
                end
                
                % Prepare new step
                d = d + alf2 * p;
                r = r + alf2 * q;
                rho1 = rho2;
                rho2 = (r.' * r);
                beta = rho2 / rho1;
                p = -r + beta * p;
            end
            s = xc  -x;
            s(indFree) = s(indFree) + d;
            self.logger.debug(sprintf('||s|| = %9.3e', norm(s)));
            self.logger.debug(' -- Leaving Subspace minimization -- ');
        end
        
        function [x, f, g] = strongWolfe(self, x, f, g, d)
            %% StrongWolfe
            %  Wrapper to the More and Thuente line search
            %  It is necessary to have the files cvsrch.m and cstep.m on
            %  the path to use the line search
            %
            %  The subroutine finds a step that satisfies the strong Wolfe
            %  conditions whose parameters are defined in the solver.
            
            self.logger.debug('-- Entering strongWolfe --');
            
            % Set parameters
            dginit = g.' * d;
            self.logger.debug(sprintf('<d, g> = %7.1e', dginit));
            bptmin = min( self.brkpt(x, d) );
            stp = max(eps, min(1, bptmin));
            stpmin = 0;
            stpmax = max(eps, min( 1e10, bptmin ));
            
            self.logger.debug(sprintf('stpmax = %7.1e', stpmax));
            self.logger.debug(sprintf('Initial step = %7.1e', stp));
            
            % Call More and Thuente line search
            [x, f, g, ~, ~, ~] = cvsrch( ...
                @self.objParamsWrapper, ... Objective wrapper
                x, [], f, g, ... Objective value at x
                d, stp, dginit, ... Search direction and recommended step
                self.fSuffDec, self.gSuffDec, eps, ...
                stpmin, stpmax, Inf);
            
            self.logger.debug(sprintf('  Final step = %7.1e', stp));
            self.logger.debug('-- Leaving strongWolfe --');
            
        end
        
        function [f, g, params] = objParamsWrapper(self, x, ~)
            %% fParamsWrapper
            %  Wrapper to call the objective function in the line search 
            %  subroutine
            [f, g] = self.nlp.obj(x);
            params = [];
        end
            
        function initLBFGS(self)
            %% InitLBFGS - Init the LBFGS Matrix
            %  This subroutine is called in the constructor to allocate
            %  memory for the limited memory BFGS operator.
            
            % Initialize parameters
            self.LbfgsUpdates = 0;
            self.LbfgsRejects = 0;
            self.beg = self.LbfgsMem + 1;
            self.theta = 1;
            
            % Initialize arrays
            self.s = zeros(self.nlp.n, self.LbfgsMem);
            self.y = zeros(self.nlp.n, self.LbfgsMem);
            self.l = zeros(self.LbfgsMem);
            self.dd = zeros(self.LbfgsMem, 1);
            self.sqrtdd = zeros(self.LbfgsMem, 1);
            self.sts = zeros(self.LbfgsMem);
            self.J = [];
            self.LD = [];
        end
        
        function updateLBFGS(self, s, y)
            %% UpdateLBFGS - Add a new pair to the pseudohessian
            % Store the new pair {y, s} into the L-BFGS approximation
            % Discard the oldest pair if memory has been exceeded
            % The matrices D, L, StS and J are also updated
            
            self.LbfgsUpdates = self.LbfgsUpdates + 1;
            ys = dot(y, s);
            
            if ys <= eps * dot(y, y)
                warning('L-BFGS: Rejecting {s, y} pair');
                self.LbfgsRejects = self.LbfgsRejects + 1;
            else
                % Update S and Y
                self.s = [self.s(:, 2:end), s];
                self.theta = (y' * y) / ys;
                self.y = [self.y(:, 2:end), y];
                    
                if self.beg > 1
                    self.beg = self.beg - 1;
                end
                
                % Update D
                self.dd = [self.dd(2:end); ys];
                self.sqrtdd = [self.sqrtdd(2:end); sqrt(ys)];
                
                % Update StS
                v = self.s.' * s; % If prec self.s = B0 * S
                self.sts = [self.sts(2:end, 2:end), v(1:(end-1)) ; v.'];
                
                % Update L
                v = s.' * self.y(:, 1:(end-1));
                self.l = [self.l(2:end, 2:end), zeros(self.LbfgsMem-1,1); ...
                    v, 0];
                
                % Update J and LD (forming the middle matrix)
                nPairs = self.LbfgsMem - self.beg + 1;
                L = self.l(self.beg:end, self.beg:end);
                D = spdiags(self.dd(self.beg:end), nPairs, nPairs);
                sqD = spdiags(self.sqrtdd(self.beg:end), nPairs, nPairs);
                self.J = chol(self.theta * self.sts(self.beg:end, self.beg:end) ...
                    + L * (D \ L.'), ...
                    'lower');
                self.LD = sqD \ L;
            end
        end
        
        function p = Wtimes(self, v, mode, ind)
            %% WTimes - Apply the W matrix
            %  Compute the product by W = [Y, theta * S];
            %
            %  mode = 1: Direct product
            %  v is of size 2*nPairs
            %
            %  mode = 2: Adjoint product
            %  v is of size n
            if nargin < 4
                if mode == 1 % Direct product
                    nPairs = self.LbfgsMem - self.beg + 1;
                    p = self.y(:, self.beg:end) * v(1:nPairs) ...
                        + self.theta * self.s(:, self.beg:end) ...
                        * v((nPairs+1):(2*nPairs), :);
                elseif mode == 2
                    p = [self.y(:, self.beg:end).' * v; ...
                        self.theta * self.s(:, self.beg:end).' * v];
                end
            else % Reduced vectors
                if mode == 1 % Direct product
                    nPairs = self.LbfgsMem - self.beg + 1;
                    p = self.y(ind, self.beg:end) * v(1:nPairs) ...
                        + self.theta * self.s(ind, self.beg:end) ...
                        * v((nPairs+1):(2*nPairs), :);
                elseif mode == 2
                    p = [self.y(ind, self.beg:end).' * v; ...
                        self.theta * self.s(ind, self.beg:end).' * v];
                end
            end
        end
        
        function p = Mtimes(self, v)
            %% MTimes - Apply the M matrix
            %  Compute the product with the middle matrix of the L-BFGS
            %  formula
            nPairs = self.LbfgsMem - self.beg + 1;
            sqrtD = spdiags( self.sqrtdd(self.beg:end), 0, nPairs, nPairs );
            
            p = [sqrtD, zeros(nPairs) ; -self.LD, self.J] \ v;
            p = [-sqrtD, self.LD.' ; zeros(nPairs), self.J.'] \ p;
        end
        
        function p = Btimes(self, v, ind)
            %% BTimes - Apply the pseudohessian
            %  Compute the product by the L-BFGS hessian approximation
            if nargin < 3
                p = self.Wtimes(v, 2);
                p = self.Mtimes(p);
                p = self.theta * v - self.Wtimes(p, 1);
            else % Reduced vectors
                p = self.Wtimes(v, 2, ind);
                p = self.Mtimes(p);
                p = self.theta * v - self.Wtimes(p, 1, ind);
            end
        end
                
    end % private methods
    
    
    methods (Access = public, Hidden = true)
        
        function printf(self, varargin)
            fprintf(self.fid, varargin{:});
        end
        
    end % hidden public methods
    
end % class