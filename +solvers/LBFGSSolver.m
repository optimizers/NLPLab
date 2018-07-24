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
        
        % L-BFGS operator
        B; % opSpot on the pseudo hessian
    end
    
    properties (Hidden = true, Constant)
        % Constants used to manipulate the TR radius. These are the numbers
        % used by TRON.
        sig1 = 0.25;
        sig2 = 0.50;
        sig3 = 4.00;
        eta0 = 1e-4;
        eta1 = 0.25;
        eta2 = 0.75;
        
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
            p.addParameter('LbfgsPairs', 29);
            
            p.parse(varargin{:});
            
            self = self@solvers.NlpSolver(nlp, p.Unmatched);
            
            % Store various objects and parameters
            self.cgTol = p.Results.cgTol;
            self.maxIterCg = p.Results.maxIterCg;
            self.fMin = p.Results.fMin;
            self.fid = p.Results.fid;
            self.fSuffDec = p.Results.fSuffDec;
            self.gSuffDec = p.Results.gSuffDec;
            self.maxIterLS = p.Results.maxIterLS;
            
            % Initialize the L-BFGS operator
            import utils.opLBFGSB;
            self.B = opLBFGSB(self.nlp.n, p.Results.LbfgsPairs);
            
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
                [d, ~, ~] = self.subspaceMinimization(x, g, xc, c, indFree);
                
                % Line search
                [x, f, g] = strongWolfe(self, x, f, g, d);                
                
                % Update L-BFGS operator
                self.B = update(self.B, x - xOld, g - gOld);
                
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
                nFree = sum(indFree);
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
        
        function breakpoints = brkpt(self, x, w, ind)
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
            else
                bU = self.nlp.bU;
                bL = self.nlp.bL;
            end
            
            % Compute gradient sign
            inc = w > 0;
            dec = w < 0;
            
            % Compute breakpoint for each coordinate
            breakpoints= inf(self.nlp.n, 1);
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
            p = Wtimes(self.B, d, 2); % p = Wt*d
            c = zeros(size(p));
            
            % Function derivatives
            fp  = -(d.' * d);
            fpp = -self.B.theta * fp - p.' * Mtimes(self.B, p);
            
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
                wbt = Wb(self.B, b);
                gb2 = g(b)^2;
                fp = fp + deltaT * fpp + gb2 + self.B.theta * g(b) * zb ...
                    - g(b) * ( wbt * Mtimes(self.B, c, 1) );
                fpp = fpp - self.B.theta * gb2 ...
                    - 2 * g(b) * ( wbt * Mtimes(self.B, p, 1) ) ...
                    - gb2 * ( wbt * Mtimes(self.B, wbt.', 1) );
                
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
            
            iterMax = self.nlp.n;
            iter = 0;
            
            % Initialize residual and descent direction
            r = g + self.B.theta * (x - xc) ...
                - Wtimes(self.B, Mtimes(self.B, c, 1), 1);
            r = r(indFree);
            normRc = norm(r);
            self.logger.debug(sprintf('||rc|| = %9.3e', normRc))
            
            % Reduced hessian
            BHat = self.B(indFree, indFree);
            
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
                alf1 = min( self.brkpt(xc(indFree) + d, p, indFree) );
                
                % Compute step length
                q = BHat * p;
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
            [x, f, g, ~, ~, ~, ~, ~] = cvsrch( ...
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
            
        
    end % private methods
    
    
    methods (Access = public, Hidden = true)
        
        function printf(self, varargin)
            fprintf(self.fid, varargin{:});
        end
        
    end % hidden public methods
    
    
    methods (Access = protected, Static)
        
    end % private static methods
    
end % class