classdef LBFGSSolver < solvers.NlpSolver
    %% LBFGSSolver


    properties (SetAccess = protected, Hidden = false)
        maxcgiter; % maximum number of CG iters per Newton step
        cgiter; % total number of CG iters
        cgtol;
        fmin;
        fid; % File ID of where to direct log output
        boxed;

        % Wolfe linesearch parameters
        lsmu;     % Sufficient decrease for the objective function
        lseta;    % Sufficient decrease for the objective gradient
        maxIterLS;

        % Cauchy parameters
        mu0;         % Sufficient decrease parameter for the Cauchy search
        inexcauchy;  % True to perform an inexact TRON-like Cauchy search

        % L-BFGS data
        mem;  % Size of the LBFGS history
        nrejects; % Number of rejected pairs
    end

    properties (SetAccess = protected, Hidden = true)
        % Parts of the L-BFGS operator

        head;     % First index of stored pairs
        col;      % Number of stored pairs
        insert;   % Where the last part was inserted
        ws;       % Array of s vectors
        wy;       % Array of y vectors
        theta;    % Scaling parameter
        sy;       % Matrix of the YtS
        ss;       % Matrix of the StS
        wt;       % cholesky factorization of the middle matrix
        l;        % L matrix of the compact L-BFGS Formula
        dd;       % diagonal of the D matrix of the compact L-BFGS formula
        iprs      % indices of the current pairs
        iReject;  % Number of successive step rejections
        TRIUP;
        TRIDOWN;
    end

    properties (Hidden = true, Constant)
        % Minimal accepted condition number
        MINRCOND = eps;
        % Parameters for system solving in mtimes
        UP = struct('UT', true);
        UPTR = struct('UT', true, 'TRANSA', true);
        % Log header and body formats.
        LOG_HEADER_FORMAT = '\n%5s  %13s  %13s  %13s  %4s  %5s  %9s  %4s  %9s\n';
        LOG_BODY_FORMAT = ['%5i  %13.6e  %13.6e  %13.6e  %4i  %5i', ...
            '  %9d  %4d  %9f\n'];
        LOG_HEADER = {'iter', 'f(x)', 'RelDec', '|g(x)|', 'cgit', 'cginf', ...
            'nFree', 'LSit',  'time'};
    end % constant properties


    methods (Access = public)

        function self = LBFGSSolver(nlp, varargin)
            %% Constructor

            % Parse input parameters and initialize local variables
            p = inputParser;
            p.KeepUnmatched = true;
            p.PartialMatching = false;
            p.addParameter('maxcgiter', length(nlp.x0));
            p.addParameter('cgtol', 0.1);
            p.addParameter('fmin', -1e32);
            p.addParameter('fid', 1);
            p.addParameter('maxIterLS', 10);
            p.addParameter('lsmu', 1e-3);
            p.addParameter('lseta', .9);
            p.addParameter('mem', 29);
            p.addParameter('mu0', 1e-2);
            p.addParameter('inexcauchy', false);

            p.parse(varargin{:});

            self = self@solvers.NlpSolver(nlp, p.Unmatched);

            % Store various objects and parameters
            self.cgtol     = p.Results.cgtol;
            self.maxcgiter = p.Results.maxcgiter;
            self.fmin      = p.Results.fmin;
            self.fid       = p.Results.fid;
            self.lsmu  = p.Results.lsmu;
            self.lseta  = p.Results.lseta;
            self.maxIterLS = p.Results.maxIterLS;
            self.mem  = max(p.Results.mem, 1);
            self.boxed = all(nlp.jTwo);
            self.mu0       = p.Results.mu0;
            self.inexcauchy = p.Results.inexcauchy;

            % Initialize the L-BFGS operator
            self.initLBFGS();

            self.TRIUP = [triu(true(self.mem-1,self.mem-1)), ...
                          false(self.mem-1,1); ...
                          false(1,self.mem)];
            self.TRIDOWN = [false(1,self.mem); ...
                            false(self.mem-1,1), ...
                            triu(true(self.mem-1,self.mem-1))];

            import utils.PrintInfo;
        end % constructor

        function self = solve(self)
            %% Solve

            self.solveTime = tic;
            self.iter = 1;
            self.cgiter = 0;
            self.iStop = self.EXIT_NONE;
            self.nlp.resetCounters();

            printObj = utils.PrintInfo('L-BFGS-B');

            if self.verbose >= 2
                extra = containers.Map({'fmin', 'cgtol'}, ...
                    {self.fmin, self.cgtol});
                printObj.header(self, extra);
                self.printf(self.LOG_HEADER_FORMAT, self.LOG_HEADER{:});
            end

            % Make sure initial point is feasible
            x = self.project(self.nlp.x0);

            % First objective and gradient evaluation.
            [f, g] = self.nlp.obj(x);

            % Initialize stopping tolerance
            gNorm = norm(self.gpstep(x, -1, g));
            self.gNorm0 = gNorm;
            rOptTol = self.rOptTol * gNorm;

            % Actual and predicted reductions. Initial inf value prevents
            % exits based on related on first iter.
            actRed = inf;
            relRed = inf;

            % Miscellaneous iter
            LSiter = 0;
            LSfailed = false;
            cgout = NaN;
            if self.inexcauchy
                alph = 1;
            end

            gOld = g;
            xOld = x;
            fOld = f;

            %% Main loop
            while ~self.iStop

                % Check stopping conditions
                pgNorm = norm(self.gpstep(x, -1, g));
                now = toc(self.solveTime);
                if pgNorm <= rOptTol + self.aOptTol
                    self.iStop = self.EXIT_OPT_TOL;
                elseif f < self.fmin
                    self.iStop = self.EXIT_UNBOUNDED;
                elseif abs(actRed) <= self.aFeasTol || abs(relRed) <= self.rFeasTol
                    self.iStop = self.EXIT_FEAS_TOL;
                elseif self.iter >= self.maxIter
                    self.iStop = self.EXIT_MAX_ITER;
                elseif self.nlp.ncalls_fobj + self.nlp.ncalls_fcon >= ...
                        self.maxEval
                    self.iStop = self.EXIT_MAX_EVAL;
                elseif now >= self.maxRT
                    self.iStop = self.EXIT_MAX_RT;
                elseif LSfailed
                    self.iStop = self.EXIT_MAX_ITER_LS;
                end

                % Print current iter to log
                if self.verbose >= 2
                    [~, nFree] = self.getIndFree(x);
                    self.printf(self.LOG_BODY_FORMAT, self.iter, f, relRed, ...
                        pgNorm, self.cgiter, cgout, nFree, ...
                        LSiter, now);
                end

                % Act on exit conditions
                if self.iStop
                    self.x = x;
                    self.fx = f;
                    self.pgNorm = pgNorm;
                    break
                end

                % Cauchy step
                if ~self.inexcauchy
                    [xc, c, indFree, nFree, ~, failed] = self.cauchy(x, g);
                else
                    [xc, c, indFree, nFree, alph, failed] = self.cauchyb(x, g, alph);
                end
                if failed
                    % Mtimes has failed: reset LBFGS matrix
                    % and restart iteration
                    self.resetLBFGS();
                    self.logger.debug('Mtimes has failed inside cauchy: restart iteration');
                    continue
                end

                % Subspace minimization
                if nFree == 0 || self.col == 0
                    cgout = NaN;
                    d = xc - x;
                    self.logger.debug('Skipping Subspace minimization');
                else
                    [d, cgit, cgout, failed] = self.subspaceMinimization(x, g, xc, c, ...
                                                                      indFree, nFree);
                    self.cgiter = self.cgiter + cgit;
                    if failed
                        % Mtimes has failed: reset LBFGS matrix
                        self.resetLBFGS();
                        self.logger.debug('Mtimes has failed inside subsmin: restart iteration');
                        continue
                    end
                end

                self.logger.trace(['xmin = ', sprintf('%9.2e  ', x + d)]);

                % Line search
                dg = dot(d, g);
                if dg > -eps
                    % This is not a descent direction: restart iteration
                    self.resetLBFGS();
                    self.logger.debug('Not a descent direction: restart iteration');
                    continue
                end

                [x, f, g, LSiter, failed] = strongWolfe(self, x, f, g, d, dg);
                if failed || LSiter > self.maxIterLS
                    x = xOld;
                    f = fOld;
                    g = gOld;
                    if self.col == 0
                        LSfailed = true;
                        self.logger.debug('Abnormal termination in linesearch');
                    else
                        self.resetLBFGS();
                        self.logger.debug('Linesearch failed: restart iteration');
                    end
                    self.iter = self.iter + 1;
                    continue
                end

                % Update L-BFGS operator
                failed = self.updateLBFGS(x - xOld, g - gOld);
                if failed
                    % The cholesky factorization has failed: reset matrix
                    self.initLBFGS();
                    self.logger.debug('failed chol in update: restart iteration');
                end

                self.logger.trace(['x = ', sprintf('%9.2e  ', x)])
                self.logger.trace(['g = ', sprintf('%9.2e  ', g)])

                actRed = (fOld - f);
                relRed = actRed / max([abs(fOld), abs(f), 1]);
                if actRed < 0
                    % Function increased
                    self.initLBFGS();
                    self.logger.debug('failed chol in update: restart iteration');
                end

                self.iter = self.iter + 1;

                gOld = g;
                xOld = x;
                fOld = f;
            end % main loop

            self.nObjFunc = self.nlp.ncalls_fobj;
            self.nGrad = self.nlp.ncalls_gobj;
            self.nHess = 0;

            %% End of solve
            self.solveTime = toc(self.solveTime);
            % Set solved attribute
            self.isSolved();

            printObj.footer(self);
        end % solve

        %% Operations and constraints handling

        function x = project(self, x, ind)
            %% Project
            % Project a vector onto the box defined by bL, bU.
            if nargin > 2
                x = min(self.nlp.bU(ind), max(x, self.nlp.bL(ind)));
            else
                x = min(self.nlp.bU, max(x, self.nlp.bL));
            end
        end

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
            %      - nfree: the number of free variables
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
            inc = w >= eps;
            dec = w <= -eps;

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
            %      - g: the gradient of the objective function at x
            %      - indFree: the set of free indices

            % Compute breakpoints along the projected gradient path
            breakpoints = self.brkpt(x, -g);

            % Compute a direction whose coordinates are zeros if the
            % associated constraints are exposed.
            indFree = (abs(breakpoints) >= eps);
            d = zeros(self.nlp.n, 1);
            d(indFree) = -g(indFree);

            if nargout > 2
                [~, F] = sort(breakpoints);
                F = F(nnz(~indFree)+1:end);
            end
        end

        function [xc, c, indFree, nFree, tOld, resetLBFGS] = cauchy(self, x, g)
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
            %        minimization = W'(xc - x)

            self.logger.debug('-- Entering Cauchy --');

            % Compute breakpoints in the gradient direction
            [breakpoints, d, F, indFree] = self.cauchyDirection(x, g);
            self.logger.debug(sprintf('brptMax = %7.1e', max(breakpoints)));
            self.logger.debug(sprintf('brptMin = %7.1e', min(breakpoints)));
            self.logger.debug(sprintf('brpt(F1) = %7.1e', breakpoints(F(1)) ));

            %% Initialization
            xc = x;
            iter = 1;
            nFree = nnz(indFree);
            kvec = mod(self.head + (-1 : self.col - 2), self.mem) + 1;

            % Starting point for the subspace minimization
            p = self.wtrtimes(d); % p = Wt*d
            c = zeros(size(p));

            % Function derivatives
            fp  = -(d.' * d);
            if self.col > 0
                [Mp, resetLBFGS] = self.mtimes(p);
                if resetLBFGS
                    tOld = 0;
                    return
                end
                fpp = -self.theta * fp - p.' * Mp;
            else
                fpp = -self.theta * fp;
            end

            % Function minimum on the segment
            deltaTMin = -fp / fpp;
            tOld = 0;

            % First breakpoint
            b = F(iter);
            t = breakpoints(b);
            deltaT = t;

            %% Examination of subsequent segments
            while deltaTMin >= deltaT && toc(self.solveTime) < self.maxRT
                iter = iter + 1;
                % Update Cauchy point
                if d(b) > 0
                    xc(b) = self.nlp.bU(b);
                elseif d(b) < 0
                    xc(b) = self.nlp.bL(b);
                end

                %Update active constraint
                indFree(b) = false;
                nFree = nFree - 1;

                % Update c
                c  = c + deltaT * p;

                % We leave if all the constraints are active
                if iter > length(F)
                    tOld = t;
                    self.logger.trace( ['xc = ', sprintf('%9.2e ', xc)] )
                    self.logger.debug( sprintf('Iterations : %d', iter) );
                    self.logger.debug( sprintf('nFree      : %d', nFree ));
                    self.logger.debug('-- Leaving Cauchy --');
                    return
                end

                % Update directional derivatives
                zb = xc(b) - x(b);
                gb2 = g(b)^2;
                wbt = [self.wy(b, kvec), ...
                    self.theta * self.ws(b, kvec)];
                fp = fp + deltaT * fpp + gb2 + self.theta * g(b) * zb ...
                    - g(b) * ( wbt * self.mtimes(c) );
                fpp = fpp - self.theta * gb2 ...
                    - 2 * g(b) * ( wbt * self.mtimes(p) ) ...
                    - gb2 * ( wbt * self.mtimes(wbt.') );

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
            xc(F(iter:end)) = x(F(iter:end)) + tOld * d(F(iter:end));
            c = c + deltaTMin * p;

            resetLBFGS = false;

            self.logger.trace( ['xc = ', sprintf('%9.2e ', xc)] )
            self.logger.debug( sprintf('Iterations : %d', iter) );
            self.logger.debug( sprintf('nFree      : %d', nFree ));
            self.logger.debug('-- Leaving Cauchy --');
        end

        function [xc, c, indFree, nFree, alph, resetLBFGS] = cauchyb(self, x, g, alph)
            %% Cauchyb
            %  Compute the Cauchy point by backtracking

            self.logger.debug('-- Entering Cauchy --');
            self.logger.debug(sprintf('α = %7.1e',  alph));
            interpf =  0.5;     % interpolation factor
            extrapf = 1 / interpf;     % extrapolation factor

            % Compute the search direction
            [breakpoints, p] = self.cauchyDirection(x, g); % p = -g !!!

            % Find the minimal and maximal break-point on x - alph*g.
            brkptmax = max(breakpoints);
            self.logger.debug(sprintf('brptMax = %7.1e', brkptmax));
            alph = min(alph, brkptmax);

            % Evaluate the initial alph and decide if the algorithm
            % must interpolate or extrapolate.
            step = self.gpstep(x, alph, p);
            gts = g' * step;
            [Bs, failed] = self.btimes(step);
            if failed
                xc = x;
                c = [];
                indFree = [];
                nFree = [];
                resetLBFGS = true;
                return
            end
            interp = (0.5 * step' * Bs + gts >= self.mu0 * gts);

            % Either interpolate or extrapolate to find a successful step.
            if interp
                self.logger.debug('Interpolating');
                % Reduce alph until a successful step is found.
                while (toc(self.solveTime) < self.maxRT)
                    % This is a crude interpolation procedure that
                    % will be replaced in future versions of the code.
                    alph = interpf * alph;
                    step = self.gpstep(x, alph, p);
                    gts = g' * step;
                    [Bs, failed] = self.btimes(step);
                    if failed
                        xc = x;
                        c = [];
                        indFree = [];
                        nFree = [];
                        resetLBFGS = true;
                        return
                    end
                    if 0.5 * step' * Bs + gts < self.mu0 * gts
                        break
                    end
                end
            else
                self.logger.debug('Extrapolating');
                % Increase alph until a successful step is found.
                alphs = alph;
                iter = 1;
                while alph <= brkptmax && toc(self.solveTime) < self.maxRT...
                        && iter <= 100
                    % This is a crude extrapolation procedure that
                    % will be replaced in future versions of the code.
                    alph = extrapf * alph;
                    step = self.gpstep(x, alph, p);
                    gts = g' * step;
                    [Bs, failed] = self.btimes(step);
                    if failed
                        xc = x;
                        c = [];
                        indFree = [];
                        nFree = [];
                        resetLBFGS = true;
                        return
                    end
                    if 0.5 * step' * Bs + gts > self.mu0 * gts
                        break
                    end
                    alphs = alph;
                    iter = iter + 1;
                end
                % Recover the last successful step.
                alph = min(alphs, brkptmax);
                step = self.gpstep(x, alph, p);
            end
            % Prepare output
            xc = x + step;
            c = self.wtrtimes(step);
            indFree = ~(xc == self.nlp.bL & p > 0) |...
                (xc == self.nlp.bU & p < 0);
            nFree = nnz(indFree);
            resetLBFGS = false;
            self.logger.debug(sprintf('Leaving Cauchy, alpha = %7.1e', alph));
        end % cauchyBacktrack

        function [s, iter, flag, failed] = subspaceMinimization(self, x, g, xc, c, indFree, nFree)
            %% SubspaceMinimization - Minimize the quadratic on the free subspace
            %  Find the solution of the problem
            %
            %        min 1/2*s'*B*s + g'*s  s.t. s(~indFree) = (xc - x)(~indFree)
            %
            %  The solution is found by conjugate gradient
            %  Flag : 0 - Convergence
            %         1 - Constraints are violated
            %         2 - Maximum number of iteration reached
            %         3 - Maximum solve time reached

            self.logger.debug('-- Entering Subspace minimization --');

            iterMax = self.maxcgiter;
            iter = 0;

            % Initialize residual and descent direction
            [Mc, failed] = self.mtimes(c);
            if failed
                s = [];
                flag = -1;
                return
            end
            rc = g + self.theta * (xc - x) ...
                - self.wtimes(Mc);
            r = rc(indFree);
            normRc = norm(r);
            self.logger.debug(sprintf('||rc|| = %9.3e', normRc));

            if normRc <= self.aOptTol
                s = xc - x;
                flag = 0;
                self.logger.debug('Exit CG: xc is the solution');
                return
            end

            epsilon = min(self.cgtol, sqrt(normRc)) * normRc;
            p = -r;
            d = zeros(length(r),1);
            rho2 = r .'* r;

            while true
                iter = iter + 1;

                % Check exit condition
                if sqrt(rho2) < epsilon
                    flag = 0;
                    self.logger.debug(sprintf('||r|| = %9.3e', sqrt(rho2)));
                    self.logger.debug('Exit CG: Convergence');
                    break;
                end
                if iter > iterMax
                    flag = 2;
                    self.logger.debug('Exit CG: Max iteration number reached');
                    break;
                end
                if toc(self.solveTime) > self.maxRT
                    flag = 3;
                    self.logger.debug('Exit CG: Max runtime reached');
                    break;
                end

                % Compute step length
                [q, failed] = self.btimes(p, indFree);
                if failed
                    s = [];
                    flag = -1;
                    return
                end
                alf = rho2 / (p.' * q);

                % Prepare new step
                d = d + alf * p;
                r = r + alf * q;
                rho1 = rho2;
                rho2 = (r.' * r);
                beta = rho2 / rho1;
                p = -r + beta * p;
            end

            s = xc - x;
            alf = min( self.brkpt(xc(indFree), d, indFree, nFree) );
            s(indFree) = s(indFree) + min(alf, 1) * d;
            failed = false;
            self.logger.debug(sprintf('||s|| = %9.3e', norm(s)));
            self.logger.debug(' -- Leaving Subspace minimization -- ');
        end

        function [x, f, g, nfev, failed] = strongWolfe(self, x, f, g, d, dginit)
            %% StrongWolfe
            %  Wrapper to the More and Thuente line search
            %  The files cvsrch.m and cstep.m are stored in the utils
            %  folder
            %
            %  The subroutine finds a step that satisfies the strong Wolfe
            %  conditions whose parameters are defined in the solver.

            self.logger.debug('-- Entering strongWolfe --');

            % Set parameters
            self.logger.debug(sprintf('<d, g> = %7.1e', dginit));
            brkpts = self.brkpt(x, d);
            [bptmin, m] = min( brkpts );
            while bptmin <= eps
                brkpts(m) = Inf;
                [bptmin, m] = min(brkpts);
            end
            stpmax = max(eps, min( 1e10, bptmin ));
            stp = min(1, stpmax);

            if self.iter == 1
                stpmax = min(1, stpmax);
                if ~self.boxed
                    stpmax = min(1/norm(d), stpmax);
                end
            end

            self.logger.debug(sprintf('stpmax = %7.1e', stpmax));
            self.logger.debug(sprintf('Initial step = %7.1e', stp));

            % Call More and Thuente line search
            [x, f, g, stp, info, nfev] = utils.cvsrch( ...
                @self.objParamsWrapper, ... Objective wrapper
                x, [], f, g, ... Objective value at x
                d, stp, dginit, ... Search direction and recommended step
                self.lsmu, self.lseta, 0.1, ...
                0, stpmax, ...
                min(self.maxEval - self.nlp.ncalls_fobj, self.maxIterLS));

            switch info
                case 0
                    error('Linesearch: improper input parameters');
                case {1, 2, 4, 5, 6}
                    failed = false;
                case 3
                    failed = true;
                otherwise
                    error('Unknown linesearch output')
            end

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

        %% L-BFGS operations

        function initLBFGS(self)
            %% InitLBFGS - Init the LBFGS Matrix
            %  This subroutine is called in the constructor to allocate
            %  memory for the limited memory BFGS operator.

            % Initialize parameters
            self.nrejects = 0;
            self.insert = 0;
            self.theta = 1.0;
            self.head = 1;
            self.col = 0;
            self.iprs = [];

            % Initialize arrays
            self.ws = zeros(self.nlp.n, self.mem);
            self.wy = zeros(self.nlp.n, self.mem);
            self.sy = zeros(self.mem);
            self.ss = zeros(self.mem);
            self.wt = zeros(self.mem);
        end

        function resetLBFGS(self)
            %% ResetLBFGS - Discard stored pairs

            self.nrejects = 0;
            self.insert = 0;
            self.theta = 1.0;
            self.head = 1;
            self.col = 0;
            self.iprs = [];
        end

        function [ys, yy] = dotProds(~, s, y)
            %% DotProds - Prepare the dot products y'y and y's
            ys = y' * s;
            yy = y' * y;
        end

        function dtd = updateW(self, s, y, ys, yy)
            %% UpdateW - Update ws, wy, theta and return s's
            dtd = s' * s;
            self.ws(:,self.insert) = s;
            self.wy(:,self.insert) = y;
            self.theta = yy / ys;
        end

        function failed = updateLBFGS(self, s, y)
            %% UpdateLBFGS - Add a new pair to the pseudohessian
            % Store the new pair {y, s} into the L-BFGS approximation
            % Discard the oldest pair if memory has been exceeded
            % The matrices D, L, StS and J are also updated

            self.logger.debug('-- Entering updateLBFGS --');

            [ys, yy] = self.dotProds(s, y);
            
            % If ys is too small, the pair is rejected
            if ys <= eps * max(yy, 1)
                self.logger.debug('L-BFGS: Rejecting {s, y} pair');
                fprintf('L-BFGS-B : Rejecting pair\n-------------------\n')
                fprintf('||s|| = %13.6e\n||y|| = %13.6e\n', norm(s), norm(y));
                fprintf(' sy   = %13.6e\n yy   = %13.6e\n', ys, yy);
                
                self.nrejects = self.nrejects + 1;
                self.iReject = self.iReject + 1;
                if self.iReject >= 2
                    self.iReject = 0;
                    failed = true;
                else
                    failed = false;
                end
                return
            end
            
            

            % Set the column indices where to put the new pairs
            if self.col < self.mem
                self.col = self.col + 1;
                self.insert = mod(self.head + self.col - 2, self.mem) + 1;
            else
                self.insert = mod(self.insert, self.mem) + 1;
                self.head = mod(self.head, self.mem) + 1;

                % Move old information in sy and ss
                self.ss(self.TRIUP) = self.ss(self.TRIDOWN);
                self.sy(self.TRIUP') = self.sy(self.TRIDOWN');
            end
            
            c = self.col;
            m = self.mem;
            hd = self.head;

            % Update S and Y matrices and the scaling factor theta
            dtd = self.updateW(s, y, ys, yy);
            self.iprs = mod(hd - 1 : hd + c - 2, m) + 1;

            % Add new information in sy and ss
            self.sy(c,1:c-1) = s.' * self.wy(:,self.iprs(1:end-1));
            self.ss(1:c-1,c) = self.ws(:,self.iprs(1:end-1)).' * s;

            self.ss(c,c) = dtd;
            self.sy(c,c) = ys;
            self.l  = tril(self.sy(1:c,1:c), -1);
            self.dd = diag(self.sy(1:c,1:c));

            % Compute the Cholesky factorization
            % of T = theta * ss + L * D^(-1) * L' and store it
            % in the upper triangle of wt.
            self.wt(1:c,1:c) = ...
                self.theta * self.ss(1:c,1:c) ...
                + self.l * ...
                ( spdiags(self.dd, 0, c, c) \ ...
                self.l.' );

            [tmp, p] = chol(self.wt(1:c,1:c));
            if p > 0
                % The Cholesky factorization has failed
                failed = true;
                return
            end
            self.wt(1:c,1:c) = tmp;

            failed = false;
        end

        function v = wtimes(self, p, ind)
            %% wtimes - Direct product by the W matrix
            c = self.col;
            cc = 2 * c;
            if nargin < 3 % Full vector
                v = self.wy(:,self.iprs) * p(1:c) ...
                    + self.theta * self.ws(:,self.iprs) * p(c+1:cc);
            else % Subvector
                v = self.wy(ind,self.iprs) * p(1:c) ...
                    + self.theta * self.ws(ind,self.iprs) * p(c+1:cc);
            end
        end

        function p = wtrtimes(self, v, ind)
            %% wtrtimes - Adjoint product by the matrix
            if nargin < 3 % Full vector
                    p = [self.wy(:,self.iprs).' * v; self.theta * self.ws(:,self.iprs).' * v];
            else
                p = [self.wy(ind,self.iprs).' * v; ...
                     self.theta * self.ws(ind,self.iprs).' * v];
            end
        end

        function [p, illCond] = mtimes(self, q)
            %% MTimes - Apply the M matrix
            %  Compute the product with the middle matrix of the L-BFGS
            %  formula

            if self.col == 0
                illCond = false;
                p = [];
                return
            end
            
            c = self.col;
            p = zeros(2 * c, 1);
            i1 = 1:c;
            i2 = i1 + c;

            % solve [  D^(1/2)      O ] [ p1 ] = [ q1 ]
            %       [ -L*D^(-1/2)   J ] [ p2 ]   [ q2 ].

            % solve Jp2 = q2 + L * D^(-1) * q1

            p(i2) = q(i2)   + self.l * (q(i1) ./ self.dd);

            [p(i2), R] = linsolve(self.wt(i1,i1),  p(i2), self.UPTR);

            if R < self.MINRCOND
                % If the system is ill-conditioned, we leave
                illCond = true;
                p = [];
                return
            end

            % Solve D^(1/2) * p1 = q1
            p(i1) = q(i1) ./ sqrt(self.dd);

            % Solve [ -D^(1/2)   D^(-1/2)*L'  ] [ p1 ] = [ p1 ]
            %       [  0         J'           ] [ p2 ] = [ p2 ]

            % Solve J'p2 = p2
            [p(i2), R] = linsolve(self.wt(i1, i1),  p(i2), self.UP);
            if R < self.MINRCOND
                % If the system is ill-conditioned, we leave
                illCond = true;
                p = [];
                return
            end

            % Compute p1 = -D^(-1/2) (p1 - D^(-1/2)*L'*p2)
            p(i1) = -p(i1) ./ sqrt(self.dd) + (self.l' * p(i2)) ./ self.dd;
            illCond = false;
        end

        function [v, failed] = btimes(self, v, ind)
            %% BTimes - Apply the pseudohessian
            %  Compute the product by the L-BFGS hessian approximation
            if self.col > 0
                if nargin < 3
                    p = self.wtrtimes(v);
                    [p, failed] = self.mtimes(p);
                    if failed
                        % The Mtimes function has returned an error
                        return
                    end
                    v = self.theta * v - self.wtimes(p);
                else % Reduced vectors
                    p = self.wtrtimes(v, ind);
                    [p, failed] = self.mtimes(p);
                    if failed
                        % The Mtimes function has returned an error
                        return
                    end
                    v = self.theta * v - self.wtimes(p, ind);
                end
            end
            failed = false;
        end

    end % methods


    methods (Access = public, Hidden = true)

        function printf(self, varargin)
            fprintf(self.fid, varargin{:});
        end

    end % hidden public methods

end % class