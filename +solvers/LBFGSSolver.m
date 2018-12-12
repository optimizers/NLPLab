classdef LBFGSSolver < solvers.NlpSolver
    %% LBFGSSolver
    
    
    properties (SetAccess = protected, Hidden = false)
        maxIterCg; % maximum number of CG iters per Newton step
        iterCg; % total number of CG iters
        cgTol;
        fMin;
        fid; % File ID of where to direct log output
        boxed;
        cnstnd;
        
        % Linesearch parameters
        fSuffDec;
        gSuffDec;
        maxIterLS;
        
        % Cauchy parameters
        mu0; % Sufficient decrease parameter for the Cauchy search
        
        % L-BFGS data
        mem;  % Size of the LBFGS history
        nrejects; % Number of rejected pairs
    end
    
    properties (SetAccess = protected, Hidden = true)
        % Parts of the L-BFGS operator
        
        hd;       % First index of stored pairs
        cl;       % Number of stored pairs
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
        
        function o = LBFGSSolver(nlp, varargin)
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
            p.addParameter('fSuffDec', 1e-3);
            p.addParameter('gSuffDec', .9);
            p.addParameter('mem', 29);
            p.addParameter('mu0', 1e-2);
            
            p.parse(varargin{:});
            
            o = o@solvers.NlpSolver(nlp, p.Unmatched);
            
            % Store various objects and parameters
            o.cgTol     = p.Results.cgTol;
            o.maxIterCg = p.Results.maxIterCg;
            o.fMin      = p.Results.fMin;
            o.fid       = p.Results.fid;
            o.fSuffDec  = p.Results.fSuffDec;
            o.gSuffDec  = p.Results.gSuffDec;
            o.maxIterLS = p.Results.maxIterLS;
            o.mem  = max(p.Results.mem, 1);
            o.boxed = all(nlp.jTwo);
            o.cnstnd = any(nlp.jUpp | nlp.jLow);
            o.mu0       = p.Results.mu0;
            
            % Initialize the L-BFGS operator
            o.initLBFGS();                         
            
            o.TRIUP = [triu(true(o.mem-1,o.mem-1)), false(o.mem-1,1); ...
                       false(1,o.mem)];
            o.TRIDOWN = [ false(1,o.mem); ...
                false(o.mem-1,1), triu(true(o.mem-1,o.mem-1))];
            
            import utils.PrintInfo;
        end % constructor
        
        function o = solve(o)
            %% Solve
            
            o.solveTime = tic;
            o.iter = 1;
            o.iterCg = 0;
            o.iStop = o.EXIT_NONE;
            o.nlp.resetCounters();
            
            printObj = utils.PrintInfo('L-BFGS-B');
            
            if o.verbose >= 2
                extra = containers.Map({'fMin', 'cgTol'}, ...
                    {o.fMin, o.cgTol});
                printObj.header(o, extra);
                o.printf(o.LOG_HEADER_FORMAT, o.LOG_HEADER{:});
            end
                
            % Make sure initial point is feasible
            x = o.project(o.nlp.x0);
            
            % First objective and gradient evaluation.
            [f, g] = o.nlp.obj(x);
            
            % Initialize stopping tolerance
            gNorm = norm(g, inf);
            o.gNorm0 = gNorm;
            rOptTol = o.rOptTol * gNorm;
            
            % Actual and predicted reductions. Initial inf value prevents
            % exits based on related on first iter.
            actRed = inf;
            relRed = inf;
            
            % Miscellaneous iter
            LSiter = 0;
            LSfailed = false;
            cgout = NaN;
            alph = 1;
            
            gOld = g;
            xOld = x;
            fOld = f;
            
            %% Main loop
            while ~o.iStop
                
                % Check stopping conditions
                pgNorm = norm(o.gpstep(x, -1, g), inf);
                now = toc(o.solveTime);
                if pgNorm <= rOptTol + o.aOptTol
                    o.iStop = o.EXIT_OPT_TOL;
                elseif f < o.fMin
                    o.iStop = o.EXIT_UNBOUNDED;
                elseif abs(actRed) <= o.aFeasTol || abs(relRed) <= o.rFeasTol
                    o.iStop = o.EXIT_FEAS_TOL;
                elseif o.iter >= o.maxIter
                    o.iStop = o.EXIT_MAX_ITER;
                elseif o.nlp.ncalls_fobj + o.nlp.ncalls_fcon >= ...
                        o.maxEval
                    o.iStop = o.EXIT_MAX_EVAL;
                elseif now >= o.maxRT
                    o.iStop = o.EXIT_MAX_RT;
                elseif LSfailed
                    o.iStop = o.EXIT_MAX_ITER_LS;
                end
                
                % Print current iter to log
                if o.verbose >= 2
                    [~, nFree] = o.getIndFree(x);
                    o.printf(o.LOG_BODY_FORMAT, o.iter, f, relRed, ...
                        pgNorm, o.iterCg, cgout, nFree, ...
                        LSiter, now);
                end
                
                % Act on exit conditions
                if o.iStop
                    o.x = x;
                    o.fx = f;
                    o.pgNorm = pgNorm;
                    break
                end
                
                % Cauchy step
                [d, indFree, nFree, alph, failed] = ...
                    o.cauchy(x, g, alph);
                if failed
                    % Mtimes has failed: reset LBFGS matrix
                    o.resetLBFGS();
                    o.iter = o.iter + 1;
                    o.logger.debug('Mtimes has failed inside cauchy: restart iteration');
                    continue
                end
                    
                % Subspace minimization
                if nFree == 0 || o.cl == 0
                    cgout = NaN;
                    o.logger.debug('Skipping Subspace minimization');
                else
                    [d, cgit, cgout, failed] = o.subspaceMinimization(x, g, d, indFree, nFree);
                    o.iterCg = o.iterCg + cgit;
                    if failed
                        % Mtimes has failed: reset LBFGS matrix
                        o.resetLBFGS();
                        o.iter = o.iter + 1;
                        o.logger.debug('Mtimes has failed inside subsmin: restart iteration');
                        continue
                    end
                end
                
                o.logger.trace(['xmin = ', sprintf('%9.2e  ', x + d)]);
                
                % Line search

                dg = dot(d, g);
                if dg > -eps
                    % This is not a descent direction: restart iteration
                    o.resetLBFGS();
                    o.iter = o.iter + 1;
                    o.logger.debug('Not a descent direction: restart iteration');
                    continue
                end
                
                [x, f, g, LSiter, failed] = strongWolfe(o, x, f, g, d, dg);
                if failed || LSiter > o.maxIterLS
                    x = xOld;
                    f = fOld;
                    g = gOld;
                    if o.cl == 0
                        LSfailed = true;
                        o.logger.debug('Abnormal termination in linesearch');
                    else
                        o.resetLBFGS();
                        o.logger.debug('Linesearch failed: restart iteration');
                    end
                    o.iter = o.iter + 1;
                    continue
                end
                    
                % Update L-BFGS operator
                failed = o.updateLBFGS(x - xOld, g - gOld);
                if failed
                    % The cholesky factorization has failed: reset matrix
                    o.initLBFGS();
                    o.logger.debug('failed chol in update: restart iteration');
                end
                
                o.logger.trace(['x = ', sprintf('%9.2e  ', x)])
                o.logger.trace(['g = ', sprintf('%9.2e  ', g)])
                
                actRed = (fOld - f);
                relRed = actRed / max([abs([fOld, f]), 1]);
                if actRed < 0
                    % Function increased
                    o.initLBFGS();
                    o.logger.debug('failed chol in update: restart iteration');
                end
                
                o.iter = o.iter + 1;
                
                gOld = g;
                xOld = x;
                fOld = f;
            end % main loop
            
            o.nObjFunc = o.nlp.ncalls_fobj;
            o.nGrad = o.nlp.ncalls_gobj;
            o.nHess = 0;
            
            %% End of solve
            o.solveTime = toc(o.solveTime);
            % Set solved attribute
            o.isSolved();
            
            printObj.footer(o);
        end % solve
        
        %% Operations and constraints handling
        
        function x = project(o, x, ind)
            %% Project
            % Project a vector onto the box defined by bL, bU.
            if nargin > 2
                x = min(o.nlp.bU(ind), max(x, o.nlp.bL(ind)));
            else
                x = min(o.nlp.bU, max(x, o.nlp.bL));
            end
        end
        
        function [indFree, nFree] = getIndFree(o, x)
            %% GetIndFree
            % Find the free variables
            indFree = (o.nlp.bL < x) & (x < o.nlp.bU);
            if nargout > 1
                nFree = nnz(indFree);
            end
        end

        function s = gpstep(o, x, alph, w, ind)
            %% GpStep - Compute the gradient projection step.
            % Compute the gradient projection step
            %
            % s = P[x + alph*w] - x,
            %
            % where P is the projection onto the box defined by bL, bU.
            if nargin > 4
                s = o.project(x + alph * w, ind) - x;
            else
                s = o.project(x + alph * w) - x;
            end
        end
        
        function breakpoints = brkpt(o, x, w, ind, nFree)
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
                bU = o.nlp.bU(ind);
                bL = o.nlp.bL(ind);
                breakpoints = inf(nFree, 1);
            else
                bU = o.nlp.bU;
                bL = o.nlp.bL;
                breakpoints = inf(o.nlp.n, 1);
            end
            
            % Compute gradient sign
            inc = w >= eps;
            dec = w <= -eps;
            
            % Compute breakpoint for each coordinate
            breakpoints(inc) = ( bU(inc) - x(inc) ) ./ w(inc);
            breakpoints(dec) = ( bL(dec) - x(dec) ) ./ w(dec);
            breakpoints = max(breakpoints, 0);
        end
        
        function d = cauchyDirection(o, g, indFree)
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
            %      - indFree: the unexposed constraints at x
            
            % Compute a direction whose coordinates are zeros if the
            % associated constraints are exposed.
            
            d = zeros(o.nlp.n, 1);
            d(indFree) = -g(indFree);
        end
        
        function [step, indFree, nFree, alph, resetLBFGS] = cauchy(o, x, g, alph)
            %% Cauchy
            %  Compute the Cauchy point by backtracking

            o.logger.debug('-- Entering Cauchy --');
            o.logger.debug(sprintf('α = %7.1e',  alph));
            interpf =  0.5;     % interpolation factor
            extrapf = 1 / interpf;     % extrapolation factor
            
            % Compute breakpoints in the gradient direction
            breakpoints = o.brkpt(x, -g);
            indFree = (abs(breakpoints) >= eps);
            brkptmax = max(breakpoints);
            
            % Compute the search direction
            p = o.cauchyDirection(g, indFree); % p = -g !!!
            
            % Find the minimal and maximal break-point on x - alph*g.
            o.logger.debug(sprintf('brptMax = %7.1e', brkptmax));
            alph = min(alph, brkptmax);
            
            % Evaluate the initial alph and decide if the algorithm
            % must interpolate or extrapolate.
            step = o.gpstep(x, alph, p);
            gts = g' * step;
            [Bs, failed] = o.btimes(step);
            if failed
                resetLBFGS = true;
                return
            end
            interp = (0.5 * step' * Bs + gts >= o.mu0 * gts);
            
            % Either interpolate or extrapolate to find a successful step.
            if interp
                o.logger.debug('Interpolating');
                % Reduce alph until a successful step is found.
                while (toc(o.solveTime) < o.maxRT)
                    % This is a crude interpolation procedure that
                    % will be replaced in future versions of the code.
                    alph = interpf * alph;
                    step = o.gpstep(x, alph, p);
                    gts = g' * step;
                    [Bs, failed] = o.btimes(step);
                    if failed
                        resetLBFGS = true;
                        return
                    end
                    if 0.5 * step' * Bs + gts < o.mu0 * gts
                        break
                    end
                end
            else
                o.logger.debug('Extrapolating');
                % Increase alph until a successful step is found.
                alphs = alph;
                iter = 1;
                while alph <= brkptmax && toc(o.solveTime) < o.maxRT...
                        && iter <= 100
                    % This is a crude extrapolation procedure that
                    % will be replaced in future versions of the code.
                    alph = extrapf * alph;
                    step = o.gpstep(x, alph, p);
                    gts = g' * step;
                    [Bs, failed] = o.btimes(step);
                    if failed
                        resetLBFGS = true;
                        return
                    end
                    if 0.5 * step' * Bs + gts > o.mu0 * gts
                        break
                    end
                    alphs = alph;
                    iter = iter + 1;
                end
                % Recover the last successful step.
                alph = min(alphs, brkptmax);
                step = o.gpstep(x, alph, p);
            end
            % Prepare output
            indFree = ~(x + step == o.nlp.bL & p > 0) |...
                (x + step == o.nlp.bU & p < 0);
            nFree = nnz(indFree);
            resetLBFGS = false;
            o.logger.debug(sprintf('Leaving Cauchy, alpha = %7.1e', alph));
        end % cauchyBacktrack

        function [s, iter, flag, failed] = subspaceMinimization(o, x, g, s, indFree, nFree)
            %% SubspaceMinimization - Minimize the quadratic on the free subspace
            %  Find the solution of the problem
            %
            %        min 1/2*s'*B*s + g'*s  s.t. s(~indFree) = (xc - x)(~indFree)
            %
            %  The solution is found by conjugate gradient
            %  Info : 0 - Convergence
            %         1 - Constraints are violated
            %         2 - Maximum number of iteration reached
            %         3 - Maximum solve time reached
            
            o.logger.debug('-- Entering Subspace minimization --');
            failed = false;
            
            % Initialize residual and descent direction
            rc = g + o.btimes(s);
            r = rc(indFree);
            normRc = norm(r);
            
            [d, flag, ~, iter] = pcg(@(v)o.btimes(v, indFree), -r, ...
                min(o.cgTol, sqrt(normRc)), o.maxIterCg);
            
            % Find smallest breakpoint in the found direction
            alf1 = min( o.brkpt(x(indFree) + s(indFree), ... 
                d, indFree, nFree) );
            s(indFree) = s(indFree) + alf1 * d;
            %s(indFree) = s(indFree) + d;
            
            o.logger.debug(sprintf('||s|| = %9.3e', norm(s)));
            o.logger.debug(' -- Leaving Subspace minimization -- ');
        end
        
        function [x, f, g, nfev, failed] = strongWolfe(o, x, f, g, d, dginit)
            %% StrongWolfe
            %  Wrapper to the More and Thuente line search
            %  The files cvsrch.m and cstep.m are stored in the utils
            %  folder
            %
            %  The subroutine finds a step that satisfies the strong Wolfe
            %  conditions whose parameters are defined in the solver.
            
            if o.iter == 128
                fprintf('');
            end
            o.logger.debug('-- Entering strongWolfe --');
            
            % Set parameters
            o.logger.debug(sprintf('<d, g> = %7.1e', dginit));
            brkpts = o.brkpt(x, d);
            [bptmin, m] = min( brkpts );
            while bptmin <= eps
                brkpts(m) = Inf;
                [bptmin, m] = min(brkpts);
            end
            stpmax = max(eps, min( 1e10, bptmin ));
            stp = min(1, stpmax);
            
            if o.iter == 1
                stpmax = min(1, stpmax);
                if ~o.boxed
                    stpmax = min(1/norm(d), stpmax);
                end
            end
            
            o.logger.debug(sprintf('stpmax = %7.1e', stpmax));
            o.logger.debug(sprintf('Initial step = %7.1e', stp));
            
            % Call More and Thuente line search
            [x, f, g, stp, info, nfev] = utils.cvsrch( ...
                @o.objParamsWrapper, ... Objective wrapper
                x, [], f, g, ... Objective value at x
                d, stp, dginit, ... Search direction and recommended step
                o.fSuffDec, o.gSuffDec, 0.1, ...
                0, stpmax, ...
                min(o.maxEval - o.nlp.ncalls_fobj, o.maxIterLS));
            
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
            
            o.logger.debug(sprintf('  Final step = %7.1e', stp));
            o.logger.debug('-- Leaving strongWolfe --');
        end
        
        function [f, g, params] = objParamsWrapper(o, x, ~)
            %% fParamsWrapper
            %  Wrapper to call the objective function in the line search 
            %  subroutine
            [f, g] = o.nlp.obj(x);
            params = [];
        end
        
        %% L-BFGS operations
            
        function initLBFGS(o)
            %% InitLBFGS - Init the LBFGS Matrix
            %  This subroutine is called in the constructor to allocate
            %  memory for the limited memory BFGS operator.
            
            % Initialize parameters
            o.nrejects = 0;
            o.insert = 0;
            o.theta = 1.0;
            o.hd = 1;
            o.cl = 0;
            o.iprs = [];
            
            % Initialize arrays
            o.ws = zeros(o.nlp.n, o.mem);
            o.wy = zeros(o.nlp.n, o.mem);
            o.sy = zeros(o.mem);
            o.ss = zeros(o.mem);
            o.wt = zeros(o.mem);
        end
        
        function resetLBFGS(o)
            %% ResetLBFGS - Discard stored pairs
            
            o.nrejects = 0;
            o.insert = 0;
            o.theta = 1.0;
            o.hd = 1;
            o.cl = 0;
            o.iprs = 0;
        end
        
        function [ys, yy] = dotProds(~, s, y)
            %% DotProds - Prepare the dot products y'y and y's
            ys = y' * s;
            yy = y' * y;
        end
        
        function dtd = updateW(o, s, y, ys, yy)
            %% UpdateW - Update ws, wy, theta and return s's
            dtd = s' * s;
            o.ws(:,o.insert) = s;
            o.wy(:,o.insert) = y;
            o.theta = yy / ys;
        end
        
        function failed = updateLBFGS(o, s, y)
            %% UpdateLBFGS - Add a new pair to the pseudohessian
            % Store the new pair {y, s} into the L-BFGS approximation
            % Discard the oldest pair if memory has been exceeded
            % The matrices D, L, StS and J are also updated
            
            o.logger.debug('-- Entering updateLBFGS --');
            
            [ys, yy] = o.dotProds(s, y);
            
            if ys <= eps * max(yy, 1)
                o.logger.debug('L-BFGS: Rejecting {s, y} pair');
                o.nrejects = o.nrejects + 1;
                o.iReject = o.iReject + 1;
                if o.iReject >= 2
                    o.iReject = 0;
                    failed = true;
                else
                    failed = false;
                end
                return
            end
            
            % Set the column indices where to put the new pairs
            if o.cl < o.mem
                o.cl = o.cl + 1;
                o.insert = mod(o.hd + o.cl - 2, o.mem) + 1;
            else
                o.insert = mod(o.insert, o.mem) + 1;
                o.hd = mod(o.hd, o.mem) + 1;
                
                % Move old information in sy and ss
                o.ss(o.TRIUP) = o.ss(o.TRIDOWN);
                o.sy(o.TRIUP') = o.sy(o.TRIDOWN');
            end
            
            
            % Update S and Y matrices and the scaling factor theta
            dtd = o.updateW(s, y, ys, yy);
            o.iprs = mod(o.hd - 1 : o.hd + o.cl - 2, o.mem) + 1;
            
            % Add new information in sy and ss
            o.sy(o.cl,1:o.cl-1) = s.' * o.wy(:,o.iprs(1:end-1));
            o.ss(1:o.cl-1,o.cl) = o.ws(:,o.iprs(1:end-1)).' * s;
            
            o.ss(o.cl,o.cl) = dtd;
            o.sy(o.cl,o.cl) = ys;
            o.l  = tril(o.sy(1:o.cl,1:o.cl), -1);
            o.dd = diag(o.sy(1:o.cl,1:o.cl));
            
            % Compute the Cholesky factorization 
            % of T = theta * ss + L * D^(-1) * L' and store it
            % in the upper triangle of wt.
            o.wt(1:o.cl,1:o.cl) = ...
                o.theta * o.ss(1:o.cl,1:o.cl) ...
                + o.l * ...
                ( spdiags(o.dd, 0, o.cl, o.cl) \ ...
                o.l.' );
            
            [tmp, p] = chol(o.wt(1:o.cl,1:o.cl));
            if p > 0
                % The Cholesky factorization has failed
                failed = true;
                return
            end
            o.wt(1:o.cl,1:o.cl) = tmp;
            
            failed = false;
        end
        
        function v = wtimes(o, p, ind)
            %% wtimes - Direct product by the W matrix
            if nargin < 3 % Full vector
                v = o.wy(:,o.iprs) * p(1:o.cl) ...
                    + o.theta * o.ws(:,o.iprs) * p(o.cl+1:2*o.cl);
            else % Subvector
                v = o.wy(ind,o.iprs) * p(1:o.cl) ... 
                    + o.theta * o.ws(ind,o.iprs) * p(o.cl+1:2*o.cl);
            end
        end
        
        function p = wtrtimes(o, v, ind)
            %% wtrtimes - Adjoint product by the matrix
            if nargin < 3 % Full vector
                p = [o.wy(:,o.iprs).' * v; o.theta * o.ws(:,o.iprs).' * v];
            else
                p = [o.wy(ind,o.iprs).' * v; ...
                     o.theta * o.ws(ind,o.iprs).' * v];
            end
        end
        
        function [p, illCond] = mtimes(o, q)
            %% MTimes - Apply the M matrix
            %  Compute the product with the middle matrix of the L-BFGS
            %  formula
            
            if o.cl == 0
                illCond = false;
                p = [];
                return
            end
            p = zeros(2 * o.cl, 1);
            i1 = 1:o.cl;
            i2 = i1 + o.cl;
            
            % solve [  D^(1/2)      O ] [ p1 ] = [ q1 ]
            %       [ -L*D^(-1/2)   J ] [ p2 ]   [ q2 ].
            
            % solve Jp2 = q2 + L * D^(-1) * q1
            
            p(i2) = q(i2)   + o.l * (q(i1) ./ o.dd);
            
            [p(i2), R] = linsolve(o.wt(i1,i1),  p(i2), o.UPTR);
            
            if R < o.MINRCOND
                % If the system is ill-conditioned, we leave
                illCond = true;
                p = [];
                return
            end

            % Solve D^(1/2) * p1 = q1
            p(i1) = q(i1) ./ sqrt(o.dd);

            % Solve [ -D^(1/2)   D^(-1/2)*L'  ] [ p1 ] = [ p1 ]
            %       [  0         J'           ] [ p2 ] = [ p2 ]

            % Solve J'p2 = p2
            [p(i2), R] = linsolve(o.wt(i1, i1),  p(i2), o.UP);
            if R < o.MINRCOND
                % If the system is ill-conditioned, we leave
                illCond = true;
                p = [];
                return
            end

            % Compute p1 = -D^(-1/2) (p1 - D^(-1/2)*L'*p2)
            p(i1) = -p(i1) ./ sqrt(o.dd) + (o.l' * p(i2)) ./ o.dd;
            illCond = false;
        end

        function [v, failed] = btimes(o, v, ind)
            %% BTimes - Apply the pseudohessian
            %  Compute the product by the L-BFGS hessian approximation
            if o.cl > 0
                if nargin < 3
                    p = o.wtrtimes(v);
                    [p, failed] = o.mtimes(p);
                    if failed
                        % The Mtimes function has returned an error
                        return
                    end
                    v = o.theta * v - o.wtimes(p);
                else % Reduced vectors
                    p = o.wtrtimes(v, ind);
                    [p, failed] = o.mtimes(p);
                    if failed
                        % The Mtimes function has returned an error
                        return
                    end
                    v = o.theta * v - o.wtimes(p, ind);
                end
            end
            failed = false;
        end
                
    end % methods
    
    
    methods (Access = public, Hidden = true)
        
        function printf(o, varargin)
            fprintf(o.fid, varargin{:});
        end
        
    end % hidden public methods
    
end % class