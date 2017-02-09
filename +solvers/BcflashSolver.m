classdef BcflashSolver < solvers.NlpSolver
    %% BcflashSolver
    
    
    properties (SetAccess = private, Hidden = false)
        maxCgIter      % maximum number of CG iters per Newton step
        nSuccessIter = 0 % number of successful iters
        iterCg = 0    % total number of CG iters
        gNorm0         % norm of the gradient at x0
        exitMsg       % string indicating exit
        mu0            % sufficient decrease parameter
        cgTol
        fMin
        fid            % File ID of where to direct log output
    end % hidden gettable private properties
    
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
        LOG_HEADER_FORMAT = '\n%5s  %13s  %13s  %5s  %9s  %9s\n';
        LOG_BODY_FORMAT = '%5i  %13.6e  %13.6e  %5i  %9.3e  %9.3e  %3s\n';
        LOG_HEADER = {'iter', 'f(x)', '|g(x)|', 'cg', 'preRed', 'radius'};
    end % constant properties
    
    
    methods (Access = public)
        
        function self = BcflashSolver(nlp, varargin)
            %% Constructor
            
            % Parse input parameters and initialize local variables
            p = inputParser;
            p.KeepUnmatched = true;
            p.PartialMatching = false;
            p.addParameter('maxCgIter', length(nlp.x0));
            p.addParameter('cgTol', 0.1);
            p.addParameter('fMin', -1e32);
            p.addParameter('mu0', 0.01);
            p.addParameter('fid', 1);
            
            p.parse(varargin{:});
            
            self = self@solvers.NlpSolver(nlp, p.Unmatched);
            
            % Store various objects and parameters
            self.cgTol = p.Results.cgTol;
            self.maxCgIter = p.Results.maxCgIter;
            self.fMin = p.Results.fMin;
            self.mu0 = p.Results.mu0;
            self.fid = p.Results.fid;
            
            import utils.PrintInfo;
        end % constructor
        
        function self = solve(self)
            %% Solve
            self.solveTime = tic;
            self.iter = 0;
            
            printObj = utils.PrintInfo('bcflash');
                    
            if self.verbose >= 2
                extra = containers.Map({'fMin', 'cgTol', 'mu0'}, ...
                    {self.fMin, self.cgTol, self.mu0});
                printObj.header(self, extra);
            end
            
            % Make sure initial point is feasible
            x = solvers.BcflashSolver.project(self.nlp.x0, self.nlp.bL, ...
                self.nlp.bU);
            
            % First objective and gradient evaluation.
            [f, g] = self.nlp.obj(x);
            
            % Initialize stopping tolerance and initial TR radius
            gNorm = norm(g);
            delta = gNorm;
            self.gNorm0 = gNorm;
            self.rOptTol = self.aOptTol * gNorm;
            self.rFeasTol = self.aFeasTol * abs(f);
            
            % Actual and predicted reductions. Initial inf value prevents
            % exits based on related on first iter.
            actRed = inf;
            preRed = inf;
            
            % Miscellaneous iter
            alphc = 1;
            itCg = 0;
            sigma1 = self.sig1;
            sigma2 = self.sig2;
            sigma3 = self.sig3;
            self.iStop = self.EXIT_NONE;
            
            %% Main loop
            while true
                % Check stopping conditions
                pgNorm = solvers.BcflashSolver.gpnrm2(x, self.nlp.bL, ...
                    self.nlp.bU, g);
                exit = pgNorm <= self.rOptTol;
                if ~self.iStop && exit
                    self.iStop = self.EXIT_OPT_TOL;
                end
                
                exit = f < self.fMin;
                if ~self.iStop && exit
                    self.iStop = self.EXIT_UNBOUNDED;
                end
                
                exit = abs(actRed) <= self.aFeasTol && ...
                    preRed  <= self.aFeasTol;
                if ~self.iStop && exit
                    self.iStop = self.EXIT_FEAS_TOL;
                end
                
                exit = abs(actRed) <= self.rFeasTol && ...
                    preRed  <= self.rFeasTol;
                if ~self.iStop && exit
                    self.iStop = self.EXIT_FEAS_TOL;
                end
                
                exit = self.iter >= self.maxIter;
                if ~self.iStop && exit
                    self.iStop = self.EXIT_MAX_ITER;
                end
                
                exit = toc(self.solveTime) >= self.maxRT;
                if ~self.iStop && exit
                    self.iStop = self.EXIT_MAX_RT;
                end
                
                % Print current iter to log
                if self.verbose >= 2
                    if mod(self.iter, 20) == 0
                        fprintf(self.LOG_HEADER_FORMAT, self.LOG_HEADER{:});
                    end
                    if self.iter == 0 || successful
                        status = '';
                    else
                        status = 'rej';
                    end
                    self.printf(self.LOG_BODY_FORMAT, self.iter, f, pgNorm, itCg, ...
                        preRed, delta, status);
                end
                
                % Act on exit conditions
                if self.iStop
                    self.exitMsg = self.EXIT_MSG{self.iStop};
                    self.x = x;
                    self.fx = f;
                    self.pgNorm = pgNorm;
                    break
                end
                
                %% Iteration starts here
                self.iter = self.iter + 1;
                fc = f;
                xc = x;
                
                % Hessian operator.
                Aprod = @(v) self.nlp.hobjprod(x, [], v);
                
                [alphc, s] = self.cauchy(Aprod, x, g, delta, alphc);
                
                % Projected Newton step.
                [x, s, itCg, ~] = self.spcg(Aprod, x, g, delta, ...
                    self.cgTol, s, self.maxCgIter, self.nlp.bL, ...
                    self.nlp.bU);
                self.iterCg = self.iterCg + itCg;
                
                % Predicted reduction.
                As = self.nlp.hobjprod(x, [], s);
                preRed = -(s'*g + 0.5*s'*As);
                
                % Compute the objective at this new point.
                f = self.nlp.fobj(x);
                actRed = fc - f;
                sNorm = norm(s);
                
                % Update the trust-region radius.
                if self.nSuccessIter == 0
                    delta = min(delta, sNorm);
                end
                gts = g'*s;
                if f - fc - gts <= 0
                    alph = sigma3;
                else
                    alph = max(sigma1, -0.5*gts/(f-fc-gts));
                end
                
                if actRed < self.eta0 * preRed || actRed == -inf;
                    delta = min(max(alph, sigma1) * sNorm, sigma2 * delta);
                elseif actRed < self.eta1 * preRed
                    delta = max(sigma1 * delta, min(alph * sNorm, ...
                        sigma2 * delta));
                elseif actRed < self.eta2 * preRed
                    delta = max(sigma1 * delta, min(alph * sNorm, ...
                        sigma3 * delta));
                else
                    delta = max(delta, min(alph * sNorm, sigma3 * delta));
                end
                
                if actRed > self.eta0 * preRed;
                    successful = true;
                    self.nSuccessIter = self.nSuccessIter + 1;
                    g = self.nlp.gobj(x);
                else
                    successful = false;
                    f = fc;
                    x = xc;
                end
                
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
        
    end % public methods
    
    
    methods (Access = private)
        
        function [alph, s] = cauchy(self, Aprod, x, g, delta, alph)
            %CAUCHY
            %
            % This subroutine computes a Cauchy step that satisfies a trust
            % region constraint and a sufficient decrease condition.
            %
            % The Cauchy step is computed for the quadratic
            %
            %       q(s) = 0.5*s'*A*s + g'*s,
            %
            % where A is a symmetric matrix in compressed row storage, and
            % g is a vector. Given a parameter alph, the Cauchy step is
            %
            %       s[alph] = P[x - alph*g] - x,
            %
            % with P the projection onto the n-dimensional interval
            % [bL, bU]. The Cauchy step satisfies the trust region
            % constraint and the sufficient decrease condition
            %
            %       || s || <= delta,      q(s) <= mu_0*(g'*s),
            %
            % where mu_0 is a constant in (0,1).
            
            interpf =  0.1;     % interpolation factor
            extrapf = 10.0;     % extrapolation factor
            bL = self.nlp.bL;
            bU = self.nlp.bU;
            
            % Find the minimal and maximal break-point on x - alph*g.
            [~, ~, brptMax] = solvers.BcflashSolver.breakpt(x, -g, bL, bU);
            
            % Evaluate the initial alph and decide if the algorithm
            % must interpolate or extrapolate.
            s = solvers.BcflashSolver.gpstep(x, -alph, g, bL, bU);
            
            if norm(s) >= delta
                interp = true;
            else
                wa = Aprod(s);
                gts = g'*s;
                q = 0.5*s'*wa + gts;
                interp = (q >= self.mu0*gts);
            end
            
            % Either interpolate or extrapolate to find a successful step.
            if interp
                
                % Reduce alph until a successful step is found.
                search = true;
                while search
                    
                    % This is a crude interpolation procedure that
                    % will be replaced in future versions of the code.
                    alph = interpf*alph;
                    
                    s = solvers.BcflashSolver.gpstep(x, -alph, g, bL, bU);
                    if norm(s) <= delta
                        wa = Aprod(s);
                        gts = g'*s;
                        q = 0.5*s'*wa + gts;
                        search = (q >= self.mu0*gts);
                    end
                end
                
            else
                
                % Increase alph until a successful step is found.
                search = true;
                alphs = alph;
                while search && alph <= brptMax
                    
                    % This is a crude extrapolation procedure that
                    % will be replaced in future versions of the code.
                    
                    alph = extrapf*alph;
                    s = solvers.BcflashSolver.gpstep(x, -alph, g, bL, bU);
                    if norm(s) <= delta
                        wa = Aprod(s);
                        gts = g'*s;
                        q = 0.5*s'*wa + gts;
                        if q <= self.mu0*gts
                            search = true;
                            alphs = alph;
                        end
                    else
                        search = false;
                    end
                end
                
                % Recover the last successful step.
                alph = alphs;
                s = solvers.BcflashSolver.gpstep(x, -alph, g, bL, bU);
            end
            
        end % cauchy
        
        function [x, w] = prsrch(self, Aprod, x, g, w, bL, bU)
            %PRSRCH  Projected search.
            %
            % [x, w] = prsrch(Aprod, x, g, w, bL, bU) where
            %
            %     Inputs:
            %     Aprod is a function handle to compute matrix-vector
            %     products
            %     x        current point
            %     g        current gradient
            %     w        search direction
            %     bL       vector of lower bounds
            %     bU       vector of upper bounds
            %     mu0      linesearch parameter
            %     interpf  interpolation parameter
            %
            %     Output:
            %     x is the final point P[x + alph*w]
            %     w is the step s[alph]
            %
            %     This subroutine uses a projected search to compute a step
            %     that satisfies a sufficient decrease condition for the
            %     quadratic
            %
            %           q(s) = 0.5*s'*A*s + g'*s,
            %
            %     where A is a symmetric matrix in compressed column
            %     storage, and g is a vector. Given the parameter alph,
            %     the step is
            %
            %           s[alph] = P[x + alph*w] - x,
            %
            %     where w is the search direction and P the projection onto
            %     the n-dimensional interval [bL,bU]. The final step
            %     s = s[alph] satisfies the sufficient decrease condition
            %
            %           q(s) <= mu_0*(g'*s),
            %
            %     where mu_0 is a constant in (0,1).
            %
            %     The search direction w must be a descent direction for
            %     the quadratic q at x such that the quadratic is
            %     decreasing in the ray  x + alph*w for 0 <= alph <= 1.
            
            interpf = 0.5; % Interpolation factor
            
            % Set the initial alph = 1 because the quadratic function is
            % decreasing in the ray x + alph*w for 0 <= alph <= 1.
            alph = 1;
            nSteps = 0;
            
            % Find the smallest break-point on the ray x + alph*w.
            [~, brptMin, ~] = solvers.BcflashSolver.breakpt(x, w, bL, bU);
            
            % Reduce alph until the sufficient decrease condition is
            % satisfied or x + alph*w is feasible.
            search = true;
            while search && alph > brptMin
                
                % Calculate P[x + alph*w] - x and check the sufficient
                % decrease condition.
                nSteps = nSteps + 1;
                s = solvers.BcflashSolver.gpstep(x, alph, w, bL, bU);
                As = Aprod(s);
                gts = g' * s;
                q = 0.5 * s' * As + gts;
                
                if q <= self.mu0 * gts
                    search = false;
                    
                else
                    % This is a crude interpolation procedure that
                    % will be replaced in future versions of the code.
                    alph = interpf * alph;
                    
                end
            end
            
            % Force at least one more constraint to be added to the active
            % set if alph < brptMin and the full step is not successful.
            % There is sufficient decrease because the quadratic function
            % is decreasing in the ray x + alph*w for 0 <= alph <= 1.
            if alph < 1 && alph < brptMin
                alph = brptMin;
            end
            % Compute the final iterate and step.
            s = solvers.BcflashSolver.gpstep(x, alph, w, bL, bU);
            x = solvers.BcflashSolver.project(x + alph*w, bL, bU);
            w = s;
        end % prsrch
        
        function [x, s, iters, info] = spcg(self, Aprod, x, g, delta, ...
                rTol, s, iterMax, bL, bU)
            %SPCG  Minimize a bound-constraint quadratic.
            %
            % This subroutine generates a sequence of approximate
            % minimizers for the subproblem
            %
            %       min { q(x) : bL <= x <= bU }.
            %
            % The quadratic is defined by
            %
            %       q(x[0]+s) = 0.5*s'*A*s + g'*s,
            %
            % where x[0] is a base point provided by the user, A is a
            % symmetric matrix in compressed column storage, and g is a
            % vector.
            %
            % At each stage we have an approximate minimizer x[k], and
            % generate a direction p[k] by using a preconditioned conjugate
            % gradient method on the subproblem
            %
            %       min { q(x[k]+p) : || L'*p || <= delta, s(fixed) = 0 },
            %
            % where fixed is the set of variables fixed at x[k], delta is
            % the trust region bound, and L is an incomplete Cholesky
            % factorization of the submatrix
            %
            %       B = A(free:free),
            %
            % where free is the set of free variables at x[k]. Given p[k],
            % the next minimizer x[k+1] is generated by a projected search.
            %
            % The starting point for this subroutine is x[1] = x[0] + s,
            % where x[0] is a base point and s is the Cauchy step.
            %
            % The subroutine converges when the step s satisfies
            %
            %       || (g + A*s)[free] || <= rTol*|| g[free] ||
            %
            % In this case the final x is an approximate minimizer in the
            % face defined by the free variables.
            %
            % The subroutine terminates when the trust region bound does
            % not allow further progress, that is, || L'*p[k] || = delta.
            % In this case the final x satisfies q(x) < q(x[k]).
            %
            % On exit info is set as follows:
            %
            %      info = 1  Convergence. The final step s satisfies
            %                || (g + A*s)[free] || <= rTol*|| g[free] ||,
            %                and the final x is an approximate minimizer
            %                in the face defined by the free variables.
            %
            %      info = 2  Termination. The trust region bound does
            %                not allow further progress.
            %
            %      info = 3  Failure to converge within iterMax iters.
            
            n = length(x);
            
            % Compute A*(x[1] - x[0]) and store in w.
            As = Aprod(s);
            
            % Compute the Cauchy point.
            x = solvers.BcflashSolver.project(x + s, bL, bU);
            
            % Start the main iter loop.
            % There are at most n iters because at each iter
            % at least one variable becomes active.
            iters = 0;
            for nfaces = 1:n
                
                % Determine the free variables at the current minimizer.
                % The indices of the free variables are stored in the first
                % n free positions of the array indFree.
                indFree = (bL < x) & (x < bU);
                nFree = sum(indFree);
                
                % Exit if there are no free constraints.
                if nFree == 0
                    info = 1;
                    return
                end
                
                % Compute the gradient grad q(x[k]) = g + A*(x[k] - x[0]),
                % of q at x[k] for the free variables.
                % Recall that w contains  A*(x[k] - x[0]).
                % Compute the norm of the reduced gradient Z'*g.
                wa = g(indFree);
                gFree = As(indFree) + wa;
                gfNorm = norm(wa);
                
                % Solve the trust region subproblem in the free variables
                % to generate a direction p[k]. Store p[k] in the array w.
                tol = rTol*gfNorm;
                stol = 0;
                
                % Create the submatrix operator.
                Bprod = @(x)solvers.BcflashSolver.Afree(x, Aprod, ...
                    indFree, n);
                
                L = speye(nFree); % No preconditioner for now.
                [w, iterTR, infoTR] = ...
                    solvers.BcflashSolver.trpcg(Bprod, gFree, delta, L, ...
                    tol, stol, iterMax);
                
                iters = iters + iterTR;
                w = L' \ w;
                
                % Use a projected search to obtain the next iterate.
                % The projected search algorithm stores s[k] in w.
                xFree = x(indFree);
                bLFree = bL(indFree);
                bUFree = bU(indFree);
                
                [xFree, w] = self.prsrch(Bprod, xFree, gFree, w, ...
                    bLFree, bUFree);
                
                % Update the minimizer and the step.
                % Note that s now contains x[k+1] - x[0].
                x(indFree) = xFree;
                s(indFree) = s(indFree) + w;
                
                % Compute A*(x[k+1] - x[0]) and store in w.
                As = Aprod(s);
                
                % Compute the gradient grad q(x[k+1]) = g + A*(x[k+1]-x[0])
                % of q at x[k+1] for the free variables.
                gFree = As(indFree) + g(indFree);
                gfNormF = norm(gFree);
                
                % Convergence and termination test.
                % We terminate if the preconditioned conjugate gradient
                % method encounters a direction of negative curvature, or
                % if the step is at the trust region bound.
                
                if gfNormF <= rTol * gfNorm
                    info = 1;
                    return
                elseif infoTR == 3 || infoTR == 4
                    info = 2;
                    return
                elseif iters > iterMax
                    info = 3;
                    return
                end
            end % faces
        end % spcg
        
    end % private methods
    
    
    methods (Access = public, Hidden = true)
        
        function printf(self, varargin)
            fprintf(self.fid, varargin{:});
        end
        
    end % hidden public methods
    
    
    methods (Access = private, Static)
        
        function xFree = Afree(xFree, Aprod, indFree, n)
            z = zeros(n,1);
            z(indFree) = xFree;
            z = Aprod(z);
            xFree = z(indFree);
        end
        
        function [w, iters, info] = trpcg(Aprod, g, delta, L, tol, ...
                stol, iterMax)
            %TRPCG
            %
            % Given a sparse symmetric matrix A in compressed column
            % storage, this subroutine uses a preconditioned conjugate
            % gradient method to find an approximate minimizer of the trust
            % region subproblem
            %
            %       min { q(s) : || L'*s || <= delta }.
            %
            % where q is the quadratic
            %
            %       q(s) = 0.5*s'*A*s + g'*s,
            %
            % A is a symmetric matrix in compressed column storage, L is a
            % lower triangular matrix in compressed column storage, and g
            % is a vector.
            %
            % This subroutine generates the conjugate gradient iterates for
            % the equivalent problem
            %
            %       min { Q(w) : || w || <= delta }.
            %
            % where Q is the quadratic defined by
            %
            %       Q(w) = q(s),      w = L'*s.
            %
            % Termination occurs if the conjugate gradient iterates leave
            % the trust region, a negative curvature direction is
            % generated, or one of the following two convergence tests is
            % satisfied.
            %
            % Convergence in the original variables:
            %
            %       || grad q(s) || <= tol
            %
            % Convergence in the scaled variables:
            %
            %       || grad Q(w) || <= stol
            %
            % Note that if w = L'*s, then L*grad Q(w) = grad q(s).
            %
            % On exit info is set as follows:
            %
            %       info = 1  Convergence in the original variables.
            %                 || grad q(s) || <= tol
            %
            %       info = 2  Convergence in the scaled variables.
            %                 || grad Q(w) || <= stol
            %
            %       info = 3  Negative curvature direction generated.
            %                 In this case || w || = delta and a direction
            %                 of negative curvature w can be recovered by
            %                 solving L'*w = p.
            %
            %       info = 4  Conjugate gradient iterates exit the
            %                 trust region. In this case || w || = delta.
            %
            %       info = 5  Failure to converge within iterMax iters.
            
            n = length(g);
            
            % Initialize the iterate w and the residual r.
            w = zeros(n,1);
            
            % Initialize the residual t of grad q to -g.
            % Initialize the residual r of grad Q by solving L*r = -g.
            % Note that t = L*r.
            t = -g;
            r = L \ -g;
            
            % Initialize the direction p.
            p = r;
            
            % Initialize rho and the norms of r and t.
            rho = r'*r;
            rnorm0 = sqrt(rho);
            
            % Exit if g = 0.
            if rnorm0 == 0
                iters = 0;
                info = 1;
                return
            end
            
            for iters = 1:iterMax
                % Compute z by solving L'*z = p.
                z = L' \ p;
                % Compute q by solving L*q = A*z and save L*q for
                % use in updating the residual t.
                Az = Aprod(z);
                z = Az;
                q = L \ Az;
                % Compute alph and determine sigma such that the TR
                % constraint || w + sigma*p || = delta is satisfied.
                ptq = p'*q;
                if ptq > 0
                    alph = rho/ptq;
                else
                    alph = 0;
                end
                sigma = solvers.BcflashSolver.trqsol(w, p, delta);
                % Exit if there is negative curvature or if the
                % iterates exit the trust region.
                if (ptq <= 0 || alph >= sigma)
                    if sigma ~= 0
                        w = w + sigma*p;
                    end
                    if ptq <= 0
                        info = 3;
                    else
                        info = 4;
                    end
                    
                    return
                end
                % Update w and the residuals r and t.
                % Note that t = L*r.
                w = w + alph*p;
                r = r - alph*q;
                t = t - alph*z;
                % Exit if the residual convergence test is satisfied.
                rtr = r'*r;
                rnorm = sqrt(rtr);
                tnorm = norm(t);
                if tnorm <= tol
                    info = 1;
                    return
                end
                if rnorm <= stol
                    info = 2;
                    return
                end
                % Compute p = r + betaFact*p and update rho.
                betaFact = rtr/rho;
                p = r + betaFact*p;
                rho = rtr;
            end % for loop
            iters = itmax;
            info = 5;
        end % trpcg
        
        function x = project(x, bL, bU)
            %Project  Project a vector onto the box defined by bL, bU.
            x = max( x, bL );
            x = min( x, bU );
        end
        
        function s = gpstep(x, alph, w, bL, bU)
            %GPSTEP  Compute the gradient projection step.
            %
            % Compute the gradient projection step
            %
            % s = P[x + alph*w] - x,
            %
            % where P is the projection onto the box defined by bL, bU.
            aw = alph*w;
            s = x + aw;
            
            iLow = s < bL;         % violate lower bound
            iUpp = s > bU;         % violate upper bound
            iFre = ~(iLow | iUpp); % free
            
            s(iLow) = bL(iLow) - x(iLow);
            s(iUpp) = bU(iUpp) - x(iUpp);
            s(iFre) = aw(iFre);
        end
        
        function pNorm = gpnrm2(x, bL, bU, g)
            %% GpNrm2 - Norm-2 of the projected gradient
            notFixed = bL < bU;
            low = notFixed & x == bL;
            upp = notFixed & x == bU;
            fre = notFixed & ~(low | upp);
            
            pNorm1 = norm(g(low & g < 0))^2;
            pNorm2 = norm(g(upp & g > 0))^2;
            pNorm3 = norm(g(fre))^2;
            
            pNorm = sqrt(pNorm1 + pNorm2 + pNorm3);
        end
        
        function [nBrpt, brptMin, brptMax] = breakpt(x, w, bL, bU)
            %BREAKPT
            %     This subroutine computes the number of break-points, and
            %     the minimal and maximal break-points of the projection of
            %     x + w on the n-dimensional interval [bL,bU].
            inc = x < bU & w > 0;     % upper bound intersections
            dec = x > bL & w < 0;     % lower bound intersections
            
            nBrpt = sum(inc | dec);   % number of breakpoints
            
            % Quick exit if no breakpoints
            if nBrpt == 0
                brptMin = 0;
                brptMax = 0;
                return
            end
            
            brptInc = (bU(inc) - x(inc)) ./ w(inc);
            brptDec = (bL(dec) - x(dec)) ./ w(dec);
            
            brptMin =  inf;
            brptMax = -inf;
            if any(brptInc)
                brptMinInc = min(brptInc);
                brptMin = min(brptMin, brptMinInc);
                
                brptMaxInc = max(brptInc);
                brptMax = max(brptMax, brptMaxInc);
            end
            if any(brptDec)
                brptMinDec = min(brptDec);
                brptMin = min(brptMin, brptMinDec);
                
                brptMaxDec = max(brptDec);
                brptMax = max(brptMax, brptMaxDec);
            end
        end % breakpt
        
        function sigma = trqsol(x, p, delta)
            %TRQSOL  Largest solution of the TR equation.
            %     This subroutine computes the largest (non-negative)
            %     solution of the quadratic trust region equation
            %
            %           ||x + sigma*p|| = delta.
            %
            %     The code is only guaranteed to produce a non-negative
            %     solution if ||x|| <= delta, and p != 0. If the trust
            %     region equation has no solution, sigma = 0.
            ptx = p'*x;
            ptp = p'*p;
            xtx = x'*x;
            dsq = delta^2;
            
            % Guard against abnormal cases.
            
            rad = ptx^2 + ptp*(dsq - xtx);
            rad = sqrt(max(rad,0));
            
            if ptx > 0
                sigma = (dsq - xtx)/(ptx + rad);
            elseif rad > 0
                sigma = (rad - ptx)/ptp;
            else
                sigma = 0;
            end
        end % trqsol
        
    end % private static methods
    
end % class