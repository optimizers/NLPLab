classdef LBFGSPrecSolver < solvers.LBFGSSolver
    %% LBFGSPrecSolver
    %  Subclass of LBFGSSolver to take in account preconditioning
       
    methods (Access = public)
        
        function o = LBFGSPrecSolver(nlp, varargin)
            %% Constructor
            o = o@solvers.LBFGSSolver(nlp, varargin{:});
            o.cauchyBacktrack = true;
        end % constructor
        
        %% Operations and constraints handling
        
        function [breakpoints, d, F, indFree] = cauchyDirection(o, x, g)
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
            breakpoints = o.brkpt(x, -g);
            
            % Compute a direction whose coordinates are zeros if the
            % associated constraints are exposed.
            indFree = (abs(breakpoints) >= eps);
            d = zeros(o.nlp.n, 1);
            d(indFree) = -g(indFree);
            d = o.nlp.precTimes(d);
            d(~indFree) = 0;
            
            % Compute the indices of the breakpoints sorted form the
            % smallest breakpoint to the greatest
            if nargout > 2
                [~, F] = sort(breakpoints);
                F = F(nnz(~indFree)+1:end);
            end
        end

        function [s, iter, info, failed] = subspaceMinimization(o, x, g, xc, c, indFree, nFree)
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
            
            iterMax = o.maxIterCg;
            iter = 0;
            
            % Initialize residual and descent direction
            [Mc, failed] = o.mtimes(c);
            if failed
                s = [];
                info = -1;
                return
            end
            rc = g + o.theta * (xc - x) ...
                - o.wtimes(Mc);
            r = rc(indFree);
            
            % Check if initial point is the solution
            normRc = norm(r);
            o.logger.debug(sprintf('||rc|| = %9.3e', normRc));
            if normRc <= o.aOptTol
                s = xc - x;
                info = 0;
                o.logger.debug('Exit CG: xc is the solution');
                return
            end
            
            epsilon = min(o.cgTol, sqrt(normRc)) * normRc;
            z = o.nlp.precSubTimes(r, indFree);
            p = -z;
            d = zeros(length(r),1);
            rho2 = r .'* z;
            
            while true
                iter = iter + 1;
                
                % Check exit condition
                if sqrt(rho2) < epsilon
                    info = 0;
                    o.logger.debug(sprintf('||r|| = %9.3e', sqrt(rho2)));
                    o.logger.debug('Exit CG: Convergence');
                    break;
                end
                if iter > iterMax
                    info = 2;
                    o.logger.debug('Exit CG: Max iteration number reached');
                    break;
                end
                if toc(o.solveTime) > o.maxRT
                    info = 3;
                    o.logger.debug('Exit CG: Max runtime reached');
                    break;
                end
                % Compute the minimal breakpoint
                %alf1 = min( o.brkpt(xc(indFree) + d, p, indFree, nFree) );
                alf1 = Inf;
                
                % Compute step length
                [q, failed] = o.btimes(p, indFree);
                if failed
                    s = [];
                    info = -1;
                    return
                end
                alf2 = rho2 / (p.' * q);
                
                % Exit if we hit the domain boundary
                if alf2 > alf1
                    d = d + alf1 * p;
                    info = 1;
                    o.logger.debug('Exit CG: Constraints violated');
                    break;
                end
                
                % Prepare new step
                d = d + alf2 * p;
                r = r + alf2 * q;
                z = o.nlp.precSubTimes(r, indFree);
                rho1 = rho2;
                rho2 = (r.' * z);
                beta = rho2 / rho1;
                p = -z + beta * p;
            end
            s = xc  -x;
            alf1 = min( o.brkpt(xc(indFree), d, indFree, nFree) );
            s(indFree) = s(indFree) + alf1 * d;
            %s(indFree) = s(indFree) + d;
            failed = false;
            o.logger.debug(sprintf('||s|| = %9.3e', norm(s)));
            o.logger.debug(' -- Leaving Subspace minimization -- ');
        end
        
        %% L-BFGS operations
        
        function failed = updateLBFGS(o, s, y)
            %% UpdateLBFGS - Add a new pair to the pseudohessian
            % Store the new pair {y, s} into the L-BFGS approximation
            % Discard the oldest pair if memory has been exceeded
            % The matrices D, L, StS and J are also updated
            
            o.logger.debug('-- Entering updateLBFGS --');
            
            ys = dot(y, s);
            yH0y = dot(y, o.nlp.precTimes(y));
            
            if ys <= eps * max(yH0y, 1)
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
            
            B0s = o.nlp.precDiv(s);
            stB0s = dot(s, B0s);
            
            % Set the column indices where to put the new pairs
            if o.cl < o.mem
                o.cl = o.cl + 1;
                o.insert = mod(o.hd + o.cl - 2, o.mem) + 1;
            else
                o.insert = mod(o.insert, o.mem) + 1;
                o.hd = mod(o.hd, o.mem) + 1;
                
                % Move old information in sy and ss
                o.ss(o.triup) = o.ss(o.tridown);
                o.sy(o.triup') = o.sy(o.tridown');
            end
            
            % Update S and Y matrices and the scaling factor theta
            o.ws(:,o.insert) = B0s;
            o.wy(:,o.insert) = y;
            o.theta = yH0y / ys;
            o.iprs = mod(o.hd - 1 : o.hd + o.cl - 2, o.mem) + 1;
            
            % Add new information in sy and ss
            o.sy(o.cl,1:o.cl-1) = s.' * o.wy(:,o.iprs(1:end-1));
            o.ss(1:o.cl-1,o.cl) = o.ws(:,o.iprs(1:end-1)).' * s;
            
            o.ss(o.cl,o.cl) = stB0s;
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
                    v = o.theta * o.nlp.precDiv(v) - o.wtimes(p);
                else % Reduced vectors
                    p = o.wtrtimes(v, ind);
                    [p, failed] = o.mtimes(p);
                    if failed
                        % The Mtimes function has returned an error
                        return
                    end
                    v = o.theta * o.nlp.precSubDiv(v, ind) ...
                        - o.wtimes(p, ind);
                end
            else
                if nargin < 3
                    v = o.theta * o.nlp.precDiv(v);
                else
                     v = o.theta * o.nlp.precSubDiv(v, ind);
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