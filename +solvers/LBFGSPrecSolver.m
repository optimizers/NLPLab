classdef LBFGSPrecSolver < solvers.LBFGSSolver
    %% LBFGSPrecSolver
    %  Subclass of LBFGSSolver to take in account preconditioning
       
    methods (Access = public)
        
        function o = LBFGSPrecSolver(nlp, varargin)
            %% Constructor
            o = o@solvers.LBFGSSolver(nlp, varargin{:});
        end % constructor
        
        %% Operations and constraints handling
        
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
            
            % Compute breakpoints along the projected gradient path
            
            % Compute a direction whose coordinates are zeros if the
            % associated constraints are exposed.
            d = zeros(o.nlp.n, 1);
            d(indFree) = -g(indFree);
            d = o.nlp.precTimes(d);
            d(~indFree) = 0;
        end

        function [s, iter, info, failed] = subspaceMinimization(o, x, g, s, indFree, nFree)
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
            failed = false;
            
            % Initialize residual and descent direction
            rc = g + o.btimes(s);
            r = rc(indFree);
            
            % Check if initial point is the solution
            normRc = norm(r);
            o.logger.debug(sprintf('||rc|| = %9.3e', normRc));
            
            if normRc <= o.aOptTol
                info = 0;
                o.logger.debug('Exit CG: xc is the solution');
                return
            end
            
            epsilon = min(o.cgTol, sqrt(normRc)) * normRc;
            z = o.nlp.precSubTimes(r, indFree);
            p = -z;
            d = zeros(nFree,1);
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
                %alf1 = min( o.brkpt(x(indFree) + s(indFree) + d, ...
                %    p, indFree, nFree) );
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
            
            % Find smallest breakpoint in the found direction
            alf1 = min( o.brkpt(x(indFree) + s(indFree),...
                d, indFree, nFree) );
            s(indFree) = s(indFree) + alf1 * d;
            %s(indFree) = s(indFree) + d;
            
            o.logger.debug(sprintf('||s|| = %9.3e', norm(s)));
            o.logger.debug(' -- Leaving Subspace minimization -- ');
        end
        
        %% L-BFGS operations
        
        function [ys, yy] = dotProds(o, s, y)
            %% DotProds - Prepare the dot products y'H0y and y's
            ys = dot(y, s);
            yy = dot(y, o.nlp.precTimes(y));
        end
        
        function dtd = updateW(o, s, y, ys, yy)
            %% UpdateW - Update ws, wy, theta and return s'B0s
            B0s = o.nlp.precDiv(s);
            dtd = dot(s, B0s);
            o.ws(:,o.insert) = B0s;
            o.wy(:,o.insert) = y;
            o.theta = yy / ys;
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