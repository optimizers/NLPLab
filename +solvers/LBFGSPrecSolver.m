classdef LBFGSPrecSolver < solvers.LBFGSSolver
    %% LBFGSPrecSolver
    %  Subclass of LBFGSSolver to take in account preconditioning
       
    methods (Access = public)
        
        function self = LBFGSPrecSolver(nlp, varargin)
            %% Constructor
            self = self@solvers.LBFGSSolver(nlp, varargin{:});
            self.inexcauchy = true; % Ensure we use the inexact Cauchy search
        end % constructor
        
        %% Operations and constraints handling
        
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
            d = self.nlp.precTimes(d);
            d(~indFree) = 0;

            if nargout > 2
                [~, F] = sort(breakpoints);
                F = F(nnz(~indFree)+1:end);
            end
        end
        
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
            z = self.nlp.precSubTimes(r, indFree);
            p = -z;
            d = zeros(length(r),1);
            rho2 = r .' * z;

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
                z = self.nlp.precSubTimes(r, indFree);
                rho1 = rho2;
                rho2 = (r.' * z);
                beta = rho2 / rho1;
                p = -z + beta * p;
            end

            s = xc - x;
            alf = min( self.brkpt(xc(indFree), d, indFree, nFree) );
            s(indFree) = s(indFree) + min(alf, 1) * d;
            failed = false;
            self.logger.debug(sprintf('||s|| = %9.3e', norm(s)));
            self.logger.debug(' -- Leaving Subspace minimization -- ');
        end

        %% L-BFGS operations
        
        function [ys, yy] = dotProds(self, s, y)
            %% DotProds - Prepare the dot products y'H0y and y's
            ys = dot(y, s);
            yy = dot(y, self.nlp.precTimes(y));
        end
        
        function dtd = updateW(self, s, y, ys, yy)
            %% UpdateW - Update ws, wy, theta and return s'B0s
            B0s = self.nlp.precDiv(s);
            dtd = dot(s, B0s);
            self.ws(:,self.insert) = B0s;
            self.wy(:,self.insert) = y;
            self.theta = min(yy / ys, 1);
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
                    v = self.theta * self.nlp.precDiv(v) - self.wtimes(p);
                else % Reduced vectors
                    p = self.wtrtimes(v, ind);
                    [p, failed] = self.mtimes(p);
                    if failed
                        % The Mtimes function has returned an error
                        return
                    end
                    v = self.theta * self.nlp.precSubDiv(v, ind) ...
                        - self.wtimes(p, ind);
                end
            else
                if nargin < 3
                    v = self.theta * self.nlp.precDiv(v);
                else
                     v = self.theta * self.nlp.precSubDiv(v, ind);
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