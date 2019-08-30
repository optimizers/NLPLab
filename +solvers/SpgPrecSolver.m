classdef SpgPrecSolver < solvers.SpgSolver
    %% SpgSolver - Calls the MinConf_SPG solver
    % Original documentation follows:
    % ---------------------------------------------------------------------
    % function [x,f,self.nObjFunc,self.nProj] = minConf_SPG(funObj,x,
    % funProj, options)
    %
    % Function for using Spectral Projected Gradient to solve problems of
    % the form
    %   min funObj(x) s.t. x in C
    %
    %   @funObj(x): function to minimize (returns gradient as second
    %               argument)
    %   @funProj(x): function that returns projection of x onto C
    %
    %   options:
    %       verbose: level of verbosity (0: no output, 1: final,
    %                     2: iter (default), 3: debug)
    %       optTol: tolerance used to check for optimality (default: 1e-5)
    %       progTol: tolerance used to check for lack of progress (default:
    %                1e-9)
    %       maxIter: maximum number of calls to funObj (default: 500)
    %       numDiff: compute derivatives numerically (0: use user-supplied
    %       derivatives (default), 1: use finite differences, 2: use
    %                                 complex differentials)
    %       suffDec: sufficient decrease parameter in Armijo condition
    %       (default: 1e-4)
    %       interp: type of interpolation (0: step-size halving, 1:
    %       quadratic, 2: cubic)
    %       memory: number of steps to look back in non-monotone Armijo
    %       condition
    %       useSpectral: use spectral scaling of gradient direction
    %       (default: 1)
    %       projectLS: backtrack along projection Arc (default: 0)
    %       testOpt: test optimality condition (default: 1)
    %       feasibleInit: if 1, then the initial point is assumed to be
    %       feasible
    %       bbType: type of Barzilai Borwein step (default: 1)
    %
    %   Notes:
    %       - if the projection is expensive to compute, you can reduce the
    %           number of projections by setting self.testOpt to 0    
    
    methods (Access = public)
        
        function self = SpgPrecSolver(nlp, varargin)
            %% Constructor
            % Inputs:
            %   - nlp: a subclass of a nlp model containing the 'obj'
            %   function that returns variable output arguments among the
            %   following: objective function, gradient and hessian at x.
            %   The hessian can be a Spot operator if it is too expensive
            %   to compute. The method also supports a L-BFGS approximation
            %   of the hessian.
            %   - varargin (optional): the various parameters of the
            %   algorithm
            
            self = self@solvers.SpgSolver(nlp, varargin{:});
        end % constructor
        
    end % Public methods
    
    
    methods (Access = protected)
        
        function d = descentdirection(self, alph, x, g)
            %% DescentDirection - Compute the search direction
            %  Inputs:
            %  - alph: the Barzilai-Borwein steplength
            %  - g: objective gradient
            Iplus = ((self.nlp.bU == x) & (g < 0) |...
                (x == self.nlp.bL) & (g > 0));
            
            d = -alph * g;
            d(Iplus) = 0;
            d = self.nlp.precTimes(d);
            d(Iplus) = 0;
        end
            
        function alph = bbstep(self, s, y)
            %% BBStep - Compute the spectral steplength
            %  Inputs:
            %  - s: step between the two last iterates
            %  - y: difference between the two last gradient
           
            if self.bbType == 1
                alph = (s' * self.nlp.precDiv(s)) / (s' * y);
            else
                alph = (s' * y) / (y' * self.nlp.precTimes(y));
            end
        end
        
        function s = gpstep(self, x, g)
            %% GPStep - computing the projected gradient
            % Inputs:
            %   - x: current point
            %   - g: gradient at x
            % Output:
            %   - s: projected gradient
            % Calling project to increment projection counter
            s = self.project(x - g) - x;
        end
        
        function z = project(self, x)
            %% Project - projecting x on the constraint set
            [z, solved] = self.nlp.project(x);
            if ~solved
                % Propagate throughout the program to exit
                self.iStop = self.EXIT_PROJ_FAILURE;
            end
            self.nProj = self.nProj + 1;
        end
        
        function [xNew, fNew, gNew, t, failed] = backtracking(self, ...
                x, xNew, f, fNew, g, gNew, d, t, funRef)
            % Backtracking Line Search
            failed = false;
            iterLS = 1;
            while fNew > funRef + self.suffDec* g' * (xNew - x)
                t = t / 2;
                
                % Check whether step has become too small
                if max(abs(t * d)) < self.aFeasTol * norm(d) ...
                        || t == 0 || iterLS > self.maxIterLS
                    failed = true;
                    t = 0;
                    xNew = x;
                    fNew = f;
                    gNew = g;
                    return;
                end
                
                if self.projectLS
                    % Projected linesearch
                    xNew = self.project(x + t * d);
                else
                    xNew = x + t * d;
                end
                
                [fNew, gNew] = self.nlp.obj(xNew);
                iterLS = iterLS + 1;
            end
        end % backtracking
        
    end % private methods
    
end % class