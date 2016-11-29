classdef testQPProjModel < model.qpmodel
    %% A simple test projection model
    % This projection model was made to test solvers requiring projections
    % on the constraint set, such as minConf & Cflash.
    %
    % FULL PROBLEM := { min_x c'*x + x'*Q*x : x \in X
    %
    % where X : { x | cU >= A*x >= cL }
    % 
    % The 'project' function defined in this class projects a vector on the
    % constraint set X, i.e. solves
    %
    % PROJ PROBLEM := { min_y 0.5*||y - x||^2 : cU >= A*y >= cL
    %
    % which is a part of the constrained problem
    % ---
    
    properties (Access = public)
        % cU; % Upper bound - defined in nlpmodel
        % cL; % Lower bound - defined in nlpmodel
        % A; % Jacobian of constraints - defined in qpmodel
        x; % Vector to project
        y; % Projection of x on the constraint set
    end
    
    methods (Access = public)
        function self = testQPProjModel(cL, cU, A, c, Q)
            %% Constructor
            % Building (useless) nlpmodel
            n = size(A, 2); % In case A is not square
            bL = -inf(n, 1);
            bU = inf(n, 1);
            x0 = zeros(n, 1);
            self = self@model.qpmodel('', x0, cL, cU, bL, bU, A, c, Q);
        end
        
        function y = project(self, x)
           %% Project x on the constraint set
           % Calling lsqlin to solve this problem; lsqlin solves
           %        { min_x 0.5 * || C*x - d ||^2 : A*x <= b
           % Therefore, we must do a few tricks...
           self.x = x;
           C = eye(length(x));
            % cL <= A*x <= cU <-> [A; -A] * x <= [cU; -cL]
           newA = [self.A; -self.A];
           b = [self.cU; -self.cL];
           d = x;
           options = optimoptions('lsqlin', 'Display', 'off', ...
               'Algorithm', 'interior-point');
           % Calling lsqlin
            y = lsqlin(C, d, newA, b, [], [], [], [], [], options);
           self.y = y;
        end
    end
end
