classdef opLBFGSB < opSpot
%OPLBFGSB Store a L-BFGS approximation as recommendended in the L-BFGS-B
%article

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    properties (SetAccess = protected)
        mem;    % Maximum number of stored updates
        b;      % First index of stored pairs
        s;      % Array of s vectors
        y;      % Array of y vectors
        theta;  % Scaling parameter
        l;      % Matrix of the 
        sts;    % Matrix of the StS
        J;      % cholesky factorization of the middle matrix
        d;      % Diagonal of the D matrix containting the ys
    end
    
    properties (SetAccess = public)
        rejects;        % Number of rejects
        updates;        % number of update attempts
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    methods
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function op = opLBFGSB(n, mem)
            %opLBFGSB constructor
            if nargin == 1
                mem = 1;
            end
            if nargin > 2
                error('At most one argument must be specified');
            end
            % Check if input is an integer
            if ~(isnumeric(mem) || mem ~= round(mem))
               error('Memory parameter must be an integer.');
            end
            
            % Create object
            op = op@opSpot('L-BFGS-B', n, n);
            op.cflag = false;
            op.sweepflag = true;
            op.mem = min(max(mem, 1), n);
            
            % Initialize parameters
            op.updates = 0;
            op.rejects = 0;
            op.b       = op.mem + 1;
            op.theta   = 1;
            
            % Allocate memory
            op.s = zeros(n, op.mem);
            op.y = zeros(n, op.mem);
            op.l = zeros(op.mem, op.mem);
            op.d = zeros(op.mem,1);
            op.sts = zeros(op.mem, op.mem);
            op.J = [];
        end % function opLBFGS
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function op = update(op, s, y)
            %UPDATE Store the new pair {y, s} into the L-BFGS approximation
            % Discard the oldest pair if memory has been exceeded
            % The matrices D, L, StS and J are also updated
            
            op.updates = op.updates + 1;
            ys = dot(s,y);
            
            if ys <= 1.0e-20
                warning('L-BFGS: Rejecting {s, y} pair');
                op.rejects = op.rejects + 1;
            else
                % Update S and Y
                op.s = [op.s(:, 2:end), s];
                op.y = [op.y(:, 2:end), y];
                %op.theta = ys / (y' * y);
                
                if op.b > 1
                    op.b = op.b - 1;
                end
               
                
                % Update D
                op.d = [op.d(2:end); ys];
                
                % Update StS
                v = op.s.' * s;
                op.sts = [op.sts(2:end, 2:end), v(1:(end-1)) ; v.'];
                
                % Update L
                v = s.' * op.y(:, 1:(end-1));
                op.l = [op.l(2:end, 2:end), zeros(op.mem-1,1); ...
                    v, 0];
                
                
                % Update J
                op.J = chol(op.theta * op.StS + op.L * (op.D \ op.L.'), ...
                    'lower');
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function p = Wtimes(op, v, mode)
            %WTIMES Compute the product by W = [Y , theta * S]
            if mode == 1
                nPairs = op.mem - op.b + 1;
                p = op.Y * v(1:nPairs,:) ...
                    + op.theta * op.S * v((nPairs+1):(2*nPairs), :);
            elseif mode == 2
                p = [op.Y.' * v ; op.theta * op.S.' * v];
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function p = Mtimes(op, v, ~)
            %MTIMES Compute the product with the middle matrix of the
            % L-BFGS formula
            nPairs = op.mem - op.b + 1;
            sqD = sqrtD(op);
            LD = op.L / sqD;
            
            p = [sqD, zeros(nPairs) ; -LD, op.J] \ v;
            p = [-sqD, LD.' ; zeros(nPairs) , op.J.'] \ p;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function vBv = scalPro(op, v)
            % BQUAD Compute the scalar product of the vector v with itself
            % through the operator B
            p = op.Wtimes(v, 2);
            Mp = op.Mtimes(p, 1);
            vBv = op.theta * (v.'* v) - p.'*Mp;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function wb = Wb(op, b)
            %WB Return a line of the W matrix
            wb = [op.y(b, op.b:op.mem), ...
                op.theta * op.s(b, op.b:op.mem)];
        end
        
    end % Public methods
                
    methods (Access = protected)
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function yr = Y(op)
            yr = op.y(:, op.b:end);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function sr = S(op)
            sr = op.s(:, op.b:end);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function Dr = D(op)
            nPairs = op.mem - op.b + 1;
            Dr = spdiags(op.d(op.b:end), 0, nPairs, nPairs);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function sqDr  = sqrtD(op)
            nPairs = op.mem - op.b + 1;
            sqDr = spdiags(sqrt(op.d(op.b:end)), 0, nPairs, nPairs);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function Lr = L(op)
            Lr = op.l(op.b:end, op.b:end);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function StSr = StS(op)
            StSr = op.sts(op.b:end, op.b:end);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function Bv = multiply(op, v, ~)
            % MULTIPLY Multiply operator with a vector.
            Bv = op.Wtimes(v, 2); % Wt * v
            Bv = op.Mtimes(Bv, 1);
            Bv = op.theta * v - op.Wtimes(Bv, 1);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end
end

