classdef PrintInfo < handle
    %% PrintInfo
    % Object defining basic header/footer prints for solvers
    
    properties (Access = public)
        solverName;
    end
    
    methods (Access = public)
        
        function self = PrintInfo(solverName)
            %% Constructor
            self.solverName = solverName;
        end
        
        function footer(self, solver)
            %% Footer
            % Print footer, must receive a NlpSolver object
            
            if solver.verbose
                solver.printf('\nEXIT %s: %s\nCONVERGENCE: %d\n', ...
                    self.solverName, solver.EXIT_MSG{solver.iStop}, ...
                    solver.solved);
                solver.printf('||Pg|| = %8.1e\n', solver.pgNorm);
                solver.printf('Stop tolerance = %8.1e\n', solver.rOptTol);
            end
            
            if solver.verbose >= 2
                solver.printf('\n')
                solver.printf(' %-27s  %6i     %-17s  %15.8e\n', ...
                    'No. of iterations', solver.iter, ...
                    'Objective value', solver.fx);
                t1 = solver.nlp.ncalls_fobj + solver.nlp.ncalls_fcon;
                t2 = solver.nlp.ncalls_gobj + solver.nlp.ncalls_gcon;
                solver.printf(' %-27s  %6i     %-17s    %6i\n', ...
                    'No. of calls to objective' , t1, ...
                    'No. of calls to gradient', t2);
                solver.printf(' %-27s  %6i \n', ...
                    'No. of Hessian-vector prods', ...
                    solver.nlp.ncalls_hvp + solver.nlp.ncalls_hes);
                solver.printf('\n');
                tt = solver.solveTime;
                t1 = solver.nlp.time_fobj + solver.nlp.time_fcon;
                t1t = round(100 * t1/tt);
                t2 = solver.nlp.time_gobj + solver.nlp.time_gcon;
                t2t = round(100 * t2/tt);
                solver.printf([' %-24s %6.2f (%3d%%)  %-20s %6.2f', ...
                    '(%3d%%)\n'], 'Time: function evals' , t1, t1t, ...
                    'gradient evals', t2, t2t);
                t1 = solver.nlp.time_hvp + solver.nlp.time_hes;
                t1t = round(100 * t1/tt);
                solver.printf([' %-24s %6.2f (%3d%%)  %-20s %6.2f', ...
                    '(%3d%%)\n'], 'Time: Hessian-vec prods', t1, ...
                    t1t, 'total solve', tt, 100);
            end
            solver.printf('\n');
        end % printHeaderFooter
        
        function header(self, solver, map)
            %% Header
            % Prints header, must receive NlpSolver object. Can also
            % receive a map container of extra arguments to print
            
            solver.printf('\n');
            solver.printf('%s\n', ['*', repmat('-',1,58), '*']);
            solver.printf([repmat('\t', 1, 3), '%s\n'], self.solverName);
            solver.printf('%s\n\n', ['*', repmat('-',1,58), '*']);
            solver.printf(solver.nlp.formatting())
            solver.printf('\nParameters\n----------\n')
            solver.printf('%-15s: %3s %8.1e', 'aOptTol', '', ...
                solver.aOptTol);
            solver.printf('\t%-15s: %3s %8.1e\n', ' rOptTol', '', ...
                solver.rOptTol);
            solver.printf('%-15s: %3s %8.1e', 'aFeasTol', '', ...
                solver.aFeasTol);
            solver.printf('\t%-15s: %3s %8.1e\n', ' rFeasTol', '', ...
                solver.rFeasTol);
            solver.printf('%-15s: %3s %8d', 'maxIter', '', ...
                solver.maxIter);
            solver.printf('\t%-15s: %3s %8d\n', ' maxEval', '', ...
                solver.maxEval);
            
            if nargin == 3
                ind = 1;
                for key = map.keys
                    if mod(ind, 2) ~= 0
                        solver.printf('%-15s: %3s %8.1e', key{1}, '', ...
                            map(key{1}));
                    else
                        solver.printf('\t%-15s: %3s %8.1e\n', key{1}, '', ...
                            map(key{1}));
                    end
                    ind = ind + 1;
                end
            end
            if isprop(solver.nlp, 'projSolver')
               solver.printf('\nProjection Solver = %s', ...
                   class(solver.nlp.projSolver));
            end
            solver.printf('\n');
        end
        
    end
    
end