classdef MakeProjectModel < handle
    %% MakeProjectModel
    % Metaclass that creates subclasses of a given NlpModel providing a
    % project function on the constraint set.
    %
    % To call, make sure nlplab is own MATLAB's path and:
    % >> import utils.MakeProjectModel
    % >> utils.MakeProjectModel('someModel', 'someProjModel')
    % 'someModel' & 'someProjModel' must be under +model/.
    % Done! The new model should be located under +model/
    
    
    properties (SetAccess = private)
        % Name of the new class that will be created
        newClassName;
        % Name of the NLP model to which a project function is added
        modelName;
        % Name of the project handle class that defines the project func
        projName;
    end
    
    
    properties (Constant, Hidden = true)
        % General class template that will be written in a .m file
        GEN_FORMAT = [ ...
            'classdef %s < %s & %s\n', ...
            '    methods (Access = public)\n', ...
            '        function self = %s(varargin)\n', ...
            '            self = self@%s(varargin{:});\n', ...
            '        end\n', ...
            '    end\n', ...
            'end\n'];
    end
    
    
    methods (Access = public)
        
        function self = MakeProjectModel(modelName, projName)
            %% Constructor
            
            % Append 'model.' prefix if missing
            if isempty(strfind(modelName, 'model.'))
                modelName = ['model.', modelName];
            end
            if isempty(strfind(projName, 'model.'))
                projName = ['model.', projName];
            end
            
            % Checking if both entries are valid models
            try
                try
                    import(modelName);
                catch
                    error('Model should be in the +model package');
                end
                % Ensure that modelName represents a NlpModel
                super = superclasses(modelName);
                if isempty(strfind(super, 'model.NlpModel'))
                    error('Model isn''t a subclass of NlpModel');
                end
            catch ME
                rethrow(ME);
            end
            try
                import(projName);
            catch
                error('Projection type should be in the +model package');
            end
            
            self.modelName = modelName;
            self.projName = projName;
            
            % Creating the new classname
            posModel = strfind(self.modelName, '.') + 1;
            posProj = strfind(self.projName, '.') + 1;
            self.newClassName = [self.projName(posProj : end), ...
                self.modelName(posModel : end)];
            
            % Building the new class
            self.buildVariant();
        end
        
    end
    
    
    methods (Access = private)
        
        function classText = formatTemplate(self)
            %% FormatTemplate - creating text to write in .m file
            
            % Filling the general template with provided arguments
            classText = sprintf(self.GEN_FORMAT, ... % General template
                self.newClassName, ... % Classdef
                self.modelName, ... % Inheritance 1
                self.projName, ... % Inheritance 2
                self.newClassName, ... % Constructor
                self.modelName); % Call to parent class' constructor
        end
        
        function buildVariant(self)
            %% BuildVariant - creating a .m file for the new class
            % Saving under +model folder
            modelLoc = which('model.NlpModel');
            pos = strfind(modelLoc, '/');
            % Keeping path to +model/
            modelLoc = modelLoc(1 : pos(end));
            % Complete filename with path to +model/
            fname = [modelLoc, self.newClassName, '.m'];
            % Checking if file already exists
            chk = exist(fname, 'file');
            
            if chk == 0
                % If it doesn't exist, create it
                fid = fopen(fname, 'w');
                code = self.formatTemplate();
                fprintf(fid, code);
                fclose(fid);
                rehash path
            elseif chk == 2
                % If it exists, return
                return
            else
                error('Problem during buildVariant');
            end
            
        end
        
    end
    
end