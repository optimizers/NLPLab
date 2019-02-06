function classify(dirname, varargin)
%CLASSIFY  List of problems in a directory and their characteristics.
%
% classify('dir name')

p = inputParser;
p.addParameter('criterion','');
p.addParameter('file','');
p.parse(varargin{:});

criterion = p.Results.criterion;
outFile = p.Results.file;

if ~isempty(outFile)
   fid = fopen(outFile,'w');
else
   fid = 1;
end

% Log header and body formats.
logH = '\n%15s  %6s  %6s  %6s  %6s  %6s  %6s  %6s  %15s\n';
logB =   '%15s  %6i  %6i  %6i  %6i  %6i  %6i  %6i  %15s\n';
logT = {'name','n','m','n_bnd','m_bnd','m_fix','m_nln','m_lin','desc'};

d = dir(fullfile(dirname,'*.nl'));

fprintf(fid,logH,logT{:});
fprintf(fid,'%s\n',repmat('-',length(sprintf(logH,logT{:})),1));

for i = 1:length(d)

   % Problem and file name.
   pname = d(i).name;
   fname = fullfile(dirname, pname);

   % Instantiate problem.
   p = model.amplmodel(fname,true);

   % Gather stats.
   n = p.n;
   m = p.m;
   m_lin = sum(p.linear);
   m_nln = p.m - m_lin;
   m_fix = sum( p.iFix);
   m_bnd = sum(~p.iFix);
   n_bnd = sum(p.jLow | p.jUpp);
   
   if m == 0
      if n_bnd > 0
         desc = 'bnd constrained';
      else
         desc = 'unconstrained';
      end
   elseif n_bnd == 0 && m_bnd == 0
      desc = 'equality';
   else
      desc = '';
   end

   % Only print if it meets the criterion.
   if ~isempty(criterion) && ~strcmp(desc,criterion)
      continue
   end
   
   if mod(i,20) == 0
      fprintf(fid,logH,logT{:});
      fprintf(fid,'%s\n',repmat('-',length(sprintf(logH,logT{:})),1));
   end
   fprintf(fid,logB, p.name, n, m, n_bnd, m_bnd, m_fix, m_nln, m_lin, desc);
end

end
