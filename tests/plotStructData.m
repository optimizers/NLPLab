function plotStructData(data, HEADER, FACTOR)
dataFields = fields(data);
% Assuming that each data.entry contains a matrix of same size.
% Increase markers if needed...
cc = {'bo-', 'go-', 'ro-', 'co-', 'mo-', 'ko-', 'bd-.', 'gd-.', 'rd-.'};
for varInd = 2:size(HEADER, 2)
    % Handling each column seperatly, 1st is the x, others are ys
    h = figure;
    
    figName = [HEADER{1}, HEADER{varInd}];
    figName = ['data/fact', num2str(FACTOR), ...
        figName(isstrprop(figName, 'alpha')), '.eps'];
    
    fieldInd = 1;
    for temp = dataFields'
        mat = data.(temp{1});
        semilogy(mat(:, 1), mat(:, varInd), cc{fieldInd});
        hold on;
        fieldInd = fieldInd + 1;
    end
    hold off;
    xlabel(HEADER{1});
    ylabel(HEADER{varInd});
    axis tight;
    
    legend(dataFields{:}, 'Location', 'BestOutside');
    % Save figure
    print(h, '-depsc', figName);
end
end