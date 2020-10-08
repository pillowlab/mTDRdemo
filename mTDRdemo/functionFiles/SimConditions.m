function [Xsamp, Xcond]= SimConditions(var_uniq,N)

% Create array of all possible conditions
P = length(var_uniq);
levP = zeros(P,1);
for p = 1:P
    levP(p) = numel(var_uniq{p}); %number of levels per covariate
end

switch P
    case 1
        Xcond = ndgrid(var_uniq{:});
    case 2
        [x1, x2] = ndgrid(var_uniq{:});
        Xcond = [x1(:) x2(:)]; %All possible experimental conditions
        
    case 3
        [x1, x2, x3] = ndgrid(var_uniq{:});
        Xcond = [x1(:) x2(:) x3(:)]; %All possible experimental conditions
    case 4
        [x1, x2, x3, x4] = ndgrid(var_uniq{:});
        Xcond = [x1(:) x2(:) x3(:) x4(:)]; %All possible experimental conditions
    case 5
        [x1, x2, x3, x4,x5] = ndgrid(var_uniq{:});
        Xcond = [x1(:) x2(:) x3(:) x4(:) x5(:)]; %All possible experimental conditions
end
% fprintf('%6.0f possible experiments\n',length(Xcond))

% Sample conditions
Xsamp = Xcond(datasample(1:prod(levP),N),:);
