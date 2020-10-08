function [XX, XY,Yn,allstim,n,T] = MkSuffStats_BilinReg_Sims(Z,X,hk)


[n,T,N] = size(Z);
% P = cols(X);
P = size(X,2);
allstim = cell(n,1);
ntrials = sum(hk,1);
Yn = cell(n,1); % neural responses (each cell has resps for one neuron)
for ii = 1:n  % loop over neurons
    allstim{ii} = zeros(ntrials(ii),P); % stim vals for each trial
    for jj = 1:P
        smps = X(hk(:,ii)==1,jj);
        allstim{ii}(:,jj) = smps;
    end
    Yn{ii} = squeeze(Z(ii,:,(hk(:,ii)==1)));
end
It = speye(T);  % sparse matrix of size for 1 trial for one variable
XXperneuron = cell(1,n); % cell array for XX terms
XYperneuron = zeros(T*P,n); % matrix for XY terms
for jx = 1:n
    stm = allstim{jx}; % grab relevant stimuli
    XXperneuron{jx} = kron(stm'*stm,It); % (multiply before kron)
    XYperneuron(:,jx) = vec(reshape(Yn{jx},T,ntrials(jx))*stm); % (avoids kron)
end
% Construct block-diagonal XX matrix and XY vector
XX = blkdiag(XXperneuron{:});  
XY = XYperneuron(:);

% Permute indices so coefficients grouped by matrix instead of by neuron
% Map [ntrials x nx 
nwtot = P*T*n;
iiperm = reshape(1:nwtot,T,P,n);
iiperm = vec(permute(iiperm,[1 3 2]));
XX = XX(iiperm,iiperm);
XY = XY(iiperm);
