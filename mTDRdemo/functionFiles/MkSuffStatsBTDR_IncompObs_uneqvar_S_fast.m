function [Ri,Ai,zzi,ni,Xi,Xzetai,zi] = MkSuffStatsBTDR_IncompObs_uneqvar_S_fast(X,Z,hk)
% Calculate the sufficient statistics for Bayesian TDR under the model of
% incomplete observations on each trial.
%
% [Ri,Ai,zzi,ni] = MkSuffStatsBTDR_IncompObs_uneqvar_S(X,Z,hk)
% N = # trials
% P  = # input varaibles
% n  = # neurons
% n' = # observed trials over all neurons  = sum(nk) where nk is # neurons
% observed on trial k
%
% INPUT:
% ------
%       X [N x P] - matrix of regressors
%       Z [nk x T] x N - Cell array of observed responses
%       hk [N x n] - hk(i,j) = binary array indicating which neurons were
%       observed on which trials
%
% OUTPUT:
% ------
%       Ri [nP x nP] - regressor-weighted population covariance
%       Ai [P x P] - Stimulus covariance
%       zzi [scalar] - squared Euclidian norm of all observations
%       ni           - number of observatons
% Last updated: MCA 06/06/16

[n,T,N] = size(Z);
%P = cols(X);
P = size(X,2);
ni = sum(hk,1);
zi = cell(n,1);
Xi = cell(n,1);
% Make a separate A,and R for each neuron
Ai = zeros(P,P,n); Ri = zeros(P*T,P*T,n);  zzi = zeros(n,1);Xzetai = zeros(T*P,n);
for i = 1:n
    Xi{i} = X(hk(:,i)==1,:);
    Ai(:,:,i)  = Xi{i}'*Xi{i};
    zi{i} = squeeze(Z(i,:,hk(:,i)==1));
    zetai  = vec(zi{i});
    Xzetai(:,i) = kronmult({eye(T),Xi{i}'},zetai);
    Ri(:,:,i)  = Xzetai(:,i)*Xzetai(:,i)';
    zzi(i) = zetai'*zetai;
end