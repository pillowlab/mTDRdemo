function Y = SimPopData(Xsamp,BB,d, n,T,N)
% Function for generating observations from the BTDR model.
% INPUTS: 
% Xsamp = [N x P] matrix of regressors on each trial
%    BB = [P x 1] cell array where each element is a n x T matrix of coefs
%     d = [n x 1] vector of noise variances for each neuron 
%     n = number of neurons
%     T = number of time points
%     N = number of trials
%
% OUTPUTS:
% Y = [n x T x N] array of observations


% Sample trajectories
Y = zeros(n,T,N);
for k = 1:N
    if length(d)==1
        noisek = randn(n,T)./sqrt(d);
    else
        noisek = diag(1./sqrt(d))*randn(n,T);
    end
    Y(:,:,k) = kronmult({eye(n),Xsamp(k,:)},BB)+ noisek;
end
