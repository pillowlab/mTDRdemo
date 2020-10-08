
function negloglik = neglogLikBTDR_IncompObs_uneqvar_S_nllonly(pars,Ai,Xzetai,zzi,r,ni,g)
% Negative marginal log-likelihood for basis of population regression
% model
%
%[negloglik, grad] = neglogLikBTDR_IncompObs_uneqvar_S_fast(pars,Ai,Ri,zzi,r,ni,g)
%
%Computes negative log-likelihood under linear Gaussian regression model
%where if y = vec(Y) then
% Y = x1*W1*S1 + ... +xP*WP*SP + n, n~N(0,nsevar)
% vec([S1;...SP]) ~ N(0,I)
% where Sp's have been marginalized out.
% INPUT:
% ------
%     pars [(T*sum(r)+ n) x 1] - vector of parameters [nsevar; s1; ..., sP]
%     Ai - [PxPxn] - regressor covariance matrix of neuron i
%     Ri - [TP x TPxn] - regressor-weighted auto and cross covariance
%     zzi- [scalar] - total observation variance
%     r  - [P x 1] - vector giving number of columns for each Wp matrix
%     ni - [scalar] -  total number of observations
%      g - [scalar] - ridge regularization parameter
%
% OUTPUT:
% ------
%   negloglik - negative marginal likelihood
%   grad - gradient
%   Hess - hessian matrix
% Last updated: MCA 06/09/16

%% parameters needed for calculation

% Unpack parameters
P = length(r);
TP = length(Xzetai(:,1));
T  = TP/P;
rtot = sum(r);
n = length(ni);
lambi = pars(1:n);

% Make block diagonal S matrix
S = [];
endind = n;
for p = 1:P %all other blocks
    startind = endind +1;
    endind = startind + T*r(p) - 1;
    S = blkdiag(S,reshape(pars(startind:endind),T,r(p))');
end

%% negative log likelihood only


% Make Ci matrices for all neurons
% lambAiIS = zeros(T*P,sum(r),n);
% for i = 1:n
%     lambAiIS(:,:,i) = lambi(i)*kronmult({speye(T),Ai(:,:,i)},S');
% end


% lambAiIS := lambi*kron(I,Ai)*S'
lambAiIS = zeros(T*P,n,sum(r));
lambAi = bsxfun(@times,Ai,permute(repmat(lambi,1,P),[3 2 1]));
for p = 1:sum(r)
    try
        M = mmx_mkl_single('mult',repmat(reshape(S(p,:)',T,P),1,1,n),lambAi);
    catch
        M = slowMult(repmat(reshape(S(p,:)',T,P),1,1,n),lambAi);
    end
    lambAiIS(:,:,p) = reshape(permute(M,[3 1 2]),n,T*P)';
end

% Ci := lambi*S*kron(I,Ai)*S' + eye(sum(r))
lambAiIS = permute(lambAiIS,[1 3 2]);
lambiSAiSold = reshape(S*reshape(lambAiIS,TP,rtot*n),rtot,rtot,n);
Ci = bsxfun(@plus,lambiSAiSold,eye(rtot));

% Quadratic term
SXzetai  = S*Xzetai;
try
    invCiSXzetai = mmx_mkl_single('backslash',Ci,reshape(SXzetai,sum(r),1,n));
    Qtermi = mmx_mkl_single('mult',reshape(SXzetai,sum(r),1,n),invCiSXzetai,'tn');
catch
    invCiSXzetai = slowBackslash(Ci,reshape(SXzetai,sum(r),1,n));
    Qtermi = slowMult(permute(reshape(SXzetai,sum(r),1,n),[2 1 3]),invCiSXzetai);
end
Qtermi = squeeze(Qtermi);

% Terms of log-likelihood
try
    cholCi = mmx_mkl_single('chol',Ci,[]);
catch
    cholCi = slowChol(Ci);
end
 
ind = repmat(logical(speye(rtot)),1,n); % indices for identity for each Ci-size matrix
logdetterm = 2*sum(log(cholCi(ind)));

% noncumulative operations
Qterm = (lambi.^2)'*Qtermi;
regterm = g*pars(n+1:end)'*pars(n+1:end);
negloglik = .5*( -T*ni*log(lambi) + logdetterm + ...
    zzi'*lambi - Qterm + regterm);