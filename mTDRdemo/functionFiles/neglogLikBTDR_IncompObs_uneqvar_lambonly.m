
function [negloglik, grad] = neglogLikBTDR_IncompObs_uneqvar_lambonly(pars,S,Ai,Ri,zzi,r,ni,g)
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
T = length(Ri(:,:,1))/P;
n = length(ni);
lambi = pars(1:n);

%% negative log likelihood only


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
lambAiIS = permute(lambAiIS,[1 3 2]);

try
    % Ci := lambi*S*kron(I,Ai)*S' + eye(sum(r))
    Ci = mmx_mkl_single('mult',repmat(S,1,1,n),lambAiIS);
    [x1, x2] = ndgrid(1:sum(r),1:n);
    ind = sub2ind(size(Ci), x1(:), x1(:), x2(:));
    Ci(ind) =  Ci(ind) + ones(size(Ci(ind)));
    
    % invCiSRi := Ci\(S*Ri)
    invCiSRi = mmx_mkl_single('backslash',Ci,repmat(S,1,1,n));
    invCiSRi = mmx_mkl_single('mult',invCiSRi,Ri);
    
    % Terms of log-likelihood
    Qtermi = reshape(permute(invCiSRi,[3,1,2]),n,sum(r)*T*P)*sparse(vec(S));
    cholCi = mmx_mkl_single('chol',Ci,[]);
    logdetterm = 2*sum(log(cholCi(ind)));
catch
    % Ci := lambi*S*kron(I,Ai)*S' + eye(sum(r))
    Ci = slowMult(repmat(S,1,1,n),lambAiIS);
    [x1, x2] = ndgrid(1:sum(r),1:n);
    ind = sub2ind(size(Ci), x1(:), x1(:), x2(:));
    Ci(ind) =  Ci(ind) + ones(size(Ci(ind)));
    
    % invCiSRi := Ci\(S*Ri)
    invCiSRi = slowBackslash(Ci,repmat(S,1,1,n));
    invCiSRi = slowMult(invCiSRi,Ri);
    
    % Terms of log-likelihood
    Qtermi = reshape(permute(invCiSRi,[3,1,2]),n,sum(r)*T*P)*sparse(vec(S));
    cholCi = slowChol(Ci);
    logdetterm = 2*sum(log(cholCi(ind)));
end

% noncumulative operations
Qterm = (lambi.^2)'*Qtermi;
regterm = g*pars(n+1:end)'*pars(n+1:end);
negloglik = .5*( -T*ni*log(lambi) + logdetterm + ...
    zzi'*lambi - Qterm + regterm);
%% negative log likelihood and gradient
if nargout==2
    try
        lambinvCiSAiI = mmx_mkl_single('backslash',Ci,permute(lambAiIS,[2 1 3]));
    catch
         lambinvCiSAiI = slowBackslash(Ci,permute(lambAiIS,[2 1 3]));
    end
    % dLdlambdai
    invCiSAiI = bsxfun(@times,lambinvCiSAiI,permute(repmat(1./lambi,1,P*T),[3 2 1]));
    %     SinvCiSRi = mmx_mkl_single('mult',repmat(S,1,1,n),invCiSRi,'tn');
    SinvCiSRi = reshape(S'*reshape(invCiSRi,sum(r),n*P*T),P*T,P*T,n);
    F = bsxfun(@times,SinvCiSRi,permute(repmat(lambi.^2,1,P*T),[3 2 1]));
    [x1, x2] = ndgrid(1:T*P,1:n);ind = sub2ind(size(F), x1(:), x1(:), x2(:));
    F(ind) =  ones(size(ind)) + F(ind);% Add identity
    try
        F =  mmx_mkl_single('mult',invCiSAiI,F);
    catch
        F =  slowMult(invCiSAiI,F);
    end
    
    dQdlambi = reshape(permute(F,[3,2,1]),n,sum(r)*T*P)*vec(S');
    dLdlambi = .5*(-T*ni'./lambi + zzi - 2*lambi.*Qtermi ...
        + dQdlambi);
    
    grad = dLdlambi;
end