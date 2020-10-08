function [negloglik,grad,Hess] = neglogLikBTDR_IncompObs_uneqvar_Sonly(pars,lambi,Ai,Ri,zzi,r,ni,g)
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
P = length(r);  % number of regressors
T = length(Ri(:,1))/P; % number of time bins
TP = T*P; % number of regressors times number of time bins
n = length(ni); % number neurons
rtot = sum(r); % sum of ranks
% Make block diagonal S matrix
S = mat2cell(reshape(pars,T,rtot)',r,T); % extract blocks in cell array
S = blkdiag(S{:});  % make block diagonal matrix

%% Compute negative log likelihood

% --------------------
% lambAiIS := lambi*kron(I,Ai)*S'
lambAi = reshape(bsxfun(@times,Ai,permute(lambi,[3 2 1])),P,P*n); % [4 x 4*762]
S2 = reshape(S,[],P);  % [15*22 x 4]
lambAiIS = permute(reshape(S2*lambAi,rtot,TP,n),[2 1 3]);

% --------------------
% Ci := lambi*S*kron(I,Ai)*S' + eye(rtot)
Ci = reshape(S*reshape(lambAiIS,TP,rtot*n),rtot,rtot,n);
Ci = bsxfun(@plus,Ci,eye(rtot));

% --------------------
% invCiSRi := Ci\(S*Ri)
SRi = reshape(S*reshape(Ri,TP,[]),rtot,TP,n); % multiply S by Ri
try
    invCiSRi = mmx_mkl_single('backslash',Ci,SRi); % do backslash
catch
    invCiSRi = slowBackslash(Ci,SRi); % do backslash
end

% Terms of log-likelihood
% Quadratic term
Qtermi = reshape(permute(invCiSRi,[3,1,2]),n,rtot*TP)*sparse(vec(S));
Qterm = (lambi.^2)'*Qtermi;

% log-det term
try
    cholCi = mmx_mkl_single('chol',Ci,[]);
catch
    cholCi = slowChol(Ci);
end
ind = repmat(logical(speye(rtot)),1,n); % indices for identity for each Ci-size matrix
logdetterm = 2*sum(log(cholCi(ind)));

% noncumulative operations
regterm = g*pars(n+1:end)'*pars(n+1:end);
negloglik = .5*( -T*ni*log(lambi) + logdetterm + ...
    zzi'*lambi - Qterm + regterm);

%% negative log likelihood and gradient
if nargout>=2
    try
        lambinvCiSAiI = mmx_mkl_single('backslash',Ci,permute(lambAiIS,[2 1 3]));
    catch
        lambinvCiSAiI = slowBackslash(Ci,permute(lambAiIS,[2 1 3]));
    end
    % dLdS
    lambSinvCiSAiI = reshape(S'*reshape(lambinvCiSAiI,rtot,TP*n),TP,TP,n); % removed mmx from here
    M = bsxfun(@plus,-lambSinvCiSAiI,eye(TP)); % Add identity
    M = bsxfun(@times,M,permute(lambi.^2, [3 2 1]));
    
    try
        dLdS = lambinvCiSAiI - mmx_mkl_single('mult',invCiSRi,M);
    catch
        dLdS = lambinvCiSAiI - slowMult(invCiSRi,M);
    end
    dLdS = squeeze(sum(dLdS,3)) + g*S;
    
    %Only keep the parts of dLdS we need
    dLds = [];
    for p = 1:P
        colind = 1+T*(p-1):p*T;
        if p==1
            rowind = 1:r(1);
        else
            rowind = sum(r(1:p-1))+1:sum(r(1:p));
        end
        dLdSp = dLdS(rowind,colind);
        dLds = [dLds; vec(dLdSp')];
    end
    
    grad = dLds;
end
%% negative log likelihood, gradient, and Hessian

if nargout>2    
    % Calculate 2nd derivatives over all elements of S
    invCiS = mmx_mkl_single('backslash',Ci,repmat(S,1,1,n));
    H1 = zeros(T*sum(r));H2 = zeros(T*sum(r));
    Md1L0 = bsxfun(@plus, -permute(lambSinvCiSAiI,[2 1 3]),eye(TP));
    Md1L0 = reshape(Md1L0,[],P,n);
    for i = 1:n
        % Some of the logdet terms
        invCi = inv(squeeze(Ci(:,:,i)));
        Md1L = lambi(i)*reshape(Md1L0(:,:,i)*Ai(:,:,i),[],TP)';
        
        % Q and logdet terms
        lambiAi_ITSinvCiSRi = lambAiIS(:,:,i)*invCiSRi(:,:,i);
        lambAi_ITSinvCiSRiSinvCi = lambiAi_ITSinvCiSRi*invCiS(:,:,i)';
        Md2_1_IIL = -lambi(i)^2*invCiSRi(:,:,i) + lambinvCiSAiI(:,:,i);
        Md2_1_II_IIIL = Md2_1_IIL + lambi(i)^2*lambAi_ITSinvCiSRiSinvCi';
        M_II_IIIR = lambi(i)^2*(invCiSRi(:,:,i)' - ...
            lambAi_ITSinvCiSRiSinvCi);
        M_I_IVL1 = lambi(i)^2*Md1L;
        M_I_IVL2 = lambiAi_ITSinvCiSRi - ...
            lambAiIS(:,:,i)*lambAi_ITSinvCiSRiSinvCi';
        M1_I_IIIL = Ri(:,:,i)-lambiAi_ITSinvCiSRi;
        M_d1_I_IVL = (Md1L+lambi(i)*lambi(i)*(M_I_IVL2 - M1_I_IIIL));

        H1 = H1 +...
            quickkron_rxr_PTxPT(M_I_IVL1,invCiSRi(:,:,i)*invCiS(:,:,i)',r,P,T) + ...
            quickkron_rxr_PTxPT(M_d1_I_IVL,invCi,r,P,T);
        H2 = H2 + quickkron_PTxr_rxPT(lambinvCiSAiI(:,:,i),M_II_IIIR,r,P,T) - ...
            quickkron_PTxr_rxPT(Md2_1_II_IIIL,lambinvCiSAiI(:,:,i)',r,P,T);
    end
    
    H = H1 + H2;
    Hess = .5*(H + H');
end