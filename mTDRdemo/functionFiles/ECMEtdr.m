function [parhat,Q,parerr,nll] = ECMEtdr(stopmode,stopcrit,pars0,Ai,Xi,r,ni,zetai,xbari,Ybar,Xzetai0)
%  [parhat,Q,parerr] = EMtdr_Slamb(stopmode,stopcrit,pars0,Ai,Ri,zzi,r,ni)
% Function to perform expectation maximization to estimate time-varying
% basis functions and independant noise variance for the MBTDR model.
% INPUTS:stopmode,stopcrit,pars0,Ai,Ri,zzi,r,ni
%
% OUTPUTS:
% parhat -
% Q
% parerr
maxsteps = 100;


% Unpack parameters
n = length(ni);
T = size(zetai{1},1);
P = length(r);
TP = T*P;
rtot = sum(r); % sum of ranks
lambiold = pars0(1:n);
olds = pars0(n+1:n+rtot*T);
bold = reshape(pars0(n+rtot*T+1:end),T,n);

% For first step, just set new and old pars equal to each other
lambinew = lambiold;
news= olds;
bnew = bold;
[Ri,zzi,Xzetai] = ECMEsuffstat(zetai,Xi,bold);

if strcmp(stopmode,'steps')
    stopcrit = stopcrit+1;
end

% Q = zeros(Nsteps+1,1);
pars0q = [lambiold;olds];
Q(1) = Q_TDR(pars0q,pars0q,Ai,Ri,zzi,r,ni,0);
nll(1) = neglogLikBTDR_IncompObs_uneqvar_S_nllonly(pars0,Ai,Xzetai,zzi,r,ni,0);

k = 2;stopvar = true;
while stopvar
    bold = bnew;
    bold0 = bnew;
    lambiold0 = lambinew;
    olds0 = news;
    % Reset sufficient stats using current estimate of b
    [Ri,zzi,Xzetai] = ECMEsuffstat(zetai,Xi,bold);
    
    % ------------------ M-step for lambda------------------
    Snew = mat2cell(reshape(news,T,rtot)',r,T); % extract blocks in cell array
    Snew = blkdiag(Snew{:});  % make block diagonal matrix
    Sold = Snew;
    
    % Old feature covariance
    % --------------------
    % Ci := lambi*S*kron(Ai,I)*S' + eye(rtot)
    S2 = reshape(Sold,[],P);  % [15*22 x 4]
    AiISold = permute(reshape(S2*reshape(Ai,P,P*n),rtot,TP,n),[2 1 3]);
    SAiISold = reshape(Sold*reshape(AiISold,TP,rtot*n),rtot,rtot,n);
    lambiSAiSold = bsxfun(@times,SAiISold,permute(lambiold,[3 2 1]));
    Ciold = bsxfun(@plus,lambiSAiSold,eye(rtot));
    
    % --------------------
    % BiSRi := Ciold\(S*Ri)
    SoldRi = reshape(Sold*reshape(Ri,TP,[]),rtot,TP,n); % multiply S by Ri
    try
        BiSoldRi = mmx_mkl_single('backslash',Ciold,SoldRi); % do backslash
    catch
         BiSoldRi = slowBackslash(Ciold,SoldRi); % do backslash
    end
    g1i = 2*lambiold.*(reshape(permute(BiSoldRi,[3,1,2]),n,rtot*TP)*vec(Snew));
    rAi = reshape(Ai,P,P*n);
    AiISnew= permute(reshape(S2*rAi,rtot,TP,n),[2 1 3]);
    SnewAiSnew = reshape(Snew*reshape(AiISnew,TP,rtot*n),rtot,rtot,n);
    try
        BiSAS = mmx_mkl_single('backslash',Ciold,SnewAiSnew); % do backslash
    catch
        BiSAS = slowBackslash(Ciold,SnewAiSnew); % do backslash
    end
    g2i = zeros(n,1);
    for ii = 1:n;  g2i(ii) = trace(BiSAS(:,:,ii));  end
    try
        RSBSASB = mmx_mkl_single('mult',permute(BiSoldRi,[2,1,3]),permute(BiSAS,[2,1,3]));
    catch
        RSBSASB = slowMult(permute(BiSoldRi,[2,1,3]),permute(BiSAS,[2,1,3]));
    end
    g3i = reshape(permute(RSBSASB,[3,2,1]),n,rtot*TP)*sparse(vec(Sold));
    g3i = g3i.*lambiold.^2;
    lambinew = T*ni'./(zzi - g1i + g2i + g3i);
    lambiold = lambinew;
    % ------------------------------------------------------
    
    % ------------------M-step for S------------------
    
    % Remake old feature covariance with new lambda
    % --------------------
    % Ci := lambi*S*kron(Ai,I)*S' + eye(rtot)
    lambiSAiSold = bsxfun(@times,SAiISold,permute(lambiold,[3 2 1]));
    Ciold = bsxfun(@plus,lambiSAiSold,eye(rtot));
    
    lambnewilamboldBiSoldRi = bsxfun(@times,BiSoldRi,permute(lambinew.*lambiold,[3 2 1]));
    M0 = sum(lambnewilamboldBiSoldRi,3);% LHS matrix
    m0 =  keepActive_S(M0,r);
    % Sum over terms
    SRSB_old = reshape(Sold*reshape(permute(BiSoldRi,[2,1,3]),TP,[]),rtot,rtot,n);
    lamb2oldSRSB_old = bsxfun(@times,SRSB_old,permute(lambiold.^2,[3,2,1]));
    try
        Gi = mmx_mkl_single('backslash',Ciold,bsxfun(@plus,lamb2oldSRSB_old,eye(rtot)));
    catch
        Gi = slowBackslash(Ciold,bsxfun(@plus,lamb2oldSRSB_old,eye(rtot)));
    end
    SSnew = mat2cell(reshape(news,T,rtot)',r,T); % extract blocks in cell array
    SSnew = cat(1,SSnew{:});  % make block diagonal matrix
    Gammap = cell(P,1);
    for p = 1:P
        gammapq = cell(P,1);
        for q = 1:P
            lambda_ipq = bsxfun(@times,Ai(p,q,:),permute(lambinew,[3,2,1]));
            if q==1;
                Gind = 1:r(1);
            else
                Gind = (sum(r(1:q-1))+1):sum(r(1:q));
            end
            aGi = bsxfun(@times,Gi(:,Gind,:),lambda_ipq);
            gammapq{q} =sum(aGi,3);
        end
        Gammap{p} = cat(2,gammapq{:});
    end
    G = cat(1,Gammap{:});
    GG = zeros(rtot,rtot);
    for p = 1:P
        G = Gammap{p};
        if p == 1
            GG(1:r(1),:) = G(1:r(1),:);
            GG(:,1:r(1)) = G(1:r(1),:)';
        else
            startind = sum(r(1:p-1))+1;
            endind = sum(r(1:p));
            GG(startind:endind,startind:rtot) = G(startind:endind,startind:rtot);
            GG(startind:rtot,startind:endind) = G(startind:endind,startind:rtot)';
        end
    end
    news = GG\reshape(m0,T,rtot)';
    news = vec(news');

    
    % ------------------------------------------------------
    % ---------- MMLE of condition-independent term----------
    S = mat2cell(reshape(news,T,rtot)',r,T); % extract blocks in cell array
    S = blkdiag(S{:});  % make block diagonal matrix
    S2 = reshape(S,[],P);  % [15*22 x 4]
    AiISold = permute(reshape(S2*reshape(Ai,P,P*n),rtot,TP,n),[2 1 3]);
    SAiISold = reshape(Sold*reshape(AiISold,TP,rtot*n),rtot,rtot,n);
    lambiSAiSold = bsxfun(@times,SAiISold,permute(lambiold,[3 2 1]));
    Ci = bsxfun(@plus,lambiSAiSold,eye(rtot));
    bnew = MMLE_b(Ci,S,lambiold,ni,xbari,Ybar,Xzetai0,r);
    % ------------------------------------------------------
    
    % Convergence checks
    newpars = [lambinew;news;vec(bnew)];parhat = newpars;
    oldpars = [lambiold0;olds0;vec(bold0)];
    parerr(k-1) = max((newpars-oldpars).^2./oldpars.^2);
    
    % Evaluate Q
    if nargout>1
        Q(k) = Q_TDR(newpars(1:(n+rtot*T)),oldpars(1:(n+rtot*T)),Ai,Ri,zzi,r,ni,0);
        if nargout >3
            [~,zzi,Xzetai] = ECMEsuffstat(zetai,Xi,bnew);
            nll(k) = neglogLikBTDR_IncompObs_uneqvar_S_nllonly(newpars(1:(n+rtot*T)),Ai,Xzetai,zzi,r,ni,0);
        end
    end
    
    switch stopmode
        case 'steps'
            if k>=stopcrit; stopvar = false;end
        case 'converge'
            if parerr(k-1)<stopcrit||k>=maxsteps; stopvar = false; end
    end
    k = k + 1;
end

