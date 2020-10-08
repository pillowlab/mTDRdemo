function [lambhat, shat,Shat,Shatblock,bhat] = Estpars_CoordAscent_lambi_S_b(lambhat0,shat0,bhat0,r,Ai,Xi,zetai,ni,xbari,Ybar,Xzetai0)

P = length(r);
rtot = sum(r);
[T,n] = size(Ybar);
TP = T*P;
maxsteps = 10;
stopvar = true;
stopcrit = 1e-4;
k=1;
while stopvar
    % Re-calculated sufficient stats, conditioned on b0
    [Ri,zzi,~] = ECMEsuffstat(zetai,Xi,bhat0);
    
    %% Estimate S conditioned on b0 and lambda
    options.display = 'none';
    loglikS_lamb = @(s)neglogLikBTDR_IncompObs_uneqvar_Sonly(s,lambhat0,Ai,Ri,zzi,r,ni,0);
    shat = minFunc(loglikS_lamb,shat0,options);
    Shat = mat2cell(reshape(shat,T,rtot)',r,T); % extract blocks in cell array
    Shatblock = blkdiag(Shat{:});  % make block diagonal matrix
    
    %% Estimate lambda conditioned on new S and b0
    Hesspat = sparse(n,n);
    Hesspat(logical(eye(size(Hesspat)))) = ones(numel(lambhat0),1);
    fminopts = optimset('gradobj','on','display','notify','Hessian','off',...
        'HessPattern',Hesspat,'algorithm','trust-region-reflective','maxfunevals',1000,'maxiter',1000);
    loglikfunLambda = @(lamb)neglogLikBTDR_IncompObs_uneqvar_lambonly(lamb,Shatblock,Ai,Ri,zzi,r,ni,0);
    lambhat = fminunc(loglikfunLambda,lambhat0,fminopts);
    
    %% Estimate b0 conditioned on new S and new lambda
    S2 = reshape(Shatblock,[],P);
    AiIS = permute(reshape(S2*reshape(Ai,P,P*n),rtot,TP,n),[2 1 3]);
    SAiIS = reshape(Shatblock*reshape(AiIS,TP,rtot*n),rtot,rtot,n);
    lambiSAiS = bsxfun(@times,SAiIS,permute(lambhat,[3 2 1]));
    Ci = bsxfun(@plus,lambiSAiS,eye(rtot));
    bhat = MMLE_b(Ci,Shatblock,lambhat,ni,xbari,Ybar,Xzetai0,r);
    
    % Evaluate convergence criteria
    newpars = [lambhat;shat;vec(bhat)];
    oldpars = [lambhat0;shat0;vec(bhat0)];
    parerr = max((newpars-oldpars).^2./oldpars.^2);
    if parerr<stopcrit||k>=maxsteps; stopvar = false; end
    lambhat0 = lambhat;
    shat0 = shat;
    bhat0 = bhat;
    k = k + 1;
end