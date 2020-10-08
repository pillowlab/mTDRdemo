function AIC = BTDR_AIC_S_lamb_b_wrapper(pars,Ai,Xi,zetai,r,ni,g)
n = length(ni);
T = size(zetai{1},1);
rtot = sum(r);
newpars = pars(1:n+rtot*T);
b0 = reshape(pars(n+rtot*T+1:end),T,n);
[~,zzi,Xzetai] = ECMEsuffstat(zetai,Xi,b0);

negloglik = neglogLikBTDR_IncompObs_uneqvar_S_nllonly(newpars(1:(n+rtot*T)),Ai,Xzetai,zzi,r,ni,g);
AIC = 2*negloglik + 2*numel(pars);
% AIC = 2*negloglik + 2*(numel(pars)-length(ni));