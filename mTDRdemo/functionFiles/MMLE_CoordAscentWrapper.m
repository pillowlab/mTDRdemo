function pars = MMLE_CoordAscentWrapper(EMparsfun,MMLEParfun,r,n,T)
rtot = sum(r);
EMpars = EMparsfun(r);
lamb0 = EMpars(1:n);
s0 = EMpars(n+1:n+rtot*T);
bhat0 = reshape(EMpars(n+rtot*T+1:end),T,n);
[lambhat, shat,~,~,bhat] = MMLEParfun(lamb0,s0,bhat0,r);
pars = [lambhat;shat;vec(bhat)];
