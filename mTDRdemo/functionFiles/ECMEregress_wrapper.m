function pars = ECMEregress_wrapper(r,initRegressFun,EMregressfun,Ybar)

[T,n] = size(Ybar);
pars0 = initRegressFun([r min(n,T)]);
pars = EMregressfun(r,[pars0(1:(n+sum(r)*T));vec(Ybar)]);
