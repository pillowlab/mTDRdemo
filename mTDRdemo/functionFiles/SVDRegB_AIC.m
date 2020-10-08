function AIC = SVDRegB_AIC(Yn,allstim,pars,r)

n = length(Yn);
T = size(Yn{1},1);
P = size(allstim{1},2);
Bhat = reshape(pars,n,[]);
negloglik = 0;
for ii = 1:n
    Bi = Bhat(ii,:)';
    ri = vec(Yn{ii}')- kronmult({speye(T),allstim{ii}},Bi);
    lambhat(ii) = size(allstim{ii},2)*T/(real(ri)'*real(ri));
    negloglik = negloglik  + real(ri)'*real(ri)*lambhat(ii) + numel(Yn{ii})*log(lambhat(ii));
end

K = (n*P + T*P - sum(r))*sum(r);% Number of free parameters
AIC = negloglik + 2*K;% negloglik is already multiplied by 2

