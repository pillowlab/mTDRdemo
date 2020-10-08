function [pars,b0]  = SVDRegress_S_Vdata(XX,XY,Yn,allstim,T,n,r,ridgeparam,opts)

[Bfullhat,Bthat,Bxhat] = SVDRegressB(XX,XY,[T,n],r,ridgeparam,opts);
shatsvd = [];B = [];

P=length(r);
for p = 1:P
    shatsvd = [shatsvd; vec(Bthat{p})];
    B = [B Bfullhat(:,:,p)'];
end
%% Estimate lambdas
lambhat = zeros(n,1);negloglik = 0;
for ii = 1:n
    Bi = vec(B(ii,:));
    ri = vec(Yn{ii}')- kronmult({speye(T),allstim{ii}},Bi);
    lambhat(ii) = size(allstim{ii},1)*T/(real(ri)'*real(ri));
end
pars = [lambhat;shatsvd];
b0 = Bfullhat(:,:,P);