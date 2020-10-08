function bhati = MMLE_b(Ci,S,lambi,ni,xbari,Ybar,Xzetai,r)

P = length(r);
[T,n] = size(Ybar);
bhati = zeros(T,n);
rtot = sum(r);

for ii = 1:n
%     XiS = kronmult({eye(T),xbari(ii,:)},S');

XiS=zeros(T,rtot);
for p = 1:P
    colind = 1+T*(p-1):p*T;
    if p==1
        rowind = 1:r(1);
    else
        rowind = sum(r(1:p-1))+1:sum(r(1:p));
    end
    Sp = S(rowind,colind);
    XiS(:,rowind) = xbari(ii,p)*Sp';  
end
    CIXS = squeeze(Ci(:,:,ii))\XiS';
    A = eye(T)-lambi(ii)*ni(ii)*XiS*CIXS;
    ybarhat = lambi(ii)*CIXS'*S*Xzetai(:,ii);
    bhati(:,ii) = A\(Ybar(:,ii)-ybarhat);
end
