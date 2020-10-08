function [Ri,zzi,Xzetai] = ECMEsuffstat(zetai,Xi,b)
[T,n] = size(b);
P = size(Xi{1},2);
TP = T*P;
Xzetai = zeros(TP,n);
Ri = zeros(P*T,P*T,n);
zzi = zeros(n,1);

for ii = 1:n
    zi = vec(bsxfun(@minus,zetai{ii},b(:,ii)));
    
    zi = reshape(zi,T,[]);
    Xzetai(:,ii) = vec(zi*Xi{ii});
    Ri(:,:,ii)  = Xzetai(:,ii)*Xzetai(:,ii)';
    zzi(ii) = vec(zi)'*vec(zi);
    
%     Xzetai(:,ii) = kronmult({speye(T),Xi{ii}'},zi);
%     zzi(ii) = zi'*zi;
end
