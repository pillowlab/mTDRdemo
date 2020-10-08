function s = keepActive_S(S,r)

P = length(r);
TP = size(S,2);
T = TP/P;

s=[];
for p = 1:P
    colind = 1+T*(p-1):p*T;
    if p==1
        rowind = 1:r(1);
    else
        rowind = sum(r(1:p-1))+1:sum(r(1:p));
    end
    Sp = S(rowind,colind);
    s = [s; vec(Sp')];  
end
