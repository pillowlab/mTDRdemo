function [Wt,Ci] = EBpost_W_uneqvar(S,lambi,Ai,Xzetai,sumr)

[P,~,n] = size(Ai);
T = size(S,2)/P;

I_T = eye(T);I_r = speye(sumr);
Wt = [];    SXIZi = S*Xzetai;
Ci = zeros(sumr,sumr,n);
for i = 1:n

    AiIS = kronmult({I_T,squeeze(Ai(:,:,i))},S');
    Ci(:,:,i) = lambi(i)*S*AiIS + I_r;
    Wt = [Wt (Ci(:,:,i)\SXIZi(:,i))*lambi(i)];
end
