function [Bhat, r,What,Shat,lambhat] = MakeBhat_data(histfile,Ai,Xzetai,r,b)

[TP,n] = size(Xzetai);

loadvars = load(histfile);

if iscell(loadvars.parhist)
if ~isempty(b)
    lambhat = loadvars.parhist{b}(1:n);
    shat = loadvars.parhist{b}(n+1:end);
else
    lambhat = loadvars.parhist{end}(1:n);
    shat = loadvars.parhist{end}(n+1:end);
end
else
    lambhat = loadvars.parhist(1:n);
    shat = loadvars.parhist(n+1:end);
end
if isempty(r)
    r = loadvars.rhist(end,:);
end

[P,~,~] = size(Ai);
T = TP/P;

endind = 0;Shat = cell(P,1);
for p = 1:P
    startind = endind +1;
    endind = startind + T*r(p) - 1;
    Shat{p} = reshape(shat(startind:endind),T,r(p))';
end
Shatblock = blkdiag(Shat{:});
Wt = EBpost_W_uneqvar(Shatblock,lambhat,Ai,Xzetai,sum(r));
Bhat = cell(P,1);What = cell(P,1);
for p = 1:P
    colind = (sum(r(1:p-1))+1):sum(r(1:p));
    What{p} = Wt(colind,:)';
    Bhat{p} = What{p}*Shat{p};
    
end
