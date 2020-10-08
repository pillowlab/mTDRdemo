function [Q, dQ] = Q_TDR(par_new,par_old,Ai,Ri,zzi,r,ni,alpha)

% Unpack parameters
P = length(r);  % number of regressors
T = length(Ri(:,1))/P; % number of time bins
TP = T*P;
rtot = sum(r); % sum of ranks
n = length(ni); % number neurons
lambinew = par_new(1:n);
snew = par_new(n+1:end);
lambiold = par_old(1:n);
sold = par_old(n+1:end);

% Make block diagonal S matrix
Snew = mat2cell(reshape(snew,T,rtot)',r,T); % extract blocks in cell array
Snew = blkdiag(Snew{:});  % make block diagonal matrix
Sold = mat2cell(reshape(sold,T,rtot)',r,T); % extract blocks in cell array
Sold = blkdiag(Sold{:});  % make block diagonal matrix

% Make new feature covariance
% --------------------
% lambAiIS := lambi*kron(I,Ai)*S'
lambnewAi = reshape(bsxfun(@times,Ai,permute(lambinew,[3 2 1])),P,P*n); % [4 x 4*762]
S2 = reshape(Snew,[],P);  % [15*22 x 4]
lambAiISnew = permute(reshape(S2*lambnewAi,rtot,TP,n),[2 1 3]);
% --------------------
% Ci := lambi*S*kron(I,Ai)*S' + eye(rtot)
lambiSAiSnew = reshape(Snew*reshape(lambAiISnew,TP,rtot*n),rtot,rtot,n);
Cinew = bsxfun(@plus,lambiSAiSnew,eye(rtot));


% Make old feature covariance
% --------------------
% Ci := lambi*S*kron(I,Ai)*S' + eye(rtot)
lamboldAi = reshape(bsxfun(@times,Ai,permute(lambiold,[3 2 1])),P,P*n); % [4 x 4*762]
S2 = reshape(Sold,[],P);  % [15*22 x 4]
lambAiISold = permute(reshape(S2*lamboldAi,rtot,TP,n),[2 1 3]);
lambiSAiSold = reshape(Sold*reshape(lambAiISold,TP,rtot*n),rtot,rtot,n);
Ciold = bsxfun(@plus,lambiSAiSold,eye(rtot));

% --------------------
% BiSRi := Ciold\(S*Ri)
SoldRi = reshape(Sold*reshape(Ri,TP,[]),rtot,TP,n); % multiply S by Ri
try
    error('mark here')
    BiSoldRi = mmx_mkl_single('backslash',Ciold,SoldRi); % do backslash
    BiCi = mmx_mkl_single('backslash',Ciold,Cinew); % do backslash
    RSBCB = mmx_mkl_single('mult',permute(BiSoldRi,[2,1,3]),permute(BiCi,[2,1,3]));
catch
    BiSoldRi = slowBackslash(Ciold,SoldRi); % do backslash
    BiCi = slowBackslash(Ciold,Cinew); % do backslash
    RSBCB = slowMult(permute(BiSoldRi,[2,1,3]),permute(BiCi,[2,1,3]));
end

Quadi = reshape(permute(RSBCB,[3,1,2]),n,rtot*TP)*sparse(vec(Sold'));

Quad1 = zzi'*lambinew;
Quad2 = 2*(lambinew.*lambiold)'*reshape(permute(BiSoldRi,[3,1,2]),n,rtot*TP)*sparse(vec(Snew));
Quad3 = (lambiold.^2)'*Quadi;
Quad4 = 0;
for ii = 1:n
    Quad4 = Quad4 + trace(BiCi(:,:,ii));
end

Q = T*ni*log(lambinew) - Quad1 + Quad2 - Quad3 - Quad4 - alpha*(snew-sold)'*(snew-sold);
%% Gradient
if nargout==2
    
    % --------------------
    % dQdlambdainew
    g1i = 2*lambiold.*(reshape(permute(BiSoldRi,[3,1,2]),n,rtot*TP)*vec(Snew));
    try 
        BiSASlambi = mmx_mkl_single('backslash',Ciold,lambiSAiSnew); % do backslash
    catch
         BiSASlambi = slowBackslash(Ciold,lambiSAiSnew); % do backslash
   end
    g2i = zeros(n,1);
    for ii = 1:n
        g2i(ii) = trace(BiSASlambi(:,:,ii))./lambinew(ii);
    end
    try
        RSBSASBlambi = mmx_mkl_single('mult',permute(BiSoldRi,[2,1,3]),permute(BiSASlambi,[2,1,3]));
    catch
        RSBSASBlambi = slowMult(permute(BiSoldRi,[2,1,3]),permute(BiSASlambi,[2,1,3]));
    end
    g3i = reshape(permute(RSBSASBlambi,[3,1,2]),n,rtot*TP)*vec(Sold');
    g3i = g3i./lambinew.*lambiold.*lambiold;
    dQdlamb = T*ni'./lambinew-zzi + g1i - g2i - g3i;
    % --------------------
    
    
    % ------Test kronecker trick
    %     lambnewilamboldBiSoldRi = bsxfun(@times,BiSoldRi,permute(lambinew.*lambiold,[3 2 1]));
    %     M = sum(lambnewilamboldBiSoldRi,3);
    %     M0 = sum(M,3) + alpha*Sold;% LHS matrix
    %     SRSB_old = reshape(Sold*reshape(permute(BiSoldRi,[2,1,3]),TP,[]),rtot,rtot,n);
    %     lamb2oldSRSB_old = bsxfun(@times,SRSB_old,permute(lambiold.^2,[3,2,1]));
    %     Gi = mmx_mkl_single('backslash',Ciold,bsxfun(@plus,lamb2oldSRSB_old,eye(rtot)));
    %
    %     %  Sum over terms
    %     lambiGi = bsxfun(@times,Gi,permute(lambinew,[3 2 1]));
    %     G = zeros(rtot*TP,rtot*TP);
    %     for ii = 1:n
    %         Aprimei = kron(Ai(:,:,ii),eye(T));
    %         G = G + kron(Aprimei,lambiGi(:,:,ii));
    %     end
    %     G = G + alpha*eye(rtot*TP);
    %     [M0,G] = dQdSparts(par_new,par_old,Ai,Ri,zzi,r,ni,alpha);
    %     dQdS = vec(M0) - G*vec(Snew);
    %     dQdS = reshape(dQdS,rtot,TP);
    % --------------------
    %         % --------------------
    %         % dQdSnew (sort of naive)
    %         lambnewilamboldBiSoldRi = bsxfun(@times,BiSoldRi,permute(lambinew.*lambiold,[3 2 1]));
    %         lambBSnewA = mmx_mkl_single('backslash',Ciold,permute(lambAiISnew,[2 1 3]));
    %         RSoldBSnewAilambinew =  mmx_mkl_single('mult',permute(BiSoldRi,[2,1,3]),permute(lambAiISnew,[2,1,3]));
    %         SoldRSoldBSnewAilambinew = reshape(Sold*reshape(RSoldBSnewAilambinew,TP,[]),rtot,TP,n);
    %         BiSoldRSoldBSnewAilambinew =  mmx_mkl_single('backslash',Ciold,SoldRSoldBSnewAilambinew);
    %         BiSoldRSoldBSnewAilambinewlamb2old = bsxfun(@times,BiSoldRSoldBSnewAilambinew,permute(lambiold.^2,[3 2 1]));
    %         dQdS = sum(lambnewilamboldBiSoldRi,3) - sum(lambBSnewA,3) - sum(BiSoldRSoldBSnewAilambinewlamb2old,3);
    %         % --------------------
    
    % --------------------
    % dQdSnew (less naive)
    SRSB_old = reshape(Sold*reshape(permute(BiSoldRi,[2,1,3]),TP,[]),rtot,rtot,n);
    lamb2oldSRSB_old = bsxfun(@times,SRSB_old,permute(lambiold.^2,[3,2,1]));
    try
        Gi = mmx_mkl_single('backslash',Ciold,bsxfun(@plus,lamb2oldSRSB_old,eye(rtot)));
    catch
         Gi = slowBackslash(Ciold,bsxfun(@plus,lamb2oldSRSB_old,eye(rtot)));
   end
    lambnewilamboldBiSoldRi = bsxfun(@times,BiSoldRi,permute(lambinew.*lambiold,[3 2 1]));
    SSnew = mat2cell(reshape(snew,T,rtot)',r,T); % extract blocks in cell array
    SSnew = cat(1,SSnew{:});  % make block diagonal matrix
    Gammap = cell(P,1);
    for p = 1:P
        gammapq = cell(P,1);
        for q = 1:P
            lambda_ipq = bsxfun(@times,Ai(p,q,:),permute(lambinew,[3,2,1]));
            if q==1;
                Gind = 1:r(1);
            else
                Gind = (sum(r(1:q-1))+1):sum(r(1:q));
            end
            aGi = bsxfun(@times,Gi(:,Gind,:),lambda_ipq);
            gammapq{q} =sum(aGi,3);
        end
        Gammap{p} = cat(2,gammapq{:});
    end
    tempmat = reshape(cat(1,Gammap{:})*SSnew,rtot,P,T);
    lambGiSAi = reshape(permute(tempmat,[1,3,2]),rtot,TP);
    dQdS = sum(lambnewilamboldBiSoldRi,3) - lambGiSAi - alpha*(Snew-Sold);
    
    %Only keep the parts of dQdS we need
    dQds = keepActive_S(dQdS,r);
    
    dQ = [dQdlamb;2*dQds];
end
