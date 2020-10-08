%% TestmmxTimes
clear all;close all

% Set reasonable dimensions
n = 762;
P = 6;
T = 15;
TP = T*P;
rtot = TP*2;
r = 2*ones(P,1);
%% Compare backslash
maxrtot = 200;
rtots = 10:5:maxrtot;
% for rind = 1:length(rtots)
%     S = randn(rtot,TP,n);
%     X = randn(TP,rtot,n);
%     for ii = 1:n
%         A(:,:,ii) = X(:,:,ii)'*X(:,:,ii) +TP*TP*eye(rtot);
%     end
%     
%     % compare backslash
%     tic
%     for ii = 1:n
%         invAS1(:,:,ii)= A(:,:,ii)\S(:,:,ii);
%     end
%     Tloop(rind) = toc;
%     
%     tic
%     invAS2 = mmx_mkl_single('backslash',A,S);
%     Tmmx(rind) = toc;
% end
% 
% figure;plot(rtots,Tloop,rtots,Tmmx)
% figure;plot(Tloop./Tmmx)

%% Compare multiply

    S = randn(rtot,TP,n);
    X = randn(TP,rtot,n);
    for ii = 1:n
        A(:,:,ii) = X(:,:,ii)'*X(:,:,ii) +TP*TP*eye(rtot);
    end
    % compare backslash
    tic
    for ii = 1:n
       AS1(:,:,ii)= A(:,:,ii)*S(:,:,ii);
    end
    toc
    
    tic 
    rA = 
    rS
    AS2 = reshape(
    toc
    
    tic
    AS3 = mmx_mkl_single('mult',A,S);
    toc
