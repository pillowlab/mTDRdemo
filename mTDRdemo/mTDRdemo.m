%% DemoCode
clear all;close all;
%% Set path
CrntDir = pwd;
addpath([CrntDir '/EstimatedPars'],...
    [CrntDir '/functionFiles'],...
    [CrntDir '/functionFiles/tools_kron'],...
    genpath([CrntDir '/minFunc_2012']))

%% Specify dimensions of data
n = 100; %number of neurons
T = 15; %number of time points
Nmax = 100; %maximum number of trials
P = 4; %number of covariates
Pb = P-1;
rmax = 6; % Maximum rank for demo

fprintf('True ranks')
rP = randi([1 rmax],[1 Pb])% Select dimensionality at random

%% Simulation Parameters

% Parameters generating smooth time components
len = 2*ones(P,1);%length scale
rho = 1*ones(P,1); %Variance
[Wtrue, Strue, BB] = SimWeights(n,T,P,[rP T],len,rho);

%Plot the basis functions generated
figure;
for p=1:P
    subplot(P,1,p)
    plot(cat(2,Strue{p}))
    xlabel('time')
    title(['Bases for task variable ',num2str(p)])
    box off
    xlim([1 T])
    
end
set(gcf,'color','w')
subplot(P,1,P)
title(['Bases for condition-independent component'])

figure;
for p=1:P
    subplot(P,1,p)
    bar(cat(2,Wtrue{p}))
    xlabel('Neuron index')
    title(['Weights for task variable ',num2str(p)])
    box off
    xlim([1 n])
end
set(gcf,'color','w')
subplot(P,1,P)
title(['Weights for condition-independent component'])

% Set noise variance of each neuron
mstdnse = 1/.8;% Mean standard deviation of the noise
d = exprnd(mstdnse,[n 1]);

%Define all levels of regressors
var_uniq{1} = -2:2;
var_uniq{2} = -2:2;
var_uniq{3} = [-1 1];
var_uniq{4} = 1;

% Generate samples
X = SimConditions(var_uniq,Nmax);% Task conditions
Y = SimPopData(X,BB,d, n,T,Nmax);% Neuronal responses

% Randomly drop neurons on different trials so that trial numbers differ
% across neurons
hk = zeros(Nmax,n);
Z = zeros(n,T,Nmax);
pdrop = 0.3;% Probability of a neuron being dropped from any given trial
for k = 1:Nmax
    hk(k,:) = binornd(1,1-pdrop,[1 n]);
    Z(:,:,k) = spdiags(hk(k,:)',0,n,n)*squeeze(Y(:,:,k));
end

figure;
for ii = 1:5
    subplot(5,1,ii)
    plot(squeeze(Z(ii,:,1:5)))
    ylabel(['neuron ' num2str(ii)])
    box off
end
subplot(511)
title('Sample neuronal responses')
legend('Trial 1','Trial 2','Trial 3','Trial 4','Trial 5')
set(gcf,'color','w')
%% Estimate

% Initialize estimation parameters
r0 = ones(P-1,1); % initial ranks
maxrank = min(n,T);% Specify maximum allowable rank
ridgeparam = 0;
opts.MaxIter = 500;
opts.Display = 'off';
g = 0;

% Make sufficient stats for SVD
[XX, XY,Yn,allstim,n,T] = MkSuffStats_BilinReg_Sims(Z,X,hk);

% Make sufficient stats for MMLE
Xb = X;
Xb(:,P) = [];% Remove condition-independent component from design matrix for MML estimation
[Ri,Ai,zzi,ni,Xi,Xzetai0,zetai0] = MkSuffStatsBTDR_IncompObs_uneqvar_S_fast(Xb,Z,hk);


% Regression function for estimation by SVD
SVDRegPars = @(r)SVDRegressB(XX,XY,[T n],r,ridgeparam,opts);

% AIC objective for estimation by SVD
SVD_AIC = @(pars,r)SVDRegB_AIC(Yn,allstim,pars,r);
histfileSVD  = 'EstimatedPars/RankEstDemo_SVD';

fprintf('Estimate rank with SVD parameter estimates\n')
[rEstSVD, rhist, funhist] = ...
    EstRankGreedily(SVD_AIC,SVDRegPars,[r0;maxrank],maxrank,[],histfileSVD);


% File for recording rank estimation history
histfileMMLE = 'EstimatedPars/RankEstDemoMMLE';

% Some extra sufficient stats
xbari = zeros(n,P-1);
Ybar = zeros(T,n);
for ii = 1:n
    xbari(ii,:) = mean(Xi{ii},1);
    ybar = squeeze(sum(Z(ii,:,:),3))/ni(ii);
    Ybar(:,ii) = ybar;
end
rest0 = rEstSVD(1:P-1); % Initialize with svd solution

% Functions that rank estimation will use
svdregress = @(r)SVDRegress_S_Vdata(XX,XY,Yn,allstim,T,n,r,ridgeparam,opts);
AICMMLE = ...
    @(pars,r)BTDR_AIC_S_lamb_b_wrapper(pars,Ai,Xi,zetai0,r,ni,g);
MMLEParfun =@(lamb0,s0,bhat0,r)Estpars_CoordAscent_lambi_S_b(lamb0,s0,bhat0,r,Ai,Xi,zetai0,ni,xbari,Ybar,Xzetai0);
EMregressfun = @(r,pars0)ECMEtdr('converge',1e-0,pars0,Ai,Xi,r,ni,zetai0,xbari,Ybar,Xzetai0);
EMparsfun = @(r)ECMEregress_wrapper(r,svdregress,EMregressfun,Ybar);
MMLE_EMregressfun = @(r)MMLE_CoordAscentWrapper(EMparsfun,MMLEParfun,r,n,T);

fprintf('Estimate by EM-MML\n')
[rEstMMLE_EM, rhist, funhist,parhist] = EstRankGreedily(AICMMLE,MMLE_EMregressfun,rest0,maxrank,[],histfileMMLE);

%% Plot parameter estimates for comparison with true values
load(histfileMMLE)
pars = parhist{end};
r = rEstMMLE_EM;
lambhat = pars(1:n);
rtot = sum(rP);
Bhat0 = reshape(pars(n+rtot*T+1:end),T,n);
[~,~,Xzetai] = ECMEsuffstat(zetai0,Xi,Bhat0);
[Bhat,r,~,~,lambhat] = MakeBhat_data(histfileMMLE,Ai,Xzetai0,r,[]);
Bhat{P} = Bhat0';

% Plot comparision of noise precision
maxx = max([lambhat d]);
minx = min([lambhat d]);
figure;
loglog(lambhat,d,'o','markersize',10)
hold on
plot([minx maxx],[minx maxx],'k')
hold off
title('Noise precision')
xlabel('Estimate')
ylabel('True')
% axis([minx maxx minx maxx])
box off
axis tight
axis equal
set(gcf,'color','w')


% Plot comparision of regression params
figure;
for p = 1:P
    subplot(1,P,p)
    Bptrue = Wtrue{p}*Strue{p}';
    plot(Bhat{p}(:),Bptrue(:),'.')
    minb = min([Bhat{p}(:);Bptrue(:)]);
    maxb = max([Bhat{p}(:);Bptrue(:)]);
    hold on
    plot([minb maxb],[minb maxb],'k')
    axis([minb maxb minb maxb])
    hold off
    title(['Task var ' num2str(p)])
    set(gca,'tickdir','out')
    box off
    if p==1
        xlabel('estimate')
    ylabel('true')
    end
    axis tight
    axis equal
end
subplot(1,P,P)
title('Task indep. component')
set(gcf,'color','w')