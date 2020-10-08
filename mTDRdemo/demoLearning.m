%% demoLearning
clear all;close all
% This demo code will demonstrate the pipeline to go from data array to
% learned parameters when the subspace dimensionalities have been
% specified.
%% Set Path
CrntDir = pwd;
addpath([CrntDir '/EstimatedPars'],...
    [CrntDir '/functionFiles'],...
    [CrntDir '/functionFiles/tools_kron'],...
    genpath([CrntDir '/minFunc_2012']))
%% Specify dimensions of data
n       = 100; %number of neurons
T       = 15; %number of time points
Nmax    = 100; %maximum number of trials
P       = 4; % number of covariates, including condition-independent component
Pb      = P-1;%number of covariates, not including condition-independent component
rmax    = 6; % Maximum rank for demo
rP      = randi([1 rmax],[1 Pb]);% Select dimensionality at random
rtot    = sum(rP);

fprintf('True dimensionality:\n')
fprintf(['Rank of task variable 1 = ' num2str(rP(1)) '\n'])
fprintf(['Rank of task variable 2 = ' num2str(rP(2)) '\n'])
fprintf(['Rank of task variable 3 = ' num2str(rP(3)) '\n'])



%% Simulation Parameters

% Parameters specifying smoothnes of time components
len = 2*ones(P,1);%length scale
rho = 1*ones(P,1); %Variance

% Generate components
[Wtrue, Strue, BB] = SimWeights(n,T,P,[rP T],len,rho);

%Plot the basis functions generated
figure(1);
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

figure(2);
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
d       = exprnd(mstdnse,[n 1]); % set noise precision paramters

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
pdrop = 0.3;% Probability of a neuron being dropped from any given trial
hk = zeros(Nmax,n);  Z = zeros(n,T,Nmax);
for k = 1:Nmax
    hk(k,:)     = binornd(1,1-pdrop,[1 n]);
    Z(:,:,k)    = spdiags(hk(k,:)',0,n,n)*squeeze(Y(:,:,k));
end

figure(3);
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




%% Parameter learning
%%
%%

%% Specify optimization parameters
r0              = rP; % specify ranks.  Here we just set them to the true ranks
ridgeparam      = 0;% ridge parameter for ridge regression with low-rank constraint by SVD
opts.MaxIter    = 500;
opts.Display    = 'off';
g               = 0;% ridge regression parameter for MML

%% Step 1: calculate sufficient statistics

% Start with:
% Data array Z (n x T x K)  where 
% n = number of neurons
% T = number of time points
% K = number of trials
% Design matrix X (K x P), where
% P = number of task variables
% Observation matrix hk (K x n) that describes which neurons were
% observed on which trials.


 %Sufficient stats for initial estimation by SVD
 % Estimation by SVD treats condition-independent component as a covariate.
 %  The consequence of this is that the design matrix has one column of all
 %  ones. i.e. X(i,:) = [var1 var2... varP 1]
[XX, XY,Yn,allstim,n,T] = MkSuffStats_BilinReg_Sims(Z,X,hk);


% Remove condition-independent component from design matrix for MML
% estimation since we assume full rank for this component for all models
Xb = X;     Xb(:,P) = [];


% Make sufficient stats for estimation by maximum marginal likelihood
ni = sum(hk,1); zi = cell(n,1); Xi = cell(n,1); Ai = zeros(Pb,Pb,n); 
Ri = zeros(Pb*T,Pb*T,n);  zzi = zeros(n,1);  Xzetai = zeros(T*Pb,n);
xbari = zeros(n,Pb); Ybar = zeros(T,n);
for ii = 1:n
    Xi{ii}          = Xb(hk(:,ii)==1,:);
    Ai(:,:,ii)      = Xi{ii}'*Xi{ii};
    zi{ii}          = squeeze(Z(ii,:,hk(:,ii)==1));
    zetai           = vec(zi{ii});
    Xzetai(:,ii)    = kronmult({eye(T),Xi{ii}'},zetai);
    Ri(:,:,ii)      = Xzetai(:,ii)*Xzetai(:,ii)';
    zzi(ii)         = zetai'*zetai;
    xbari(ii,:)     = mean(Xi{ii},1);
    ybar            = squeeze(sum(Z(ii,:,:),3))/ni(ii);
    Ybar(:,ii)      = ybar;
end

%% Step 2: initialize functions for optimization

% function for initialization by SVD
svdregress      = @(r)SVDRegress_S_Vdata(XX,XY,Yn,allstim,T,n,r,ridgeparam,opts);

% function for direct optimization of marginal likelihood
MMLEParfun      = @(lamb0,s0,bhat0,r)Estpars_CoordAscent_lambi_S_b(lamb0,s0,bhat0,r,Ai,Xi,zi,ni,xbari,Ybar,Xzetai);

% function for optimization by EM
EMregressfun    = @(r,pars0)ECMEtdr('converge',1e-0,pars0,Ai,Xi,r,ni,zi,xbari,Ybar,Xzetai);
EMparsfun       = @(r)ECMEregress_wrapper(r,svdregress,EMregressfun,Ybar);

%% Step 3: run optimization and parse parameter vector

% function implementing end-to-end optimization with SVD initialization
MMLE_EMregressfun   = @(r)MMLE_CoordAscentWrapper(EMparsfun,MMLEParfun,r,n,T);
parhist            = MMLE_EMregressfun(r0);
histfileMMLE = 'EstimatedPars/LearnDemoMMLE';
save(histfileMMLE,'parhist')

lambhat = parhist(1:n);% estimated noise precisions
Bhat0   = reshape(parhist(n+rtot*T+1:end),T,n);% estimated condition-independent components
[~,~,Xzetai_opt] = ECMEsuffstat(zi,Xi,Bhat0);% recalculated a sufficient stat
[Bhat,~,~,~,~] = MakeBhat_data(histfileMMLE,Ai,Xzetai_opt,rP,[]);% estimated low-rank regression parameters

%% Plot results

fntsz = 12;

% Plot comparision of noise precision
maxx = max([lambhat d]);
minx = min([lambhat d]);

figure(4);
loglog(lambhat,d,'o','markersize',10)
hold on
plot([minx maxx],[minx maxx],'k')
hold off
title('Noise precision','fontsize',fntsz)
xlabel('Estimate','fontsize',fntsz)
ylabel('True','fontsize',fntsz)
% axis([minx maxx minx maxx])
box off
axis tight
axis equal
set(gcf,'color','w')


% Plot comparision of regression params
Bhat{P} = Bhat0';
figure(5);
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
    title(['Task var ' num2str(p)],'fontsize',fntsz)
    set(gca,'tickdir','out')
    box off
    if p==1
        xlabel('estimate','fontsize',fntsz)
    ylabel('true','fontsize',fntsz)
    end
    axis tight
    axis equal
end
subplot(1,P,P)
title('Task indep. component','fontsize',fntsz)
set(gcf,'color','w')

