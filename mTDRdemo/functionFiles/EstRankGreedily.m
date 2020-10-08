function [rest, rhist, FunHist,parhist] = EstRankGreedily(ObjFun,estfun,r0,maxrank,stepthresh,histfile)


% check dimensions of r
[P1, P2] = size(r0);
if P1==1
    rest = r0;
    P = P2;
else
    rest = r0';
    P = P1;
end

rhist = rest;
if isempty(stepthresh)
    stepthresh = 0;
end

% Initialize with ranks-r0
% [shatbilin0,lambhati] = estfun(rest);
% parhat0 = [lambhati;shatbilin0];
parhat0 = estfun(rest);
Obj0 = ObjFun(parhat0,rest);

FunHist = Obj0;
iters = 0;

%% Iterate until stopping criteria are met
while all(rest<=maxrank)
    g=sprintf('%d ', rest);
    fprintf('Current estimate: r = %s, AIC = %6.3f\n', g,Obj0)
    
    % Check that step forward in each dimension is well-defined
    Obj = zeros(P,1);
    indmove = find(rest<maxrank);
    for p = 1:P
        if (rest(p))<maxrank
            
            % Set rank vector
            plusvec = zeros(1,P);   plusvec(p) = 1;
            rtest = rest + plusvec;
            
            % Update parameters
            parhat{p} = estfun(rtest);
            Obj(p) = ObjFun(parhat{p},rtest);
        end
    end
    % Decide whether to step
    dObj = (Obj - Obj0);
    if all(dObj(indmove)>stepthresh)
        fprintf('Stop estimation: Insufficient improvement in log likelihood.\nTerminating rank estimation.\n')
        break
    end
    
    % Increment in direction of greatest decrease in AIC
    [mindiff, indmin] = min(dObj(indmove));
    if numel(mindiff)>1
        fprintf('Tie\n')
        indmin = randsample(indmove(indmin),1);
    end
    rest(indmove(indmin)) = rest(indmove(indmin)) + 1;
    Obj0 = Obj(indmove(indmin));
    FunHist = [FunHist;Obj0];
    rhist = [rhist;rest];
    
    iters = iters + 1;
    parhist{iters} = parhat{indmin};
    if ~isempty(histfile)
        save(histfile,'rhist','FunHist','parhist');
    else
        error('No history file')
    end
    
end

% Save results in the case of no change in rank from r0
if iters==0
    parhist{1} = parhat0;
    FunHist = Obj0;
    rhist = r0;
    save(histfile,'rhist','FunHist','parhist');
end
