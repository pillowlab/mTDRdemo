function [W, S, BB] = SimWeights(n,T,P,rP,len,rho)


% Random weights
BB = [];
for p = 1:P
    Scov = toeplitz(exp(-((0:T-1)/len(p)).^2/2));
    Wp = rho(p)*randn(n,rP(p));
    W{p} = Wp;
    Sp = mvnrnd(zeros(T,1),Scov,rP(p))';
    Sp = flip(Sp);
    S{p} = Sp;
    BB = [BB;Wp*Sp'];
end
