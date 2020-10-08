function [wsvd, wt,wx] = SVDRegressB(xx,xy,wdims,ps,lambda,opts)
if (nargin >= 5) && ~isempty(lambda)  
    xx = xx + lambda*speye(size(xx)); % add ridge penalty to xx
end

if (nargin < 6) || isempty(opts)
    opts.default = true;
end

if ~isfield(opts, 'MaxIter'); opts.MaxIter = 25; end
if ~isfield(opts, 'TolFun'); opts.TolFun = 1e-6; end
if ~isfield(opts, 'Display'); opts.Display = 'iter'; end

% Set some params
nwtot = length(xy); % total # of regression coeffs (for linearly parametrized problem)
nt = wdims(1); % height of each matrix of coeffs
nx = wdims(2); % width of each matrix of coeffs
nw = nt*nx; % # coeffs per matrix
nmats = length(ps); % # distinct low-rank matrices
if nwtot ~= nw*nmats
    error('Mismatch in size of data (xx,xy) and size of params (wdim,p)');
end

% Create some sparse identity matrices we'll need
It = speye(nt);
Ix = speye(nx);
Mxkron = cell(1,nmats);  % initialize space for kronecker matrices
Mtkron = cell(1,nmats);  % initialize space for kronecker matrices

% Make permuation indices for reshaping wx coeffs
cumrank = [0,cumsum(ps)];
iixperm = zeros(nx,cumrank(end));
for jj = 1:nmats
    ii = reshape((cumrank(jj))*nx+1:cumrank(jj+1)*nx,ps(jj),nx)';
    iixperm(:,cumrank(jj)+1:cumrank(jj+1)) = reshape(ii,nx,[]);
end

% Initialize estimate of w by linear regression and SVD on each matrix
w0 = xx\xy;
wt = cell(1,nmats); % initialize column vecs cell array
wx = cell(1,nmats); % initialize row vecs cell array
iistrt = 1+nw*(0:nmats-1); % starting index for each coeff matrix 
iiend = nw*(1:nmats); % last index for each coeff
wsvd = zeros(nt,nx,nmats);
for jj = 1:nmats
    [wt0,s,wx0] = svd(reshape(w0(iistrt(jj):iiend(jj)),nt,nx));
    wt{jj} = wt0(:,1:ps(jj))*sqrt(s(1:ps(jj),1:ps(jj)));
    wx{jj} = wx0(:,1:ps(jj))*sqrt(s(1:ps(jj),1:ps(jj)));
    wsvd(:,:,jj) = wt{jj}*wx{jj}';
end
% wsvd = vec(wsvd);
