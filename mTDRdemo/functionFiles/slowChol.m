function C = slowChol(A)

na = size(A);
C = zeros(na);

for ii = 1:na(3)
    C(:,:,ii) = chol(A(:,:,ii));
end
    