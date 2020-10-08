function C = slowBackslash(A,B)

na = size(A);
nb = size(B);

if na(3)~=nb(3)
    error('Third dimension of matrices must match')
else
    C = zeros(na(1),nb(2),na(3));
    for ii = 1:na(3)
        C(:,:,ii) = A(:,:,ii)\B(:,:,ii);
    end
end
