param A{0..1, 0..1};
param C{0..1};
param D{0..1, 0..1};
var B{0..1};
# minimize obj: 0;
s.t. multiply{i in 0..1}: A[i,0] * B[0] + A[i,1] * B[1] = 
                C[i] + D[i,0] * B[0] + D[i,1] * B[1];

solve;
display B;
data;
param A : 0 1 := 
    0 2 1
    1 1 1;
param C :=
    0 5
    1 3;
param D: 0 1 :=
    0 1 0
    1 1 0;