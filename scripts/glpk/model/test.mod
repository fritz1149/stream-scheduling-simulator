set I;
/* canning plants */
set J;
/* markets */
param a{I};
/* capacity of plant i in cases */
param b{J};
/* demand at market j in cases */
param d{I, J};
/* distance in thousands of miles */
param f;
/* freight in dollars per case per thousand miles */
param c{i in I, j in J} := f * d[i,j] / 1000;
/* transport cost in thousands of dollars per case */
var x{I, J} >= 0;
/* shipment quantities in cases */
minimize cost: sum{i in I, j in J} c[i,j] * x[i,j];
/* total transportation costs in thousands of dollars */
s.t. supply{i in I}: sum{j in J} x[i,j] <= a[i];
/* observe supply limit at plant i */
s.t. demand{j in J}: sum{i in I} x[i,j] >= b[j];
/* satisfy demand at market j */
solve;
display x;
for {i in 1..5} printf "i= %d\n", i ;
data;
set I := Seattle San-Diego;
set J := New-York Chicago Topeka;
param a := Seattle 350
           San-Diego 600;
param b := New-York 325
           Chicago 300
           Topeka 275;
param d :       New-York  Chicago  Topeka :=
    Seattle     2.5       1.7      1.8
    San-Diego   2.5       1.8      1.4 ;
param f := 90;
end;