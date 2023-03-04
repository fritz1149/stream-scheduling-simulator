param node_num, integer, >0; #算子数量
param edge_num, integer, >0; #流式边数量
set nodes := 0..node_num-1;
set edges := 0..edge_num-1;
param endpoint{edges, 0..1}, integer; #流式计算图的关联矩阵
param flow{edges}, >0; #流式计算边流量
param source_num;
param sink_num;
param sources{0..source_num-1}, integer;
param sinks{0..sink_num-1}, integer;

var color{nodes}, binary;
var edge_cross{edges}, binary;

maximize cut: 
    sum{i in edges} edge_cross[i] * flow[i];
s.t. source_color{i in 0..source_num-1}: color[sources[i]] = 0;
s.t. sink_color{i in 0..sink_num-1}: color[sinks[i]] =1;
s.t. xor1{i in edges}: edge_cross[i] <= color[endpoint[i,0]] + color[endpoint[i,1]];
s.t. xor2{i in edges}: edge_cross[i] >= color[endpoint[i,0]] - color[endpoint[i,1]];
s.t. xor3{i in edges}: edge_cross[i] >= color[endpoint[i,1]] - color[endpoint[i,0]];
s.t. xor4{i in edges}: edge_cross[i] <= 2 - color[endpoint[i,0]] - color[endpoint[i,1]];

solve;

# display color;
printf "%d\n", cut; 