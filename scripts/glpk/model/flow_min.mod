# 参数
## 流式计算图相关
param flow_node_num, integer, >0; #算子数量
param flow_edge_num, integer, >0; #流式边数量
set flow_nodes := 0..flow_node_num-1;
set flow_edges := 0..flow_edge_num-1;
param flow_incidence{flow_nodes, flow_edges}, integer; #流式计算图的关联矩阵
param mi{flow_nodes}, >0; #算子平均每条数据所需算力
param flow{flow_edges}, >0; #流式计算边流量
param flow_node_is_sink{flow_nodes}, binary;
param tuple_size{flow_edges}, >0; #流式计算边一个数据元组的数据量
param in_flow_edge{flow_nodes, flow_edges}, binary; #算子的入边
param out_flow_edge{flow_nodes, flow_edges}, binary; #算子的出边
param in_flow_sum_reciprocal{flow_nodes}; #算子入边流量之和的倒数
param flow_edge_s{flow_edges}, symbolic in flow_nodes; #流式计算边起点
## 实际网络相关
param net_node_num, integer, >0; #计算节点数量
param net_edge_num, integer, >0; #网络边数量
param net_path_num, integer, >0; #网络链路数量
param edge_domain_num, integer, >0; #边缘域数量
set net_nodes := 0..net_node_num-1;
set net_edges := 0..net_edge_num-1;
set net_paths := 0..net_path_num-1;
set edge_domains := 0..edge_domain_num-1;
param flow_node_restr{flow_nodes, net_nodes}, binary; #算子部署位置限制
param net_incidence{net_nodes, net_edges}, integer; #网络边的关联矩阵#
# param net_path_incidence{net_nodes, net_paths}, integer; #网络链路的关联矩阵
param net_path_origin{net_nodes, net_paths}, binary; #网络链路的起点
param net_path_dest{net_nodes, net_paths}, binary; #网络链路的终点
param net_edge_in_path{net_edges, net_paths}, binary; #网络边是否在网络链路上
param bandwidth{net_edges}, >0; #网络边的带宽上限
param net_edge_intr_lat{net_edges}, >0; #网络边固有延迟
param mips_reciprocal{net_nodes}, >=0; #计算节点单核算力
param mips_positive{net_nodes}, binary; #计算节点是否具有计算能力
param cores{net_nodes}, integer, >=0; #计算节点核数
param slot{net_nodes}, integer, >=0; #算子插槽限制
param net_node_in_edge{net_nodes, edge_domains}, binary; #算子是否在某边缘域
param net_node_in_cloud{net_nodes}, binary; #算子是否在云域
## 目标计算相关
param lat_min, >0;
param lat_max, >0;
param flow_min, >0;
param flow_max, >0;
# 变量
## 决策变量
var f{flow_nodes, net_nodes}, binary; #算子到计算节点的映射
## 中间变量
var flow_edge_as_net_path{flow_edges, net_paths}, binary;
var flow_edge_as_net_path_origin{flow_edges, net_paths}, binary;
var flow_edge_as_net_path_dest{flow_edges, net_paths}, binary;
var net_edge_in_flow_edge{net_nodes, flow_edges}, binary;
## 指标
### 流量相关
var flow_edge_cross{flow_edges}, binary; #流式边是否跨云边
var flow_node_in_cloud{flow_nodes}, binary; #算子是否在云域
var flow_cross; #跨云边流量
# 目标
minimize obj: flow_cross;
# 主要约束
subject to 
    ## 不允许边边通信、从云到边的通信
    no_edge_to_edge_1{i in flow_edges, j in edge_domains}:  
        sum{k in flow_nodes, l in net_nodes}
        flow_incidence[k,i] * f[k,l] * net_node_in_edge[l,j], <=1;
    no_edge_to_edge_2{i in flow_edges, j in edge_domains}:  
        sum{k in flow_nodes, l in net_nodes}
        flow_incidence[k,i] * f[k,l] * net_node_in_edge[l,j], >=0;
    ## 特定算子部署到特定节点集的限制
    pos_restr{i in flow_nodes, j in net_nodes}: f[i,j] * flow_node_restr[i,j] = f[i,j];
    ## 计算节点插槽数量限制
    slot_capacity{i in net_nodes}: sum{j in flow_nodes} f[j,i] <= slot[i];
    ## 一个算子只能部署到一个计算节点
    one_operator_to_one_node{i in flow_nodes}: sum{j in net_nodes} f[i,j] =1;
    # ## 算子只能部署到mips非0的节点上
    node_mips_positive{i in flow_nodes}: sum{j in net_nodes} f[i,j] * mips_positive[j] >= 1;
    ## 流式边与网络路径的起点对应
    flow_edge_as_net_path_origin_{i in flow_edges, j in net_paths}:
        flow_edge_as_net_path_origin[i,j] = sum{k in flow_nodes, l in net_nodes} out_flow_edge[k,i] * f[k,l] * net_path_origin[l,j];
    ## 流式边与网络路径的终点对应
    flow_edge_as_net_path_dest_{i in flow_edges, j in net_paths}:
        flow_edge_as_net_path_dest[i,j] = sum{k in flow_nodes, l in net_nodes} in_flow_edge[k,i] * f[k,l] * net_path_dest[l,j];
    ## 算子映射f与边映射的关系（逻辑与）
    flow_edge_as_net_path_1{i in flow_edges, j in net_paths}:
        flow_edge_as_net_path[i,j] >= flow_edge_as_net_path_origin[i,j] + flow_edge_as_net_path_dest[i,j] - 1;
    flow_edge_as_net_path_2{i in flow_edges, j in net_paths}:
        flow_edge_as_net_path[i,j] <= flow_edge_as_net_path_origin[i,j];
    flow_edge_as_net_path_3{i in flow_edges, j in net_paths}:
        flow_edge_as_net_path[i,j] <= flow_edge_as_net_path_dest[i,j];
    ## 网络边在流式计算边内
    net_edge_in_flow_edge_{i in net_edges, j in flow_edges}:
        sum{k in net_paths} net_edge_in_path[i,k] * flow_edge_as_net_path[j,k] = net_edge_in_flow_edge[i,j];
## 跨云边流量相关
subject to
    flow_node_in_cloud_{i in flow_nodes}:
        flow_node_in_cloud[i] = sum{j in net_nodes} f[i,j] * net_node_in_cloud[j];
    flow_edge_cross_{i in flow_edges}:
        flow_edge_cross[i] = -1 * 
            (sum{j in flow_nodes} flow_incidence[j,i] * flow_node_in_cloud[j]);
    flow_cross_:
        flow_cross = sum{i in flow_edges} flow_edge_cross[i] * flow[i];
solve;
printf "%f\n", flow_cross;
printf "%f\n", obj;
# for {i in flow_nodes} {
#     for {j in net_nodes} {
#         printf "%d ", f[i,j];
#     }
#     printf "\n";
# }
end;