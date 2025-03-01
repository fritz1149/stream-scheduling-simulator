# 参数
## 流式计算图相关
param flow_node_num, integer, >0; #算子数量
param flow_edge_num, integer, >0; #流式边数量
set flow_nodes := 0..flow_node_num-1;
set flow_edges := 0..flow_edge_num-1;
param flow_incidence{flow_nodes, flow_edges}, integer; #流式计算图的关联矩阵
param mi{flow_nodes}, >0; #算子平均每条数据所需算力
param flow{flow_edges}, >0; #流式计算边流量
param urate{flow_edges}, >0; #流式计算边数据频率
param flow_node_is_sink{flow_nodes}, binary;
param tuple_size{flow_edges}, >0; #流式计算边一个数据元组的数据量
param flow_endpoint{flow_edges, 0..1}, integer; #流式边端点
param in_flow_edge{flow_nodes, flow_edges}, binary;
param in_urate_sum_reciprocal{flow_nodes}; #算子入边数据频率之和的倒数
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
# param net_path_incidence{net_nodes, net_paths}, integer; #网络链路的关联矩阵
param net_path_endpoint{net_paths, 0..1}, integer; #网络边端点
param net_edge_in_path{net_edges, net_paths}, binary; #网络边是否在网络链路上
param bandwidth{net_edges}, >0; #网络边的带宽上限
param net_edge_intr_lat{net_edges}, >0; #网络边固有延迟
param mips_reciprocal{net_nodes}, >=0; #计算节点单核算力
param mips_positive{net_nodes}, binary; #计算节点是否具有计算能力
param cores{net_nodes}, integer, >=0; #计算节点核数
param cores_reciprocal{net_nodes}; #计算节点核数的倒数
param slot{net_nodes}, integer, >=0; #算子插槽限制
param net_node_in_edge{net_nodes, edge_domains}, binary; #算子是否在某边缘域
param net_node_in_cloud{net_nodes}, binary; #算子是否在云域
param M := 1000000000;
## 目标计算相关
param lat_min, >0;
# param lat_max, >0;
param flow_min, >0;
# param flow_max, >0;
# 变量
## 决策变量
var f{flow_nodes, net_nodes}, binary; #算子到计算节点的映射
## 中间变量
var net_node_occupied{net_nodes}, integer, >=0;
var net_node_occupied_real{net_nodes}, integer, >=0;
var net_node_occupied_with_f{flow_nodes, net_nodes}, integer, >=0;
var u1, binary;
var u2, binary;
var flow_edge_as_net_path{flow_edges, net_paths}, binary;
var flow_edge_as_net_path_origin{flow_edges, net_paths}, binary;
var flow_edge_as_net_path_dest{flow_edges, net_paths}, binary;
var net_edge_in_flow_edge{net_edges, flow_edges}, binary;
### 延迟相关
#### 计算延迟
var comp_lat{flow_nodes}, >=0;
var net_edge_flow_sum{net_edges};
var net_edge_flow_sum_with_flow_edge{net_edges, flow_edges}, integer, >=0;
#### 固有延迟
var flow_edge_intr_lat{flow_edges}, >=0;
#### 传输延迟
var tran_lat{flow_edges}, >=0;
#### 节点延迟
var lat_operator{flow_nodes}, >=0;
## 指标
### 延迟相关
var lat, >=0;
### 流量相关
var flow_edge_cross{flow_edges}, binary; #流式边是否跨云边
var flow_node_in_cloud{flow_nodes}, binary; #算子是否在云域
var flow_cross; #跨云边流量
# 目标
minimize obj: flow_cross / flow_min + lat / lat_min;
# 主要约束
subject to 
    # 不允许边边通信、从云到边的通信
    no_edge_to_edge_1{i in flow_edges, j in edge_domains}:  
        sum{k in flow_nodes, l in net_nodes}
        flow_incidence[k,i] * f[k,l] * net_node_in_edge[l,j], <=1;
    no_edge_to_edge_2{i in flow_edges, j in edge_domains}:  
        sum{k in flow_nodes, l in net_nodes}
        flow_incidence[k,i] * f[k,l] * net_node_in_edge[l,j], >=0;
    ## 特定算子部署到特定节点集的限制
    pos_restr{i in flow_nodes, j in net_nodes}: f[i,j] <= flow_node_restr[i,j];
    ## 计算节点插槽数量限制
    net_node_occupied_real_{i in net_nodes}: sum{j in flow_nodes} f[j,i] = net_node_occupied_real[i];
    slot_capacity{i in net_nodes}: net_node_occupied_real[i] <= slot[i];
    ## 一个算子只能部署到一个计算节点
    one_operator_to_one_node{i in flow_nodes}: sum{j in net_nodes} f[i,j] =1;
    ## 算子只能部署到mips非0的节点上
    node_mips_positive{i in flow_nodes}: sum{j in net_nodes} f[i,j] * mips_positive[j] >= 1;
    ## 流式边与网络路径的起点对应
    flow_edge_as_net_path_origin_{i in flow_edges, j in net_paths}:
        flow_edge_as_net_path_origin[i,j] = f[flow_endpoint[i,0], net_path_endpoint[j,0]];
    ## 流式边与网络路径的终点对应
    flow_edge_as_net_path_dest_{i in flow_edges, j in net_paths}:
        flow_edge_as_net_path_dest[i,j] = f[flow_endpoint[i,1], net_path_endpoint[j,1]];
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
# 中间变量约束
## 延迟相关
subject to
    ### 每条网络边承载的流式边流量之和
    net_edge_flow_sum_{i in net_edges}:
        net_edge_flow_sum[i] = sum{j in flow_edges} flow[j] * net_edge_in_flow_edge[i, j];
    ### 每个计算节点分配的任务数
    net_node_occupied_0{i in net_nodes}:
        net_node_occupied[i] >= net_node_occupied_real[i];
    net_node_occupied_1{i in net_nodes}:
        net_node_occupied[i] >= cores[i];
    net_node_occupied_2{i in net_nodes}:
        net_node_occupied[i] <= M * (1 - u1) + net_node_occupied_real[i];
    net_node_occupied_3{i in net_nodes}:
        net_node_occupied[i] <= M * (1 - u2) + cores[i];
    net_node_occupied_4{i in net_nodes}:
        u1 + u2 = 1;
    ### f[flow_node, net_node] * net_node_occupied[net_node] if else语义转换成线性表达
    net_node_occupied_with_f_0{i in flow_nodes, j in net_nodes}:
        net_node_occupied_with_f[i,j] <= net_node_occupied[j] + M * (1 - f[i,j]);
    net_node_occupied_with_f_1{i in flow_nodes, j in net_nodes}:
        net_node_occupied_with_f[i,j] >= net_node_occupied[j] - M * (1 - f[i,j]);
    net_node_occupied_with_f_2{i in flow_nodes, j in net_nodes}:
        net_node_occupied_with_f[i,j] <= M * f[i,j];
    net_node_occupied_with_f_3{i in flow_nodes, j in net_nodes}:
        net_node_occupied_with_f[i,j] >= -1 * M * f[i,j];
    ### net_edge_in_flow_edge[net_edge, flow_edge] * net_edge_flow_sum[net_edge] if else语义转换成线性表达
    net_edge_flow_sum_with_flow_edge_0{i in net_edges, j in flow_edges}:
        net_edge_flow_sum_with_flow_edge[i,j] <= net_edge_flow_sum[i] + M * (1 - net_edge_in_flow_edge[i,j]);
    net_edge_flow_sum_with_flow_edge_1{i in net_edges, j in flow_edges}:
        net_edge_flow_sum_with_flow_edge[i,j] >= net_edge_flow_sum[i] - M * (1 - net_edge_in_flow_edge[i,j]);
    net_edge_flow_sum_with_flow_edge_2{i in net_edges, j in flow_edges}:
        net_edge_flow_sum_with_flow_edge[i,j] <= M * net_edge_in_flow_edge[i,j];
    net_edge_flow_sum_with_flow_edge_3{i in net_edges, j in flow_edges}:
        net_edge_flow_sum_with_flow_edge[i,j] >= -1 * M * net_edge_in_flow_edge[i,j];
    ### 计算延迟
    comp_lat_lowerbound{i in flow_nodes}:
        comp_lat[i] = mi[i] * 1000 * ( sum{j in net_nodes}
            net_node_occupied_with_f[i,j] * cores_reciprocal[j] * mips_reciprocal[j]
        );
    ### 固有延迟
    flow_edge_intr_lat_{i in flow_edges}:
        flow_edge_intr_lat[i] = sum{j in net_edges} net_edge_in_flow_edge[j,i] * net_edge_intr_lat[j];
    ### 传输延迟
    tran_lat_lowerbound{i in flow_edges}:
        tran_lat[i] = sum{j in net_edges} tuple_size[i] * 1000 * net_edge_flow_sum_with_flow_edge[j,i] / (bandwidth[j] * flow[i]);
    ### 节点延迟
    lat_operator_{i in flow_nodes}:
        lat_operator[i] = comp_lat[i] + 
            (sum{j in flow_edges} in_flow_edge[i,j] * urate[j] * (lat_operator[flow_endpoint[j,0]] + tran_lat[j] + flow_edge_intr_lat[j])) *
            in_urate_sum_reciprocal[i];
    lat_:
        lat = (sum{i in flow_nodes} flow_node_is_sink[i] * lat_operator[i]) /
              (sum{i in flow_nodes} flow_node_is_sink[i]);
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
printf "%f\n", lat;
printf "%f\n", flow_cross;
printf "%f\n", obj;
# for {i in flow_nodes} {
#     for {j in net_nodes} {
#         printf "%d ", f[i,j];
#     }
#     printf "\n";
# }
end;