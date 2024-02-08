import json


import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

all_node_type = ['inputs', 'outputs', 'input', 'kind', 'operands', 'op', 'Project', 'exprs', 'TableScan', 'table', 'condition', 'literal', 'rels', 'joinType', 'HashJoin', 'Filter', 'HashAggregate', 'group', 'aggs', 'agg', 'field', 'direction', 'nulls', 'Sort', 'collation', 'Limit', 'fetch', 'distinct', 'NestedLoopJoin', 'all', 'Union', 'Minus', 'Window', 'window', 'agg_calls', 'partition_by', 'order_by', 'frame_type', 'boundary_type_start', 'boundary_type_end', 'Correlate', 'correlation', 'requiredColumns', 'Intersect', 'expr', 'correl', 'Values', 'tuples']

def draw_graph(node, edge, edge_type):

    global all_node_type

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes to the graph
    for i, (node_type) in enumerate(zip(node)):
        node_color = 'lightblue' if node_type in all_node_type else 'pink'
        G.add_node(i, label=node_type, node_color=node_color)

    # Add edges to the graph with edge type information
    for i, et in enumerate(edge_type):
        # if et == 'ast_edge':
        G.add_edge(edge[i][0], edge[i][1], edge_type=et)

    # Tree 排版
    pos = graphviz_layout(G, prog="dot")

    # 图的大小
    plt.figure(figsize=(20, 15))

    # 定义边的颜色
    edge_colors = {'ast_edge': 'black', 'data_edge': 'green', 'new_data_edge': 'red', 'logic_edge': 'orange'}

    # 画图 with labels, colors, and edge labels
    nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), font_size=4, node_size=500,
            node_color=[G.nodes[n]['node_color'] for n in G.nodes()], font_color='black', font_weight='bold', arrowsize=10,
            edge_color=[edge_colors[G[e[0]][e[1]]['edge_type']] for e in G.edges()], linewidths=2)

    plt.title('Tree Visualization')

    plt.savefig("graph.png", dpi=500)  # 将图保存为文件并指定dpi
    plt.show()
