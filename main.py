import networkx as nx
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# dt: how the data is stored in the graph
# ('len' is the attribute that the values will go into,
#  and they will be floats)
dt = [('len', float)]

# graph A: distance array (from CSCI 321)
A = np.array([(0, 300, np.Inf, np.Inf, 2161, 2661, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf),
              (300, 0, 629, 919, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf),
              (np.Inf, 629, 0, 435, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf),
              (np.Inf, 919, 435, 0, 1225, np.Inf, 1983, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf),
              (2161, np.Inf, np.Inf, 1225, 0, 1483, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf),
              (2661, np.Inf, np.Inf, np.Inf, 1483, 0, 1582, 661, np.Inf, np.Inf, np.Inf, np.Inf),
              (np.Inf, np.Inf, np.Inf, 1983, np.Inf, 1582, 0, np.Inf, 2113, np.Inf, np.Inf, 2161),
              (np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, 661, np.Inf, 0, np.Inf, 1613, np.Inf, np.Inf),
              (np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, 2113, np.Inf, 0, 725, 383, 1709),
              (np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, 1613, 725, 0, 328, np.Inf),
              (np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, 383, 328, 0, 2145),
              (np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, 2161, np.Inf, 1709, np.Inf, 2145, 0)])
# apply dt
A = A.view(dt)

# graph B: graph A after it was passed through my implementation of Prim's algorithm
# aka, graph A's Minimum Spanning Tree
B = np.array([(0, 300, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf),
              (300, 0, 629, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf),
              (np.Inf, 629, 0, 435, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf),
              (np.Inf, np.Inf, 435, 0, 1225, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf),
              (np.Inf, np.Inf, np.Inf, 1225, 0, 1483, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf),
              (np.Inf, np.Inf, np.Inf, np.Inf, 1483, 0, 1582, 661, np.Inf, np.Inf, np.Inf, np.Inf),
              (np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, 1582, 0, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf),
              (np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, 661, np.Inf, 0, np.Inf, 1613, np.Inf, np.Inf),
              (np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, 0, np.Inf, 383, 1709),
              (np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, 1613, np.Inf, 0, 328, np.Inf),
              (np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, 383, 328, 0, np.Inf),
              (np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, 1709, np.Inf, np.Inf, 0)])
# apply dt
B = B.view(dt)

# names: the names of the locations of which the graphs above represent the distances between
names = ["Seattle",
         "San Francisco",
         "Los Angeles",
         "Las Vegas",
         "Denver",
         "Minneapolis",
         "Dallas",
         "Chicago",
         "Washington DC",
         "Boston",
         "New York",
         "Miami"]

# Ax, Bx: the networkx graph representations of graphs A and B
Ax = nx.from_numpy_array(A)
Bx = nx.from_numpy_array(B)

# relabeling the nodes, with the appropriate names
Ax = nx.relabel_nodes(Ax, dict(zip(range(len(Ax.nodes())), names)))
Bx = nx.relabel_nodes(Bx, dict(zip(range(len(Bx.nodes())), names)))

# to represent an edge that is invalid, i filled it with an Infinite distance
# so, to get rid of these edges, i grab them into invalidE
invalidE = [(u, v) for (u, v, d) in Ax.edges(data=True) if d["len"] == np.Inf]
# then remove
Ax.remove_edges_from(invalidE)
# grab into invalidE
invalidE = [(u, v) for (u, v, d) in Bx.edges(data=True) if d["len"] == np.Inf]
# and remove
Bx.remove_edges_from(invalidE)

# in order to make the graph approximate the distances between nodes properly,
# im using networkx's "spring layout"
# however, it has this annoying quirk, where the edge weights for it
# represent a *tension* instead of a *distance*
# so to counteract this, i give each edge a new attribute to represent the tension
# with the attribute being equivalent
# to the reciprocal of the edge length
# to do this, i copy down all lengths:
lengthsA = nx.get_edge_attributes(Ax, "len")
# iterate through them:
for source, value in lengthsA:
    # and for each length, calculate the reciprocal
    lengthsA[(source, value)] = 1.0 / lengthsA[(source, value)]
# and finally, add the new attributes onto the edges of graph Ax
nx.set_edge_attributes(Ax, lengthsA, 'tension')

# finally, i initialize the spring layout
# using the new tension attribute
# i only calculate it for graph Ax, as i will use the same layout for both graphs
pos = nx.spring_layout(Ax, seed=69, iterations=1024, weight='tension', k=0.13)

# with the graphs made, and node positions calculated, all thats left is to draw the graphs
# so i draw a graph,
nx.draw(Ax, pos, node_size=600)
# and its node labels
nx.draw_networkx_labels(Ax, pos)
# and its edge labels
edge_labels = nx.get_edge_attributes(Ax, "len")
nx.draw_networkx_edge_labels(Ax, pos, edge_labels, font_size=8)

# and output the graph
plt.show()

# and repeat for graph B:
nx.draw(Bx, pos, node_size=600)
nx.draw_networkx_labels(Bx, pos)
edge_labels = nx.get_edge_attributes(Bx, "len")
nx.draw_networkx_edge_labels(Bx, pos, edge_labels, font_size=8)
plt.show()
