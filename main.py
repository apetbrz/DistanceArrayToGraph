import networkx as nx
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

dt = [('len', float)]

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

A = A.view(dt)

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
              (np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, 1709, np.Inf, np.Inf, 0),
              ])

B = B.view(dt)

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

Ax = nx.from_numpy_array(A)
Bx = nx.from_numpy_array(B)

Ax = nx.relabel_nodes(Ax, dict(zip(range(len(Ax.nodes())), names)))
Bx = nx.relabel_nodes(Bx, dict(zip(range(len(Bx.nodes())), names)))

invalidE = [(u, v) for (u, v, d) in Ax.edges(data=True) if d["len"] == np.Inf]
Ax.remove_edges_from(invalidE)
invalidE = [(u, v) for (u, v, d) in Bx.edges(data=True) if d["len"] == np.Inf]
Bx.remove_edges_from(invalidE)

lengthsA = nx.get_edge_attributes(Ax, "len")
for source, value in lengthsA:
    lengthsA[(source, value)] = 1.0 / lengthsA[(source, value)]

nx.set_edge_attributes(Ax, lengthsA, 'spring')

pos = nx.spring_layout(Ax, seed=69, k=0.1, iterations=1024, weight='spring')

nx.draw(Ax, pos, node_size=600)
nx.draw_networkx_labels(Ax, pos)
edge_labels = nx.get_edge_attributes(Ax, "len")
nx.draw_networkx_edge_labels(Ax, pos, edge_labels, font_size=8)

plt.show()

nx.draw(Bx, pos, node_size=600)
nx.draw_networkx_labels(Bx, pos)
edge_labels = nx.get_edge_attributes(Bx, "len")
nx.draw_networkx_edge_labels(Bx, pos, edge_labels, font_size=8)
plt.show()
