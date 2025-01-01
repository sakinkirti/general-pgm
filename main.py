import numpy as np
from model import Graph
import matplotlib.pyplot as plt

g = Graph()
g.add_node("cad", storage_type="cat", values=["high", "low", "none"], probs=[0.1, 0.3, 0.6])
g.add_node("mi", storage_type="cont", mean=4, standard_deviation=1)
g.add_node("bmi", storage_type="cont", mean=26, standard_deviation=4)
g.add_node("diabetes", storage_type="cat", values=["yes", "no"], probs=[0.3, 0.7])

g.add_edge("cad", "mi")
g.add_edge("bmi", "mi")
g.add_edge("bmi", "cad")
g.add_edge("diabetes", "bmi")
g.add_edge("diabetes", "cad")

vis = g.visualize_graph(42)
plt.show()
