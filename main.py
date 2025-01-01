import numpy as np
import pandas as pd
from model import Graph
import matplotlib.pyplot as plt

g = Graph()

# a testing dataset from kaggle
df = pd.read_csv("/Users/sakinkirti/Downloads/heart_disease.csv")
for col in ["currentSmoker", "BPMeds", "prevalentHyp", "diabetes"]:
    df[col] = df[col].map(lambda x: np.where(x == 1, "yes", "no"))

# load just the nodes from 
g.load_nodes_from_dataframe(df)

# add specific edges
g.add_edge("cigsPerDay", "currentSmoker")
g.add_edge("cigsPerDay", "heartRate")
g.add_edge("BMI", "heartRate")
g.add_edge("glucose", "diabetes")
g.add_edge("totChol", "prevalentStroke")
g.add_edge("BMI", "diabetes")

vis = g.visualize_graph(42)
plt.show()

print(g)
