import numpy as np
import pandas as pd
from model import ProbabilisticGraph
import matplotlib.pyplot as plt

pgm = ProbabilisticGraph()

# a testing dataset from kaggle
df = pd.read_csv("/Users/sakinkirti/Downloads/heart_disease.csv")
for col in ["currentSmoker", "BPMeds", "prevalentHyp", "diabetes"]:
    df[col] = df[col].map(lambda x: np.where(x == 1, "yes", "no"))

# load just the nodes from 
pgm.load_nodes_from_dataframe(df)

# add specific edges (used a doctor's help to make these connections), want to implement some automated edge creation - need to do some research on how to do this
pgm.add_edge("cigsPerDay", "currentSmoker")
pgm.add_edge("cigsPerDay", "heartRate")
pgm.add_edge("BMI", "heartRate")
pgm.add_edge("glucose", "diabetes")
pgm.add_edge("totChol", "prevalentStroke")
pgm.add_edge("BMI", "diabetes")
pgm.add_edge("cigsPerDay", "Heart_ stroke")
pgm.add_edge("diabetes", "Heart_ stroke")
pgm.add_edge("prevalentHyp", "Heart_ stroke")
pgm.add_edge("education", "prevalentHyp")
pgm.add_edge("education", "glucose")
pgm.add_edge("education", "totChol")
pgm.add_edge("education", "cigsPerDay")
pgm.add_edge("education", "BMI")
pgm.add_edge("Gender", "Heart_ stroke")
pgm.add_edge("age", "diaBP")
pgm.add_edge("age", "sysBP")
pgm.add_edge("prevalentHyp", "diaBP")
pgm.add_edge("prevalentHyp", "sysBP")
pgm.add_edge("BPMeds", "diaBP")
pgm.add_edge("BPMeds", "sysBP")
pgm.add_edge("BPMeds", "heartRate")
pgm.add_edge("prevalentHyp", "BPMeds")

vis = pgm.visualize_graph(1234)
plt.show()

print(pgm.conditional_query("Heart_ stroke", "yes", conditional_nodes=["Gender", "diabetes", "prevalentHyp"], conditional_values=["Female", "yes", "yes"]))
