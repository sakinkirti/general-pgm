import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from typing import Literal

class ProbabilisticGraph:
    
    def __init__(self):
        
        # initialize an empty graph
        self.graph = {}
        
    def __str__(self):
        """overrides the default __str__ method
        
        Args:
            None
            
        Returns:
            str: a string representation of the graph
        """

        res = ""
        for k,v in self.graph.items():
            res += f"{v.__str__()}\n"
            
        return res
            
    def add_node(self, name:str, storage_type:Literal["cat", "cont"], *args, **kwargs):
        """adds a node to the graph

        Args:
            name (str): the name of the node
            storage_type (str): the type of data to store
        """
        
        # add a categorical or continuous node as needed
        if storage_type == "cat":
            self.graph[name] = self.CategoricalNode(name=name, values=kwargs["values"], probs=kwargs["probs"])
        elif storage_type == "cont":
            self.graph[name] = self.ContinuousNode(name=name, mean=kwargs["mean"], standard_deviation=kwargs["standard_deviation"])
        else:
            raise ValueError("The given storage type was invalid")
        
    def add_edge(self, source:str, destination:str):
        """adds a directed edge between two nodes

        Args:
            source (str): the node that the edge starts from
            destination (str): the node that the edges ends at
        """
        
        # let each node know that it has an outgoing or incoming edge
        self.graph[source].add_outgoing(destination=destination)
        self.graph[destination].add_incoming(source=source)

    def visualize_graph(self, seed:int=1234):
        """method to visualize the graph as a set of nodes and edges

        Returns:
            matplotlib.pyplot.Axes: the axes containing a visual representation of the graph
        """
        
        np.random.seed(seed)
        
        # Define the graph using an adjacency list
        graph = { name: node.get_outgoing() for name, node in self.graph.items() }

        # Calculate node positions (you can use a simple layout here)
        pos = { name: np.random.uniform(0, 1, 2) for name,_ in self.graph.items() }

        # set the figure
        fig, ax = plt.subplots(1,1,figsize=(10,10))
        
        # Draw the nodes
        for node in graph:
            ax.scatter(pos[node][0], pos[node][1], s=100, c="blue" if type(self.graph[node]) is self.CategoricalNode else "red")
            ax.text(pos[node][0], pos[node][1], str(node), fontsize=12)

        # Draw the edges
        for node, neighbors in graph.items():
            for neighbor in neighbors:
                ax.arrow(pos[node][0], pos[node][1], pos[neighbor][0]-pos[node][0], pos[neighbor][1]-pos[node][1], shape='full', lw=0, length_includes_head=True, head_width=.01, color="black")

        # remove some unnecessary formatting
        ax.set_xticks([])
        ax.set_yticks([])

        return ax

    def load_nodes_from_dataframe(self, dataframe:pd.DataFrame):
        """loads nodes from a given dataframe

        Args:
            dataframe (pd.DataFrame): the dataframe to load nodes from

        Returns:
            None: stores the data in graph format
        """
        
        # basic inference of the type of data for easy automated loading
        def infer_storage_type(values:pd.Series):
            if values.dtype == "float64" or values.dtype == "int64":
                return "cont"
            else:
                return "cat"

        # add a node for each column
        for col in dataframe.columns:
            st = infer_storage_type(dataframe[col])
            
            if st == "cat":
                self.add_node(col, storage_type=st, values=dataframe[col].unique().tolist(), probs=dataframe[col].value_counts().values/len(dataframe))
            else:
                self.add_node(col, storage_type=st, mean=dataframe[col].mean(), standard_deviation=dataframe[col].std())

    def infer_edges(self):
        pass

    def query(self, node:str, value, operation:str="equals"):
        """with base value, calculate the probability of a particular node

        Args:
            node (str): the node to calculate probability for
            value (str, int, float): the value to check probability for
            operation (str, optional): if value is numerical, the operation to check probability for. Defaults to "equals".

        Returns:
            float: the probability of the particular value at the given node
        """
        
        # if the node is categorical, just return the probability of that value
        if type(self.graph[node]) is self.CategoricalNode and type(value) is str:
            return self.graph[node].get_probs(value)
        # if it's continuous, return the probability of the operation to the value
        elif type(self.graph[node]) is self.ContinuousNode and type(value) is not str:
            return self.graph[node].get_probs(value, operation)

    def conditional_query(self, node:str, value, operation:str="equals", conditional_nodes:list=None, conditional_values:list=None):
        
        # if there are no conditions, this reverts to regular inference
        if conditional_nodes is None:
            print("You provided no conditional nodes, defaulting to normal inference")
            return self.inference(node, value, operation)
        
        # Bayes' Rule with multiple conditions
        #if type(node) is self.CategoricalNode:
        num = np.prod(np.array([self.graph[node].get_probs(value) * self.graph[cn].get_probs(cv) for cn,cv in zip(conditional_nodes, conditional_values)] + [self.graph[node].get_probs(value)**(len(conditional_nodes)+1)]))
        den = np.sum(np.array([
            np.prod(np.array([self.graph[node].get_probs(value) * self.graph[cn].get_probs(cv) for cn,cv in zip(conditional_nodes, conditional_values)] + [self.graph[node].get_probs(value)**(len(conditional_nodes)+1)])),
            np.prod(np.array([(1 - self.graph[node].get_probs(value)) * self.graph[cn].get_probs(cv) for cn,cv in zip(conditional_nodes, conditional_values)] + [(1 - self.graph[node].get_probs(value))**(len(conditional_nodes)+1)]))
        ]))
        return num / den

    class CategoricalNode:
        
        def __init__(self, name:str, values:list, probs:list):
            
            # define properties
            self.name = name
            
            # define the probabilities
            self.table = {v: p for v,p in zip(values, probs)}
            
            # edges
            self.incoming = []
            self.outgoing = []
            
        def __str__(self):
            """overrides the default __str__ method and returns a string representation of the Node
            
            Args:
                None
                
            Returns:
                str: the Node as a string
            """
            
            return f"{self.name}\tCategorical\t{self.table}\tOutoing Edges={self.outgoing}"
        
        def add_incoming(self, source:str): self.incoming.append(source)
        def add_outgoing(self, destination:str): self.outgoing.append(destination)
            
        def get_incoming(self): return self.incoming
        def get_outgoing(self): return self.outgoing
            
        def get_name(self): return self.name
        
        def get_probs(self, value:str): return self.table[value]
        
    class ContinuousNode:
        
        def __init__(self, name:str, mean:float, standard_deviation:float):
            
            # define properties
            self.name = name
            
            # define the probabilities
            self.mean = mean
            self.sd = standard_deviation
            
            # edges
            self.incoming = []
            self.outgoing = []
            
        def __str__(self):
            """overrides the default __str__ method and returns a string representation of the Node
            
            Args:
                None
                
            Returns:
                str: the Node as a string
            """
            
            return f"{self.name}\tContinuous\t\u007bmean: {self.mean}, sd:{self.sd}\u007d\tOutgoing Edges={self.outgoing}"
        
        def add_incoming(self, source:str): self.incoming.append(source)
        def add_outgoing(self, destination:str): self.outgoing.append(destination)
            
        def get_incoming(self): return self.incoming
        def get_outgoing(self): return self.outgoing
            
        def get_name(self): return self.name
        
        def get_probs(self, value:float, operation:Literal["equals", "greater", "less"]):
            """computes probability of a particular continuous value

            Args:
                value (float): the value to check probability for
                operation (Literal["equals", "greater", "less"]): the operation to check the probability for

            Returns:
                float: the probability of the operation for the input value
            """
            
            if operation == "equals":
                z = (value - self.mean) / self.sd
                return stats.norm.cdf(z+0.005) - stats.norm.cdf(z-0.005)
            elif operation == "greater":
                z = (value - self.mean) / self.sd
                return 1 - stats.norm.cdf(z)
            elif operation == "less":
                z = (value - self.mean) / self.sd
                return stats.norm.cdf(z)
    