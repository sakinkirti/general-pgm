import numpy as np
import matplotlib.pyplot as plt
from typing import Literal

class Graph:
    
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
            res += f"{v.__str__()}: {v.get_outgoing()}\n"
            
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

        return ax

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
            
            return f"Node(name:{self.name}, values:{list(self.table.keys())})"
        
        def add_incoming(self, source:str): self.incoming.append(source)
        def add_outgoing(self, destination:str): self.outgoing.append(destination)
            
        def get_incoming(self): return self.incoming
        def get_outgoing(self): return self.outgoing
            
        def get_name(self): return self.name
        
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
            
            return f"Node(name:{self.name}, mean:{self.mean}, sd:{self.sd})"
        
        def add_incoming(self, source:str): self.incoming.append(source)
        def add_outgoing(self, destination:str): self.outgoing.append(destination)
            
        def get_incoming(self): return self.incoming
        def get_outgoing(self): return self.outgoing
            
        def get_name(self): return self.name
    