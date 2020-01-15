from .pathway import Pathway, FlexFitPathway
from copy import copy
import matplotlib.pyplot as plt
import pickle


class PathwaySet(object):
    """Object class that holds a set of Pathway() objects, with methods to access data from all objects."""
    
    def __init__(self, name):
        """Initialize an instance of the PathwaySet() object."""
        self.pathway_set = {}
        self.evolved = False
        self.name = name
        
    def generate(self, model_string, num_pathways):
        """Generate a set of Pathway() objects."""
        
        if bool(self.pathway_set) == False:
            ref_model = FlexFitPathway(model_string, name = "ref")

            for i in range(num_pathways):
                self.pathway_set[i] = copy(ref_model)
                self.pathway_set[i].name = i
        else:
            raise Exception('A pathway set has already been generated.')
            
    def evolve(self, params, optimum1, numerator_indexes, denominator_indexes, total, constraint, optimum_tolerance, iterations):

        """Evolve all of the pathway objects stored in pathway_set. Using the built-in evolve function in Pathway.

        Parameters

        ----------

        params: 1D array-like
            list of parameter keys
        optimum1: int
            optimum ratio of target species to sum of all other species
        target_index: int
            index of the target species in list of floating species IDS
        constraint: float
            fractional tolerance on total SS concentration
        optimum_tolerance: float
            fractional tolerance on optimum SS concentration of target
        iterations: int  
            number of iterations to perform (max length of trajectory)
        stop: bool
            Whether or not to stop simulation when optimum is reached
        """

        # Keep track of the input values by making them attributes of the Pathway
        self.constraint = constraint
        self.numerator_indexes = numerator_indexes
        self.denominator_indexes = denominator_indexes
        self.optimum = optimum1
        self.total = total
        
        if self.evolved == False: 
            self.params = params

            for i in range(len(self.pathway_set.keys())):
                self.pathway_set[i].evolve(params, optimum1, numerator_indexes, denominator_indexes, total, constraint, optimum_tolerance, iterations)
            
            self.evolved = True
            
        else:
            raise Exception('The Pathways in this set have already been evolved.')
            
    def collate_data(self):
        """Organize the selection coefficient data from individual Pathway objects into a single dataset."""
        
        s_dict = {}
        for key in self.params:
            s_dict[key] = []
            
        for v in self.pathway_set.values():
            for key in v.fitness_effects.keys():
                for a in v.fitness_effects[key]:
                    s_dict[key].append(a)
                    
        self.fitness_effects = s_dict

        
    def save_pathways(self):
        """Save each individual Pathway object as a pickle file."""
        for key in self.pathway_set.keys():
            pickle.dump(self.pathway_set[key], open(str(self.pathway_set[key].name)+".p"), "wb")

            
    def save_set(self):
        """Save the entire PathwaySet object instance as a pickled file."""
        pickle.dump(self, open(self.name+".p", "wb"))
