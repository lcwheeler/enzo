from .pathway import Pathway, PathwayMod, PathwayFlex
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
        
    def generate(self, model_string, num_pathways, running_pickle=False):
        """Generate a set of Pathway() objects.

        Parameters

        ----------

        model_string: string
            antimony language model string
        num_pathways: int
            number of PathwayFlex objects to generate
        running_pickle: bool
            whether the object is pickled after evolution of each Pathway object
        """

        self.running_pickle = running_pickle
        
        if bool(self.pathway_set) == False:
            ref_model = Pathway(model_string, name = "ref")

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

                # This will write out the PathwaySetFlex object as a pickle file after each iteration
                # Need to convert te model object to sbml str for pickling and then convert back
                if self.running_pickle == True:
                    temp_model = copy(self.pathway_set[i].model)
                    self.pathway_set[i].model = self.pathway_set[i].model.getCurrentSBML()
                    pickle.dump(self, open(self.name+".p", "wb"))
                    self.pathway_set[i].model = temp_model
            
            
            self.evolved = True
            
        else:
            raise Exception('The Pathways in this set have already been evolved.')
            
    def collate_data(self):
        """Organize the selection coefficient and mut. effect size data from individual Pathway objects into a single dataset."""
        
        s_dict = {}
        delta_dict = {}

        for key in self.params:
            s_dict[key] = []
            delta_dict[key] = []

        for v in self.pathway_set.values():
            for key in v.fitness_effects.keys():
                for a in v.fitness_effects[key]:
                    s_dict[key].append(a)

        for v in self.pathway_set.values():
            for key in v.delta_effects.keys():
                for a in v.delta_effects[key]:
                    delta_dict[key].append(a)

        self.fitness_effects = s_dict
        self.delta_dict = delta_dict
        
    def save_pathways(self):
        """Save each individual Pathway object as a pickle file."""
        for key in self.pathway_set.keys():
            pickle.dump(self.pathway_set[key], open(str(self.pathway_set[key].name)+".p"), "wb")

            
    def save_set(self):
        """Save the entire PathwaySet object instance as a pickled file."""
        pickle.dump(self, open(self.name+".p", "wb"))




#######################################################################################################
####### Modified version of the PathwaySet object that incorporates constraint into fitness function

class PathwaySetMod(object):
    """Object class that holds a set of Pathway() objects, with methods to access data from all objects."""
    
    def __init__(self, name):
        """Initialize an instance of the PathwaySet() object."""
        self.pathway_set = {}
        self.evolved = False
        self.name = name
        
    def generate(self, model_string, num_pathways, running_pickle=False):
        """Generate a set of PathwayMod() objects.

        Parameters

        ----------

        model_string: string
            antimony language model string
        num_pathways: int
            number of PathwayMod objects to generate
        running_pickle: bool
            whether the object is pickled after evolution of each PathwayFlex object
        """

        self.running_pickle = running_pickle
        
        if bool(self.pathway_set) == False:
            ref_model = PathwayMod(model_string, name = "ref")

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
            optimum ratio of total concentration to starting total concentration
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
                # This will write out the PathwaySetFlex object as a pickle file after each iteration
                if self.running_pickle == True:
                    pickle.dump(self, open(self.name+".p", "wb"))

            self.evolved = True
            
        else:
            raise Exception('The Pathways in this set have already been evolved.')
            
    def collate_data(self):
        """Organize the selection coefficient and mut. effect size data from individual Pathway objects into a single dataset."""
        
        s_dict = {}
        delta_dict = {}

        for key in self.params:
            s_dict[key] = []
            delta_dict[key] = []

        for v in self.pathway_set.values():
            for key in v.fitness_effects.keys():
                for a in v.fitness_effects[key]:
                    s_dict[key].append(a)

        for v in self.pathway_set.values():
            for key in v.delta_effects.keys():
                for a in v.delta_effects[key]:
                    delta_dict[key].append(a)

        self.fitness_effects = s_dict
        self.delta_dict = delta_dict
        
    def save_pathways(self):
        """Save each individual Pathway object as a pickle file."""
        for key in self.pathway_set.keys():
            pickle.dump(self.pathway_set[key], open(str(self.pathway_set[key].name)+".p"), "wb")

            
    def save_set(self):
        """Save the entire PathwaySet object instance as a pickled file."""
        pickle.dump(self, open(self.name+".p", "wb"))


#######################################################################################################
####### Modified version of the PathwaySet object that corresponds to the PathwayFlex object with customizable fitness func.

class PathwaySetFlex(object):
    """Object class that holds a set of Pathway() objects, with methods to access data from all objects."""
    
    def __init__(self, name):
        """Initialize an instance of the PathwaySet() object."""
        self.pathway_set = {}
        self.evolved = False
        self.name = name
        
    def generate(self, model_string, num_pathways, running_pickle=False):
        """Generate a set of PathwayFlex() objects.

        Parameters

        ----------

        model_string: string
            antimony language model string
        num_pathways: int
            number of PathwayFlex objects to generate
        running_pickle: bool
            whether the object is pickled after evolution of each PathwayFlex object
        """

        self.running_pickle = running_pickle

        if bool(self.pathway_set) == False:
            ref_model = PathwayFlex(model_string, name = "ref")

            for i in range(num_pathways):
                self.pathway_set[i] = copy(ref_model)
                self.pathway_set[i].name = i
        else:
            raise Exception('A pathway set has already been generated.')
            
    def evolve(self, params, W_func, W_func_args, mutation_func, mutation_func_args, Pfix_func, Pfix_func_args, direct_assign_mutations, optimum_tolerance, iterations, stop, MCA): 

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
            optimum ratio of total concentration to starting total concentration
        optimum_tolerance: float
            fractional tolerance on optimum SS concentration of target
        iterations: int  
            number of iterations to perform (max length of trajectory)
        stop: bool
            Whether or not to stop simulation when optimum is reached
        """

        # Keep track of the input values by making them attributes of the Pathway
        self.W_func = W_func
        
        if self.evolved == False: 
            self.params = params

            for i in range(len(self.pathway_set.keys())):
                self.pathway_set[i].evolve(params, W_func, W_func_args, mutation_func, mutation_func_args, Pfix_func, Pfix_func_args, direct_assign_mutations, optimum_tolerance, iterations, stop, MCA)

                # This will write out the PathwaySetFlex object as a pickle file after each iteration
                # Need to convert te model object to sbml str for pickling and then convert back
                if self.running_pickle == True:
                    temp_model = copy(self.pathwayset[i].model)
                    self.pathway_set[i].model = self.pathway_set[i].model.getCurrentSBML()
                    pickle.dump(self, open(self.name+".p", "wb"))
                    self.pathway_set[i].model = temp_model
            
            self.evolved = True
            
        else:
            raise Exception('The Pathways in this set have already been evolved.')
            
    def collate_data(self):
        """Organize the selection coefficient and mut. effect size data from individual Pathway objects into a single dataset."""
        
        s_dict = {}
        delta_dict = {}

        for key in self.params:
            s_dict[key] = []
            delta_dict[key] = []

        for v in self.pathway_set.values():
            for key in v.fitness_effects.keys():
                for a in v.fitness_effects[key]:
                    s_dict[key].append(a)

        for v in self.pathway_set.values():
            for key in v.delta_effects.keys():
                for a in v.delta_effects[key]:
                    delta_dict[key].append(a)

        self.fitness_effects = s_dict
        self.delta_dict = delta_dict
        
    def save_pathways(self):
        """Save each individual Pathway object as a pickle file."""
        for key in self.pathway_set.keys():
            pickle.dump(self.pathway_set[key], open(str(self.pathway_set[key].name)+".p"), "wb")

            
    def save_set(self):
        """Save the entire PathwaySet object instance as a pickled file."""
        pickle.dump(self, open(self.name+".p", "wb"))
