import matplotlib.pyplot as plt
import numpy as np
import tellurium as te
import pickle
import pandas as pd
import seaborn as sns
import roadrunner
from copy import copy, deepcopy


class Pathway(object):
    """A Model object class that contains methods for evolving a Roadrunner/Tellurium model between defined states."""

    def __init__(self, model_string, name):
        """Initialize an instance of the Model() class.

        Parameters

        ----------

        model_string: str
            antimony string describing model equations
        name: str
            a string to name the object instance

        """
        
        # Initialize some important object attributes
        self.model_string = model_string
        self.main_model = te.loada(model_string)
        self.name = name

    def evolve(self, params, optimum1, numerator_indexes, denominator_indexes, total, constraint = 0.1, optimum_tolerance = 0.01, iterations=10000, stop=True):

        """Function to sample the parameter space of the model one parameter at a time and evolve toward a 
        user defined fitness optimum. Fitness and fixation probabilities are calculated assuming stabilizing 
        selection on steady state concentration of species.
        
        Parameters

        ----------

        params: 1D array-like
            list of parameter key strings
        optimum1: int
            optimum ratio of sum of target species to sum of denominator species
        numerator_indexes: list
            List of indexes of the target species in array of floating species IDs
        denominator_indexes: list
            List of indexes of the remaining species in array of floating species IDs
        total: float
            total desired steady state concentration
        constraint: float
            fractional tolerance on total SS concentration
        optimum_tolerance: float
            fractional tolerance on optimum value
        iterations: int  
            number of iterations to perform (max length of trajectory)
        stop: bool
            Whether or not to stop simulation when optimum is reached
            
        """

        # Store the numerator and denominator indexes for fitness calculations
        self.numerator_indexes = numerator_indexes
        self.denominator_indexes = denominator_indexes
        
        
        # Initialize some lists and dictionaries to hold the simulation trajectories
        parameters = [] #parameters[("ID", "param_value", "sel_coeff", "prob_fix")] = []
        concentrations = []
        optima = {}
        
        # Build a dictionary to hold the fitness effects of each mutation (as list of effects for each parameter).
        fitness_effects = {}
        for key in params:
            fitness_effects[key] = []
        
        main_model = self.main_model
        species = list(main_model.getFloatingSpeciesIds())
        

        # Reset the main reference model (start) and make a copy to pass into the iterations
        main_model.resetToOrigin()
        model = copy(main_model)

        # Calculate and append the initial steady state concentration data to the concentrations list

        try:
            main_model.getSteadyStateSolver().relative_tolerance = 1e-25
            SS_selections_main = main_model.steadyStateSelections = species
            SS_values_main = main_model.getSteadyStateValues()
            concentrations.append(SS_values_main)
            main_model.resetToOrigin()

        except RuntimeError as err:
            self.main_model_error = str(err)
            concentrations.append(list(np.zeros(len(species))))
            main_model.resetToOrigin()

        # Changes suggested by Kiri Choi to turn off steady state approximation
        ss = model.getSteadyStateSolver()
        ss.allow_approx = False
        ss.allow_presimulation = False
        
        # Turn off roadrunner logging to save memory
        roadrunner.Logger_disableLogging() 
        
        # Generate a random set of mutations
        np.random.seed() # Need to store the seed used as attribute for reproducibility. 
        self.mutations = np.random.gamma(0.8, 3, iterations+1)
        mutations = self.mutations
        
        IDs = np.random.choice(params, iterations+1)
        
        # Initialize tracking variables so the model can be reset in the event of error
        self.last_val = getattr(model, IDs[0]) 
        self.last_ID = IDs[0] 
        
        # Initialize a list of fixation choices for downstream: 0 = fix, 1 = don't fix
        fixation_choices = [0, 1]
        
        # Initialize a counter to keep track of the arrival time for each fixation (i.e. number of attempts)
        arrival = 0

        # Iterate over generations of selection on steady state concentrations
        for i in range(iterations):
            model.reset()
            
            # Add 1 to the arrival time counter
            arrival += 1

            # Calculate the current fitness before changing parameter values
            model.getSteadyStateSolver().relative_tolerance = 1e-25
            
            # Get the steady state selections (i.e. floating species)
            SS_selections = model.steadyStateSelections = species
            
            # Try to solve the current model steady state and reset to previous state if it fails
            try:
                SS_values_current = model.getSteadyStateValues()
                SS_values_current = np.array(SS_values_current)
                
            except RuntimeError as e:
                model.setValue(id=self.last_ID, value=self.last_val)
                model.reset()
                print(i, e)
                continue
            
            # Calculate the distance of the current model SS from optimum
            metric_current_1 = np.sum(SS_values_current[numerator_indexes])/np.sum(SS_values_current[denominator_indexes]) 
            
            # Calculate the current fitness as the negative exp of distance from optimum
            W_current = np.exp(-1*(metric_current_1 - optimum1)**2)
            
            # Choose a random parameter from the model and find the current value
            ID = IDs[i]
            val = getattr(model, ID)       
            
            # Track the mutation parameter and effect in case the model requires resetting
            self.last_val = deepcopy(val)
            self.last_ID = deepcopy(ID)
            
            # Use random parameter adjustments about the starting value (from pre-assembled lists)
            value = val * mutations[i]
            model.setValue(id=ID, value=value)
            delta = (value - val)/val # for book-keeping

            # Calculate the steady state concentrations of species in pathway
            model.getSteadyStateSolver().relative_tolerance = 1e-25

            # If the steady state solver fails, just reset to previous state and skip the iteration
            try:
                SS_values = model.getSteadyStateValues()
                SS_values = list(SS_values)
                SS_values = np.array(SS_values)

            except RuntimeError: 
                model.setValue(id=ID, value=val)
                model.reset()
                continue     

            # Add the results and the param values to their respective DataFrames if they satisfy criteria
            if all(i > 0 for i in SS_values) and np.sum(SS_values) > (total - constraint*total) and np.sum(SS_values) < (total + constraint*total):

                # This allows selection on a ratio of any set of floating species to any other set of floating species
                metric_1 = np.sum(SS_values[self.numerator_indexes])/np.sum(SS_values[self.denominator_indexes])  
                
                # Calculate the mutant fitness, relative fitness, and selection coefficient (s)
                W = np.exp(-1*(metric_1 - optimum1)**2)  
                
                # Keep track of fitness effects at each step (positive = good, negative = bad). 
                # This is just a delta_W between previous state and new mutant state
                s = W - W_current

                fitness_effects[ID].append(s)
                
                # Check if the mutant fitness is improved over previous state and calculate fixation 
                # probability accordingly. Discard neutral and deleterious mutations (because Pfix ~ 0). 
                if W >= W_current: 

                    P = (1-np.exp(-s))
                    f = np.random.choice(fixation_choices, p = [P, 1-P])

                    if f == 0:
                        # Store the steady state concentrations at this step.
                        concentrations.append(SS_values)

                        # Store the parameter ID, value, selection coefficient, delta mut. effect size, metric_1, fitness,
                        # and arrival time of mutation in a dict, add to the parameters attribute (list of dicts).
                        parameters.append({"ID": ID, "value": value, "s":s, "P_fix":P, "delta":delta, "arrival":arrival, "distance":metric_1, "fitness":W}) 

                        # reset the arrival time to 0 to begin count again until next fixation event
                        arrival = 0

                        # Check to see if the steady state is within tolerance of the optimum and save if it is.
                        if metric_1 > (optimum1 - optimum1*optimum_tolerance) and metric_1 < (optimum1 + optimum1*optimum_tolerance):
                            model.reset()
                            optima[i] = [tuple(SS_values), tuple(model.getGlobalParameterValues())]

                            if stop == True: 
                                break 
                    else:
                        model.setValue(id=ID, value=val)
                        model.reset()

                else: 
                    model.setValue(id=ID, value=val)
                    model.reset()

            else:
                model.setValue(id=ID, value=val)
                model.reset()
                continue
                
        # Store the concentrations from each step as a DataFrame
        concentrations = pd.DataFrame(concentrations, columns=species)

        # This step overwrites the model by creating a new replicate model and replacing parameters with 
        # the evolved parameter set. Circumvents potential issues with saved initial concentrations, etc. 
        gp = model.getGlobalParameterValues()
        reset_model = te.loada(self.model_string)
        reset_model.setValues(keys=params, values=gp)
        reset_model.reset()
        self.main_model.reset()
        
        # Clear the unneccessary model string input
        self.model_string = None
        
        # Builds a pd.DataFrame of the control coefficient and elasticity matrices. 
        # If it fails for some numerical (or other) reason, just ignores and saves the error message instead.
        # These can be checked later to discard any models that result in errors. 
        try:
            self.cc_matrix = model.getScaledConcentrationControlCoefficientMatrix()
            self.cc_matrix = pd.DataFrame(self.cc_matrix, columns=[name for name in self.cc_matrix.colnames], index=[name for name in self.cc_matrix.rownames])
        
            self.elasticities = model.getScaledElasticityMatrix()
            self.elasticities = pd.DataFrame(self.elasticities, columns=[name for name in self.elasticities.colnames], index=[name for name in self.elasticities.rownames])
        
        except RuntimeError as e:
            self.cc_matrix = None
            self.elasticities = None
            self.mca_error = str(e)
            
        # Assign the evolver output to attributes of the Pathway object 
        self.concentrations = concentrations 
        self.parameters = parameters
        self.main_model = self.main_model.getCurrentSBML()
        self.model = reset_model.getCurrentSBML()
        self.optima = optima
        self.fitness_effects = fitness_effects
        
    def plot_ss(self):
        """Return a plot of the steady state concentrations for the final evolved model."""
        
        model = te.loads(self.model)
        model.reset()
        SS_selections = model.steadyStateSelections = list(model.getFloatingSpeciesIds()) 
        SS_values = model.getSteadyStateValues()
        SS_values = np.array(SS_values)

        sns.barplot(SS_selections, SS_values, color="lightgray")
        plt.xticks(rotation=90)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.title("Steady-state (irreversible, competition)")
        plt.ylabel("Concentration (mM)")
        plt.xlabel("Metabolite name");

        print("The total concentration is " + str(np.sum(SS_values)) + "mM")
        print("The ratio of the target species to all others species is "+ str(np.sum(SS_values[self.numerator_indexes])/np.sum(SS_values[self.denominator_indexes])))

    def save_data(self):
        """Function to save the simulated data as compressed pickle files."""
        
        self.concentrations.to_pickle(self.name+"_concentrations.p")
        self.cc_matrix.to_pickle(self.name+"_cc_matrix.p")
        self.elasticities.to_pickle(self.name+"_elasticities_matrix.p")
        pickle.dump(self.parameters, open(self.name+"_parameters.p", "wb"))
        pickle.dump(self.fitness_effects, open(self.name+"_fitness_effects.p", "wb"))
        pickle.dump(self.sbml, open(self.name+"_model_sbml.p", "wb"))
