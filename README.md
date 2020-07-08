# enzo
*enzo* is a Python wrapper for Tellurium that allows evolution of metabolic models between defined phenotypic states.
Please use enzo in a Python 3.6 environment, to ensure compatibility with the Tellurium dependency. This package is 
a work in progress. It is neither as fast nor as generalizable as I would like it to be, but I am working toward 
improvements to make it more flexible. The main reason I have made *enzo* available is to ensure that the work I 
have used it for can be readily reproduced after installing the package. However, feel free to use *enzo* for 
other projects at your own peril! Currently there is no Documentation page, but I will work on that 
after I have solidified the changes I'm currently making to allow more flexibility in user-input fitness, mutation, and
fixation probability functions. If there are any issues, please let me know by filing an issue report on github. 
Pull requests with improvements are welcome too! 

Example:

This example re-runs the stochastic evolutionary simulations performed in our paper [Computational modeling of anthocyanin pathway evolution: Biases, hotspots, and trade-offs](https://www.biorxiv.org/content/early/2019/01/03/511089). 


```python
from enzo import PathwaySet
import tellurium as te

# Create a model string to generate models downstream
model_string = """
    
    # Constant source entering model
    Source: => PCoA;
    
    # Rate laws for all the central enzymatic reactions
    
    CHS: PCoA => cha; (k_CHS_PCoA*CHSt*PCoA)/(Km_CHS_PCoA + PCoA);
    
    CHI: cha => nar; (k_CHI_cha*CHIt*cha)/(Km_CHI_cha + cha);
    

    F3H_nar: nar => DHK; (k_F3H_nar*F3Ht*nar)/(Km_F3H_nar + nar);
    
    
    F3pH_DHK: DHK => DHQ; (k_F3pH_DHK*F3pHt*DHK)/(Km_F3pH_DHK + DHK);
    
    
    F35H_DHQ: DHQ => DHM; (k_F35H_DHQ*F35Ht*DHQ)/(Km_F35H_DHQ + DHQ);
    
    
    FLS_DHK: DHK => kam; (k_FLS_DHK*FLSt*DHK)/(Km_FLS_DHK * (1 + DHQ/Km_FLS_DHQ + DHM/Km_FLS_DHM) + DHK);
    
    FLS_DHQ: DHQ => que; (k_FLS_DHQ*FLSt*DHQ)/(Km_FLS_DHQ * (1 + DHM/Km_FLS_DHM + DHK/Km_FLS_DHK) + DHQ);
    
    FLS_DHM: DHM => myr; (k_FLS_DHM*FLSt*DHM)/(Km_FLS_DHM * (1 + DHK/Km_FLS_DHK + DHQ/Km_FLS_DHQ) + DHM);
    
    
    DFR_DHK: DHK => LCP; (k_DFR_DHK*DFRt*DHK)/(Km_DFR_DHK * (1 + DHQ/Km_DFR_DHQ + DHM/Km_DFR_DHM) + DHK);
    
    DFR_DHQ: DHQ => LCC; (k_DFR_DHQ*DFRt*DHQ)/(Km_DFR_DHQ * (1 + DHK/Km_DFR_DHK + DHM/Km_DFR_DHM) + DHQ);
    
    DFR_DHM: DHM => LCD; (k_DFR_DHM*DFRt*DHM)/(Km_DFR_DHM * (1 + DHK/Km_DFR_DHK + DHQ/Km_DFR_DHQ) + DHM);
    
    
    ANS_LCP: LCP => pel; (k_ANS_LCP*ANSt*LCP)/(Km_ANS_LCP * (1 + LCC/Km_ANS_LCC + LCD/Km_ANS_LCD) + LCP);
    
    ANS_LCC: LCC => cya; (k_ANS_LCC*ANSt*LCC)/(Km_ANS_LCC * (1 + LCP/Km_ANS_LCP + LCD/Km_ANS_LCD) + LCC);
    
    ANS_LCD: LCD => del; (k_ANS_LCD*ANSt*LCD)/(Km_ANS_LCD * (1 + LCP/Km_ANS_LCP + LCC/Km_ANS_LCC) + LCD);
    

    # Product sinks
    
    pel_sink: pel =>; k_pel_sink*pel;
    
    cya_sink: cya =>; k_cya_sink*cya;
    
    del_sink: del =>; k_del_sink*del;
    

    kam_sink: kam =>; k_kam_sink*kam;
    
    que_sink: que =>; k_que_sink*que;
    
    myr_sink: myr =>; k_myr_sink*myr;
    
        
    # Substrate Kcat's
    k_CHS_PCoA=14; k_CHI_cha=14; k_F3H_nar=14; 
    k_F3pH_DHK=14; k_F35H_DHQ=14; k_FLS_DHK=14; 
    k_FLS_DHQ=14; k_FLS_DHM=14; k_DFR_DHK=14; k_DFR_DHQ=14; k_DFR_DHM=14; k_ANS_LCP=14; k_ANS_LCC=14;
    k_ANS_LCD=14; 
   
    # Substrate Km's
    Km_CHS_PCoA=0.013; Km_CHI_cha=0.013; 
    Km_F3H_nar=0.013; Km_F3pH_DHK=0.013; Km_F35H_DHQ=0.013; 
    Km_FLS_DHK=0.013; Km_FLS_DHQ=0.013; Km_FLS_DHM=0.013; 
    Km_DFR_DHK=0.013; Km_DFR_DHQ=0.013; Km_DFR_DHM=0.013; 
    Km_ANS_LCP=0.013; Km_ANS_LCC=0.013; Km_ANS_LCD=0.013;


    # Enzyme concentrations
    CHSt=0.001; CHIt=0.001; F3Ht=0.001; F3pHt=0.001;
    F35Ht=0.001; FLSt=0.001; DFRt=0.001; ANSt=0.001; 

    
    # Rates for sinks 
    k_pel_sink=0.0005; k_cya_sink=0.0005; k_del_sink=0.0005; 
    k_kam_sink=0.0005; k_que_sink=0.0005; k_myr_sink=0.0005; 
    
    # Source influx
    const PCoA = 0.01;
"""


# Generate a reference model using the model string
r = te.loada(model_string)

# Construct a list of evolvable parameters for the evolutionary simulation (exclude sink rates)
param = r.getGlobalParameterIds()
param_trim = []
for n in param:
    if "sink" not in n:
        param_trim.append(n)
        
    else:
        pass

# Generate a PathwaySet object (containing an ensemble of Pathway objects) to evolve
evolving_set = PathwaySet("example_simulations")
evolving_set.generate(model_string, 10000)

# Evolve the PathwaySet ensemble toward a 90% delphinidin optimum
evolving_set.evolve(params=param_trim, optimum1=0.9, numerator_indexes = [13], denominator_indexes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13], total = 12.2, constraint = 0.1, optimum_tolerance = 0.1, iterations = 50000)

# Save the full PathwaySet with all data as a single pickled object
evolving_set.save_set()


```


## Install

Clone this repository and install a development version using `pip`:
```
pip install -e .
```

