'''

'''
from cobra.io.web import load_model
from cobra.util.array import create_stoichiometric_matrix, constraint_matrices
from typing import Dict, List, Optional, Tuple 

class DynamicFBASolver:
    '''
    Solver interface for COBRA model and FBA solving. 
    Used as a parameter for the SpatialSimulator class. 

    Units: 
    -- Fluxes (Cobra): mmol / gDW / hr 
    -- Concentrations: mmol / L 
    -- Biomass: gDW 
    -- Volume: L 
    -- dt: seconds 
    ''' 
    def __init__(self, 
        model_name: str = 'textbook',  
        vmax_params: Optional[Dict[str, float]] = None, 
        km_params: Optional[Dict[str, float]] = None, 
        kn_params: Optional[Dict[str, float]] = None, 
        ext_conc: Optional[Dict[str, float]] = None, 
        solver_backend='gurobi', 
        essential_exchanges: Optional[List[str]] = None, 
        BIOMASS_RXN='Biomass_Ecoli_core'
    ):  
        self.model = load_model(model_name)
        self.model.solver = solver_backend
        self.vmax_params = vmax_params or {}
        self.km_params = km_params or {}
        self.kn_params = kn_params or {}
        self.ext_conc = ext_conc or {}

        self._init_missing_metabolites()
        self.exchange_reactions_map = self._get_exchange_map()

        self.essential_exchanges = essential_exchanges
        if not self.essential_exchanges: 
            self.essential_exchanges = ["EX_nh4_e","EX_pi_e","EX_h_e","EX_h2o_e",
                    "EX_k_e","EX_na1_e","EX_cl_e","EX_mg2_e","EX_ca2_e","EX_fe2_e"]
        self._apply_essential_bounds()
        self.BIOMASS_RXN = BIOMASS_RXN
        self.solution_feasible = True

    def _init_missing_metabolites(self) -> None: 
        '''
        For any exchange reactions whose metabolites 
        are not tracked in ext_conc, set that initial 
        value to zero. 
        '''
        for rxn in self.model.exchanges: 
            for metabolite in rxn.metabolites: 
                if metabolite.id not in self.ext_conc: 
                    self.ext_conc[metabolite.id] = 0.0 

    def _get_exchange_map(self) -> Dict[str, str]: 
        '''
        Create dictionary whose keys are 
        metabolite IDs, and values are reaction IDs,
        for all exchange reactions. 
        '''
        ex_reactions_map = {}
        for reaction in self.model.exchanges: 
            metabolites = list(reaction.metabolites.keys())
            if len(metabolites) == 1: 
                if metabolites[0].compartment == 'e': 
                    ex_reactions_map[metabolites[0].id] = reaction.id 
        return ex_reactions_map

    def _apply_essential_bounds(self) -> None: 
        '''
        For all essential reactions, set 
        conservative bounds of [-1000.0, 1000.0]. 
        '''
        for rxn_id in self.essential_exchanges:
            try: 
                r = self.model.reactions.get_by_id(rxn_id)
                r.lower_bound = -1000.0
                r.upper_bound = 1000.0
            except KeyError: 
                pass 

    def _set_dynamic_bounds(self) -> float: 
        '''
        1. Update exchange bounds based on external concentrations. 
        The formula is based on Michaelis-Menten kinetics. 
        2. Compute a multiplicative biomass inhibition factor based on 
        concentration of inhibitory products, which are specified in 
        self.kn_params. 
        '''
        # 1. Compute update limits and udate reaction lower bounds accordingly 
        for metabolite_id, reaction_id in self.exchange_reactions_map.items(): 
            if metabolite_id not in self.ext_conc: 
                continue 
            reaction_current = self.model.reactions.get_by_id(reaction_id)
            conc = max(self.ext_conc[metabolite_id], 0.0)
            V_max = self.vmax_params.get(reaction_id, 10.0) # Default Vmax of 10
            Km = self.km_params.get(reaction_id, 0.01) # Default Km of 0.01 

            # Michaelis-Menten uptake lower bound 
            uptake_limit = 0.0 
            if conc > 0: 
                uptake_limit = -1.0 * V_max * conc / (Km + conc)

            reaction_current.lower_bound = uptake_limit 

        # 2. Compute inhibitory factors for biomass creation 
        biomass_inhibition_product_term = 1.0 
        for reaction_id, kn_value in self.kn_params.items(): 
            rxn_current = self.model.reactions.get_by_id(reaction_id)
            for metabolite in rxn_current.metabolites: 
                metabolite_id = metabolite.id 
                if metabolite_id.endswith('_e') and metabolite_id in self.ext_conc: 
                    conc = max(self.ext_conc[metabolite.id], 0.0)
                    if conc > 0: 
                        inhibitory_factor = kn_value / (kn_value + conc)
                        biomass_inhibition_product_term *= inhibitory_factor
        return biomass_inhibition_product_term

    def compute_fluxes(self, ext_conc, metadata_str='') -> Tuple[float, float, Dict[str, float]]: 
        '''
        Computes fluxes via COBRA backend for Linear Programming. 
        Returns biomass growth rate, product inhibition term, and fluxes. 
        '''
        self.ext_conc = ext_conc

        biomass_inhibition_product_term = self._set_dynamic_bounds() 
        sol = self.model.optimize(objective_sense='maximize', raise_error=False)
        if sol.status == 'infeasible': 
            print(f'At {metadata_str}: Infeasible solver.')
            self.solution_feasible = False
        
        v_objective = sol.objective_value 

        # mu = biomass growth rate 
        mu = sol.fluxes[self.BIOMASS_RXN]
        return mu, biomass_inhibition_product_term, sol.fluxes 