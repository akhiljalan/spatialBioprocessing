import numpy as np 
from itertools import product 
from DynamicFBASolver import DynamicFBASolver
from joblib import Parallel, delayed
from cobra.util.array import create_stoichiometric_matrix, constraint_matrices
import gurobipy as gp
import time
from time import strftime, localtime 
import logging 
from typing import Dict, List, Optional
'''
Simulates fluid advection from a steady-state velocity vield, 
and biomass growth via dynamic Flux Balance Analysis (FBA). 
'''

class SpatialSimulator: 
    '''
    Spatial simulation with a naive (rectangular) mesh geometry. 
    '''
    def __init__(self, 
        nx: int, 
        ny: int, 
        nz: int, 
        velocity_field: np.ndarray, 
        solver: DynamicFBASolver, 
        metabolites_to_track: list[str], 
        dt_fast: float = 0.01, # seconds 
        volume_total: float = None, # Liters 
        initial_biomass: float = None, # gDW, per voxel 
        initial_biomass_total: float = None, # gDW
        ext_conc: dict = None, 
        setpoints: Optional[Dict[str, float]] = None, # Metabolite ID -> External concentration (mM)
        met_subset: List[str] = [
            'glc__D_e', 'gln__L_c', 'lac__D_e', 
            'nh4_e', 'pi_e', 'h_e', 
            'h2o_e', 'pyr_e', 'glu__L_e', 
            'gln__L_e'
        ], 
        verbose: str = 'None', 
    ): 
        '''
        verbose: 
            If 'None', then minimal/no printing. 
            If 'Minimal' then prints every 10% of steps 
            If 'Debug', then prints several debugging steps as well. 
        '''
        self.nx = nx
        self.ny = ny 
        self.nz = nz 
        self.N = self.nx * self.ny * self.nz 

        self.dt = dt_fast 
        self.volume_total = volume_total
        self.voxel_volume = self.volume_total / self.N  
        print(f'{self.voxel_volume=}')
        self.initial_biomass_total = initial_biomass_total
        
        self.initial_biomass = initial_biomass
        self.ext_conc = ext_conc
        self.solver = solver

        self.verbose = verbose.lower()

        self.metabolites_to_track = metabolites_to_track
        

        # If no velocity field is pre-specified, we generate 
        # a random field with near-zero divergence. 
        if not velocity_field: 
            self.velocity_field = SpatialSimulator.generate_random_velocity_field_simple(
                nx = self.nx, ny = self.ny, nz = self.nz 
            )
        else: 
            self.velocity_field = velocity_field 
        self.concentrations_index_dict = self._init_concentrations_index_dict()
        # Initialize and check setpoints 
        self.setpoints = setpoints or {}
        self._check_setpoint_keys() 
        # Initialize concentration fields
        self._initialize_fields()
        # Precompute maps for advection steps based on velocity field. 
        self._precompute_advection_maps()
        # Precompute items for LP subproblems arising in FBA
        self._set_fba_items_init()


        # set index for biomass reaction to track 
        self.biomass_rxn_idx = self.reaction_id_to_idx.get(
            self.solver.BIOMASS_RXN, None
        )
        assert self.biomass_rxn_idx, f'Could not find biomass reaction index, expected {self.solver.BIOMASS_RXN}.'

        # Set metabolites to track 
        self.met_subset = met_subset 
        self.logger = logger.getLogger(__name__)
        
    def _init_concentrations_index_dict(self) -> Dict[str, int]: 
        '''
        Initialize dictionary used to keep tracking of 
        metabolite indexing, e.g. d['gln_E'] = 2. 
        '''
        d = {}
        idx = 0 
        for m in self.metabolites_to_track: 
            d[m] = idx 
            idx += 1
        return d 

    def _check_setpoint_keys(self) -> None: 
        '''
        For each key in self.setpoints, ensure it is 
        present in self.ext_conc. 
        '''
        for k in self.setpoints: 
            if k not in self.concentrations_index_dict: 
                raise ValueError(f"{k} set to {self.setpoints[k]} but not found in external concentrations.")


    def _initialize_fields(self) -> None: 
        '''
        Initialize concentrations of biomass and external metabolites.
        Stored as self.concentrations (4D array) of shape [nx, ny, nz, num_metabolites]. 
        '''
        num_metabolites = len(self.metabolites_to_track)
        self.concentrations = np.zeros(shape=(
            self.nx, self.ny, self.nz, 
            num_metabolites)
        )

        # Keep track of total biomass for easy reporting later. 
        self.biomass_time_series = []

        for m in self.ext_conc: 
            metabolite_idx = self.concentrations_index_dict[m]
            self.concentrations[:, :, :, metabolite_idx] = self.ext_conc[m] * np.ones(
                shape=(self.nx, self.ny, self.nz)
            )

        # Biomass is tracked in its own grid 
        # Assume initial biomass is uniform across all self.N voxels. 
        self.biomass_grid = (self.initial_biomass_total / self.N) * np.ones(
            (self.nx, self.ny, self.nz))
        
        if self.verbose in ['debug']: 
            print("Total biomass (gDW):", np.sum(self.biomass_grid))
            print("Total volume (L):", self.volume_total)


    def _precompute_advection_maps(self) -> None: 
        '''
        Precomputes weighting/advection mappings used for
        advection steps using semi-Lagrangian advection. 
        '''
        xg = np.arange(self.nx) + 0.5
        yg = np.arange(self.ny) + 0.5
        zg = np.arange(self.nz) + 0.5

        X, Y, Z = np.meshgrid(xg, yg, zg, indexing='ij') 
        vx = self.velocity_field[..., 0]
        vy = self.velocity_field[..., 1]
        vz = self.velocity_field[..., 2]

        # backtracing 
        xp = X - vx * self.dt 
        yp = Y - vy * self.dt 
        zp = Z - vz * self.dt 

        # use periodic boundary conditions 
        # each grid cell has length (1 x 1 x 1), in some units 
        # note that the velocity_field must respect these 
        xp = np.mod(xp, self.nx)
        yp = np.mod(yp, self.ny)
        zp = np.mod(zp, self.nz)

        i0 = np.floor(xp - 0.5).astype(np.int32) % self.nx 
        j0 = np.floor(yp - 0.5).astype(np.int32) % self.ny 
        k0 = np.floor(zp - 0.5).astype(np.int32) % self.nz 

        fx = (xp - 0.5) - i0 
        fy = (yp - 0.5) - j0 
        fz = (zp - 0.5) - k0 
        fx = np.clip(fx.astype(np.float32), 0.0, 1.0)
        fy = np.clip(fy.astype(np.float32), 0.0, 1.0)
        fz = np.clip(fz.astype(np.float32), 0.0, 1.0)

        self.neighbor_idx = np.zeros(shape=(self.nx, self.ny, self.nz, 8, 3), dtype=np.float32)
        self.advection_weights = np.zeros(shape=(self.nx, self.ny, self.nz, 8), dtype=np.float32)

        # 8 cube corners (di, dj, dk) ∈ {0,1}^3
        corners = [
            (0, 0, 0, lambda fx, fy, fz: (1 - fx) * (1 - fy) * (1 - fz)),
            (1, 0, 0, lambda fx, fy, fz: fx * (1 - fy) * (1 - fz)),
            (0, 1, 0, lambda fx, fy, fz: (1 - fx) * fy * (1 - fz)),
            (1, 1, 0, lambda fx, fy, fz: fx * fy * (1 - fz)),
            (0, 0, 1, lambda fx, fy, fz: (1 - fx) * (1 - fy) * fz),
            (1, 0, 1, lambda fx, fy, fz: fx * (1 - fy) * fz),
            (0, 1, 1, lambda fx, fy, fz: (1 - fx) * fy * fz),
            (1, 1, 1, lambda fx, fy, fz: fx * fy * fz),
        ]

        for idx, (di, dj, dk, wtfunc) in enumerate(corners): 
            ii = (i0 + di) % self.nx 
            jj = (j0 + dj) % self.ny 
            kk = (k0 + dk) % self.nz 

            self.neighbor_idx[..., idx, 0] = ii 
            self.neighbor_idx[..., idx, 1] = jj 
            self.neighbor_idx[..., idx, 2] = kk 
            self.advection_weights[..., idx] = wtfunc(fx, fy, fz)
        self.advection_weights = np.clip(self.advection_weights, 0.0, 1.0)
        self.advection_weights /= np.sum(self.advection_weights, axis=-1, keepdims=True)


    def _advect_fields(self): 
        '''
        Advects concnetrations according to self.velocity_field. 
        Assumes that _precompute_advection_maps has been run. 
        '''
        C = self.concentrations
        totals_before = np.sum(C, axis=(0,1,2))
        C_new = np.zeros_like(C)

        # Loop over 8 neighbors for each voxel 
        for n in range(8): 
            ii = self.neighbor_idx[..., n, 0].astype(int)
            jj = self.neighbor_idx[..., n, 1].astype(int)
            kk = self.neighbor_idx[..., n, 2].astype(int)
            w = self.advection_weights[..., n][..., None]
            C_new += w * C[ii, jj, kk, :]
        
        # mass-adjustment per metabolite 
        totals_after = np.sum(C_new, axis=(0,1,2))
        scale = np.ones_like(totals_before)
        nonzero_indices = totals_after > 0
        scale[nonzero_indices] = totals_before[nonzero_indices] / totals_after[nonzero_indices]
        C_new *= scale # broadcasts along last index 
        
        self.concentrations = C_new 

    def _perform_reactions_gurobi(self, timestep=None) -> None: 
        '''
        Updates FBA constraints based on the external concentrations 
        at each point in space. 
        Then, solves for the resulting FBA problem and obtains fluxes. 
        '''
        self._update_constraints_from_external_concentrations()

        # fluxes has shape of X_opt 
        self.fluxes = self._solve_gurobi()
        if timestep % 100 == 0 and self.verbose in ['debug']:  
            print("Flux array sanity:", self.fluxes.shape, self.L.shape)
            print("First few values EX_glc__D_e:", self.fluxes[self.reaction_id_to_idx["EX_glc__D_e"], :10])

    def _set_fba_items_init(self) -> None:
        '''
        Initializes data for LP instance arising from FBA. 
        S: Stoichiometric matrix 
        L, U: Matrices of lower/upper bounds. 
        objective_coeffs: Vector of the form [0, .., 0, 1, 0, ...]
        that is 1 only at the index of the biomass reaction. 

        The LP is of the form: 

        max <objective_coeffs, X[:, i]> 
        s.t. S @ X[:, i] = 0 
        and L[:, i] <= X[:, i] <= U[:, i]

        for all i. 
        ''' 
        num_voxels = self.N 
        n_reactions = len(self.solver.model.reactions)
        L = np.zeros(shape=(n_reactions, num_voxels))
        U = np.zeros_like(L)
        k_ones = np.ones(num_voxels)
        reaction_id_to_idx = {}
        for index, rxn in enumerate(self.solver.model.reactions): 
            lb = rxn.lower_bound
            ub = rxn.upper_bound 
            L[index, :] = lb * k_ones
            U[index, :] = ub * k_ones 
            reaction_id_to_idx[rxn.id] = index 
        self.L = L
        self.U = U 
        self.reaction_id_to_idx = reaction_id_to_idx
        self.S = create_stoichiometric_matrix(self.solver.model)

        objective_coeffs = np.zeros(shape=(self.S.shape[1]))
        biomass_reaction_index = list(self.solver.model.reactions).index(
            self.solver.model.reactions.Biomass_Ecoli_core
        )
        objective_coeffs[biomass_reaction_index] = 1.0 

        self.objective_coeffs = objective_coeffs

    def _update_constraints_from_external_concentrations(self) -> None: 
        '''
        Updates the lower bounds matrix L based on 
        external concentrations, via Michaelis-Menten formuls. 
        '''
        fba_solver = self.solver 
        L_cur = self.L.copy()

        # one term for every voxel 
        self.biomass_inhibitions = 1.0 + np.zeros(shape=(self.N))
        for metabolite, reaction in fba_solver.exchange_reactions_map.items(): 
            # print(f'Checking {metabolite} and {reaction}')
            if not (metabolite in self.ext_conc and metabolite in self.concentrations_index_dict): 
                continue 
            for i, j, k in product(
                range(self.nx), 
                range(self.ny), 
                range(self.nz)
                ): 
                if type(metabolite) != str: 
                    conc = self.concentrations[
                        i, j, k, 
                        self.concentrations_index_dict[metabolite]
                    ]
                else: 
                        conc = self.concentrations[
                        i, j, k, 
                        self.concentrations_index_dict[metabolite]
                    ]
                reaction_current = fba_solver.model.reactions.get_by_id(reaction)
                reaction_idx = self.reaction_id_to_idx[reaction]

                V_max = fba_solver.vmax_params.get(reaction_current.id, 10.0)
                Km = fba_solver.km_params.get(reaction_current.id, 0.01)
                uptake_limit = -1.0 * V_max * conc / (Km + conc)
                
                spatial_idx = (
                    i * self.ny + self.nz 
                    + j * self.nz 
                    + k
                )
                old_bound = L_cur[reaction_idx, spatial_idx]
                L_cur[reaction_idx, spatial_idx] = np.maximum(uptake_limit, old_bound)

                for reaction_id, kn_value in fba_solver.kn_params.items(): 
                    rxn_current = fba_solver.model.reactions.get_by_id(reaction_id)
                    for met in rxn_current.metabolites: 
                        conc = max(0.0, 
                            self.concentrations[
                            i, j, k, 
                            self.concentrations_index_dict[met.id]
                        ])
                        inhibitory_factor = kn_value / (kn_value + conc)
                        self.biomass_inhibitions[spatial_idx] *= inhibitory_factor
        self.L = L_cur 

        
    def _solve_gurobi(self): 
        '''
        Solves the LP arising from the parallel FBA instances
        using Gurobi with warm-starts. 

        S: Stoichiometric matrix 
        L, U: Matrices of lower/upper bounds. 
        objective_coeffs: Vector of the form [0, .., 0, 1, 0, ...]
        that is 1 only at the index of the biomass reaction. 

        The LP is of the form: 

        max <objective_coeffs, X[:, i]> 
        s.t. S @ X[:, i] = 0 
        and L[:, i] <= X[:, i] <= U[:, i]

        for all i. 
        '''
        n = self.U.shape[0] # number of vars in each LP 
        k = self.U.shape[1] # number of indepedent LPs 

        # set up the model as blank 
        m = gp.Model()
        m.Params.OutputFlag = 0  # Suppress Gurobi solver output

        # warm start by solving the first lP 
        zero_vec = np.zeros(shape=self.S.shape[0])
        
        # minimize (-1 * c^T x)
        x = m.addMVar(n, lb=self.L[:,0], ub=self.U[:,0], obj=-1.0 * self.objective_coeffs) 
        
        # Warm-start Gurobi 
        m.addMConstr(self.S, x, sense='=', b=zero_vec)
        m.optimize()
        X_opt = np.zeros_like(self.L)
        try: 
            X_opt[:, 0] = x.X 
        except gp.GurobiError as e: 
            # TODO 
            print(f'Solver infeasible for idx 0.')
            
        # Solve the rest of the instances using Gurobi. 
        for j in range(1, k):
            x.lb = self.L[:, j]
            m.optimize()  # reuses basis
            try: 
                X_opt[:, j] = x.X
            except gp.GurobiError as e: 
                print(f'Solver infeasible for idx {j}.')
        return X_opt
    
    def step(self, timestep=None): 
        # 1. Fluid Transport via semi-Lagrangian advection. 
        self._advect_fields()

        # 2. Flux Balance Analysis 
        self._perform_reactions_gurobi(timestep = timestep)

        for rxn, idx in self.reaction_id_to_idx.items(): 
            if 'EX_' in rxn and timestep % 100 == 0 and self.verbose in ['debug']:  
                print(rxn, np.mean(self.fluxes[idx, :]))

        # 3. Update biomass 
        inhibitions = self.biomass_inhibitions.reshape(
            self.nx, self.ny, self.nz
        )

        biomass_idx = self.biomass_rxn_idx 

        # Fluxes are per-hour, our self.dt is per-second 
        time_conversion_factor = 1.0 / 3600.0 

        mu_grid = self.fluxes[biomass_idx, :].reshape(
            self.nx, self.ny, self.nz
        )
        
        self.biomass_grid += mu_grid * inhibitions * self.biomass_grid * self.dt * time_conversion_factor 

        # 4. Update metabolite concentrations from fluxes 
        for met, rxn in self.solver.exchange_reactions_map.items(): 
            # Skip updates for metabolites not tracked in concentrations
            # OR those that are in setpoints. 
            met_invalid = met not in self.concentrations_index_dict or met in self.setpoints
            met_is_exchange = met.endswith('_e')
            if met_invalid or not met_is_exchange: 
                continue 
            met_idx = self.concentrations_index_dict[met]
            rxn_idx = self.reaction_id_to_idx[rxn] 
            flux_grid = self.fluxes[rxn_idx, :].reshape(
                self.nx, self.ny, self.nz
            )
            
            net_flux = np.sum(flux_grid)
            if timestep % 100 == 0 and self.verbose in ['debug']: 
                print(f"{timestep=}, {rxn=}, Sum flux_grid:", net_flux)
                if np.sign(net_flux) != np.sign(np.mean(self.fluxes[rxn_idx, :])):
                    print(f"Warning: voxel reshape sign mismatch for {rxn}")

            delta_conc = time_conversion_factor * self.dt * flux_grid * self.biomass_grid / self.voxel_volume 
            self.concentrations[:, :, :, met_idx] += delta_conc # reshape? 

        # 5. Clip 
        np.clip(self.concentrations, 0.0, None, out = self.concentrations)

    def run(self, num_timesteps): 
        start_time = time.time()
        
        start_time_str = strftime('%Y-%m-%d %H:%M:%S', localtime(start_time))

        print(f'Beginning {num_timesteps} step run at {start_time_str}')
        print_every = int(num_timesteps / 10)
        total_biomass = self.biomass_grid.sum()
        self.biomass_time_series.append(total_biomass)
        # print_every = 100 
        for t in range(num_timesteps): 
            self.step(timestep=t)
            if t < 20 and self.verbose in ['debug']: 
                print('\n' + '-' * 100)
                print(f'Step {t} done.')
                sample_rxns = ["EX_glc__D_e", "EX_o2_e", "EX_lac__D_e"]
                for rxn in sample_rxns:
                    idx = self.reaction_id_to_idx.get(rxn)
                    if idx is not None:
                        print(f"{rxn}: mean flux = {np.mean(self.fluxes[idx, :]):.4f} mmol/gDW/h")
                        print(f"{rxn}: min flux = {np.min(self.fluxes[idx, :]):.4f} mmol/gDW/h")
                        print(f"{rxn}: max flux = {np.max(self.fluxes[idx, :]):.4f} mmol/gDW/h")
            elif t % print_every == 0 and self.verbose in ['debug', 'minimal']: 
                print('\n' + '-' * 100)
                print(f'Step {t} done.')
                total_biomass = self.biomass_grid.sum()
                self.biomass_time_series.append(total_biomass)
                if self.verbose == 'debug': 
                    for rxn in sample_rxns:
                        idx = self.reaction_id_to_idx.get(rxn)
                        if idx is not None:
                            print(f"{rxn}: mean flux = {np.mean(self.fluxes[idx, :]):.4f} mmol/gDW/h")
                            print(f"{rxn}: min flux = {np.min(self.fluxes[idx, :]):.4f} mmol/gDW/h")
                            print(f"{rxn}: max flux = {np.max(self.fluxes[idx, :]):.4f} mmol/gDW/h")
                    self.report_total_masses(metabolites = self.met_subset)
                cur_time = time.time()
                elapsed = cur_time - start_time
                print(f"{t} steps completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")
            else: 
                pass 
        end_time = time.time()
        end_time_str = strftime('%Y-%m-%d %H:%M:%S', localtime(end_time))
        elapsed = end_time - start_time
        total_real_time = num_timesteps * self.dt 
        print(f"\nSimulation completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")
        print(f"Simulated {total_real_time:.2f} seconds of real time in {elapsed:.2f} seconds.")
        print("\n=== Final Total Masses ===")
        self.report_total_masses(metabolites = self.met_subset)


    def report_total_masses(self, metabolites: list[str]) -> None:
        '''
        Prints total mass (across all voxels) for given metabolites 
        and biomass.
        '''
        voxel_vol = self.voxel_volume  
        total_biomass = np.sum(self.biomass_grid) # * self.voxel_volume
        total_vol = self.voxel_volume * self.N 

        biomass_init = self.initial_biomass_total
        growth_ratio = total_biomass / biomass_init
        print(f"Initial biomass: {biomass_init:.6f} grams dry weight")
        print(f"End Biomass: {total_biomass:.6f}  grams dry weight")
        print(f"Overall rate of growth: {growth_ratio:.6f}")
        
        for met in metabolites:
            if met not in self.concentrations_index_dict:
                print(f"[Warning] {met} not tracked in simulation.")
                continue
            met_idx = self.concentrations_index_dict[met]
            total_mass = np.sum(self.concentrations[..., met_idx]) * voxel_vol
            indiv_masses = self.concentrations[..., met_idx] * voxel_vol 
            print(f"{met}: {total_mass:.6f} mM")

    '''
    Helper functions to compute a random velocity field 
    '''
    @staticmethod
    def compute_divergence(field):
        # field shape: (nx, ny, nz, 3)
        u, v, w = field[..., 0], field[..., 1], field[..., 2]

        du_dx = np.gradient(u, axis=0)
        dv_dy = np.gradient(v, axis=1)
        dw_dz = np.gradient(w, axis=2)

        return du_dx + dv_dy + dw_dz
        
    @staticmethod
    def generate_random_velocity_field_simple(nx=8, ny=8, nz=8, seed=42, div_tol = 1e-3, n_tries_limit = 1000):
        field_found = False 
        np.random.seed(seed)
        num_tries = 0 
        best_div = 1e1
        best_field = None  
        while not field_found: 
            v = np.random.normal(size=(nx, ny, nz, 3))
            v /= np.linalg.norm(v, axis=-1, keepdims=True)  # per-point init only; ok
            avg_divergence = np.abs(np.mean(SpatialSimulator.compute_divergence(v)))
            if avg_divergence < best_div: 
                best_div = avg_divergence 
                best_field = v 
            num_tries += 1 
            field_found = num_tries >= n_tries_limit or avg_divergence < div_tol 
        print(f'Generated random velocity field with mean div: {best_div}')
        return best_field 