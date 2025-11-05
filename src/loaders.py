import json
from SpatialSimulator import SpatialSimulator
from DynamicFBASolver import DynamicFBASolver
import numpy as np


def load_spatial_simulator_from_json(json_path: str) -> SpatialSimulator:
    '''
    Loads a SpatialSimulator object from JSON. 
    '''
    with open(json_path, 'r') as f:
        cfg = json.load(f)
    
    # Build the FBA solver
    solver = DynamicFBASolver(
        model_name=cfg.get("model_name", "textbook"),
        vmax_params=cfg.get("vmax_params", {}),
        km_params=cfg.get("km_params", {}),
        kn_params=cfg.get("kn_params", {}),
        ext_conc=cfg.get("ext_conc", {}),
        BIOMASS_RXN=cfg.get("biomass_rxn", "Biomass_Ecoli_core"),
    )
    
    
    sim = SpatialSimulator(
        nx=cfg.get("nx", 4), ny=cfg.get("ny", 4), nz=cfg.get("nz", 4),
        velocity_field=None,
        solver=solver,
        metabolites_to_track=list(cfg["ext_conc"].keys()),
        dt_fast=cfg.get("dt", 0.01),
        volume_total=cfg.get("volume", 64.0),
        initial_biomass_total=cfg.get("biomass_init", 1.0),
        ext_conc=cfg.get("ext_conc", {}),
        setpoints=cfg.get("setpoints", {}),
        verbose=cfg.get("verbose", "minimal")
    )

    return sim