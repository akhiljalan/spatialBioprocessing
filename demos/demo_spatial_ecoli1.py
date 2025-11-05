import os
import sys 

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
sys.path.append(src_path)
print(src_path)
from loaders import load_spatial_simulator_from_json
import io_utils



if __name__ == "__main__":
    config_path = os.path.join(project_root, "configs", "ecoli_config1.json")
    sim = load_spatial_simulator_from_json(config_path)



    n_hours = 0.5
    dt_seconds = sim.dt
    n_steps = int(3600.0 * n_hours / dt_seconds)

    sim.run(num_timesteps=n_steps)

    demo_folder_path = os.path.join(project_root, "demos", "spatial_ecoli1")
    os.makedirs(demo_folder_path, exist_ok=True)

    csv_path = f"{demo_folder_path}/results_timeseries.csv"
    pdf_path = f"{demo_folder_path}/results_grid.pdf"

    io_utils.save_results_to_csv(sim, csv_path)
    io_utils.plot_results_to_pdf_grid(csv_path, pdf_path)