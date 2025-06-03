# Train Track Simulation & Optimization

This repository provides a simulation and cost analysis tool for train wagon delivery and optimization, tailored for logistics and railway operations in Mongolia.

## Overview
- **Purpose:** Simulate the loading, delivery, and cost optimization of train wagons to various terminals using real and synthetic data.
- **Key Features:**
  - Probabilistic container generation based on real terminal distributions
  - Flexible train formation and loading strategies
  - Cost calculation (locomotive, penalty, and delivery)
  - Transition and efficiency analysis
  - Support for real Excel data and synthetic simulation

## Main Scripts
- `simulation.py`: Core simulation logic for generating containers, forming trains, loading strategies, and cost analysis.
- `train_track_sim_comparison.py`: Advanced analysis and comparison of real vs. optimized train formations using historical data.

## Data Files
- `input/UBmagad.xlsx`: Terminal probability data for simulation.
- `simulation_artifacts/containers.xlsx`: Generated containers for each simulation run.
- `Train_sim_data.xlsx`, `Import_sim_data.xlsx`: Real operational data for advanced analysis.

## Key Functions (simulation.py)
- `generate_containers`: Generate containers with terminal assignments based on probability distributions.
- `generate_trains`: Create empty trains (lists of wagons) for loading.
- `decision_to_load_containers_`: Load containers onto trains using a specified strategy (random, grouped, etc.).
- `train_to_terminal_sequence`: Convert a train's terminal assignments to integer codes for analysis.
- `count_transitions`: Count the number of terminal transitions in a train (for efficiency/cost).
- `count_out`: Calculate total operational costs for a train.
- `load_containers_to_trains`: Sequentially load containers onto trains, removing loaded containers from the pool.

## Dependencies
- Python 3.8+
- pandas
- numpy
- openpyxl

Install dependencies with:
```sh
pip install -r requirements.txt
```
Or use the provided `pyproject.toml` with Poetry or uv.

## Usage
1. Place your terminal probability Excel file in the `input/` directory (e.g., `UBmagad.xlsx`).
2. Run the simulation:
   ```sh
   python simulation.py
   ```
3. Review generated artifacts in `simulation_artifacts/` and console output for cost and transition analysis.

## Customization
- Edit simulation parameters (number of containers, trains, wagons) at the top of `simulation.py`.
- Add or update terminal information in the `TERMINALS` dictionary.
- Implement new loading strategies by defining new decision functions and passing them to the loader.

## License
This project is for research and educational purposes. Contact the author for other uses. 