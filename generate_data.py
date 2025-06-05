from simulation import NUMBER_OF_CONTAINERS, generate_containers 
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

def linear_increase(start: int, end: int, steps: int):
    """Returns a list of integers linearly increasing from start to end inclusive."""
    return [round(start + i * (end - start) / (steps - 1)) for i in range(steps)]



# GENERATING the synthetic data 
if __name__ == "__main__":
    simulation_data = Path("simulation_artifacts")
    input_dir = Path("input")
    probability_excel_path = input_dir / "UBmagad.xlsx"
    start_date = datetime(2025, 6, 1)
    end_date = datetime(2025, 6, 30)
    all_data = []

    # === Generate from May 24 to June 1 (ramping up)
    ramp_start_date = datetime(2025, 5, 24)
    ramp_end_date = datetime(2025, 6, 1)
    ramp_days = (ramp_end_date - ramp_start_date).days + 1

    # Simulate containers from small number (e.g. 10) up to NUMBER_OF_CONTAINERS
    ramp_numbers = linear_increase(10, NUMBER_OF_CONTAINERS, ramp_days)

    all_data = []

    for i in range(ramp_days):
        current_date = ramp_start_date + timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")

        containers_df = generate_containers(
            probability_excel_path,
            number_of_containers=ramp_numbers[i],
            date=date_str
        )
        all_data.append(containers_df)

    for i in range((end_date - start_date).days + 1):
        current_date = start_date + timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
    
        containers_df = generate_containers(
            probability_excel_path,
            number_of_containers=NUMBER_OF_CONTAINERS,
            date=date_str
        )
        all_data.append(containers_df)
    monthly_df = pd.concat(all_data, ignore_index=True)
    monthly_df.to_excel(simulation_data / "june_2025_containters_extended_smoothed.xlsx", index=False)
    print(monthly_df)
