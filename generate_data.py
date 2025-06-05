from simulation import generate_containers 
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

from sklearn.linear_model import LinearRegression
import numpy as np



# def linear_increase(start: int, end: int, steps: int):
#     """Returns a list of integers linearly increasing from start to end inclusive."""
#     return [round(start + i * (end - start) / (steps - 1)) for i in range(steps)]

def generate_from_regression(data, steps, round_values=True):
    """
    Fits a linear regression to the input data and generates `steps` new values using the model.

    Args:
        data (list or array): The original y-values (1D).
        steps (int): Number of new values to generate.
        round_values (bool): Whether to round the generated values to integers.

    Returns:
        list: Generated values using the linear model.
    """
    # Prepare x and y
    y = np.array(data)
    x = np.arange(len(y)).reshape(-1, 1)

    # Fit regression
    model = LinearRegression()
    model.fit(x, y)

    # Extract slope and intercept
    slope = model.coef_[0]
    intercept = model.intercept_

    # Generate new x and y values
    x_new = np.arange(steps)
    y_new = slope * x_new + intercept

    if round_values:
        y_new = y_new.round().astype(int)

    return y_new.tolist()



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


    # first few data
    data_pre_june = [6, 5, 13, 20, 28, 32, 30, 38, 40]
    # Simulate containers from small number (e.g. 10) up to NUMBER_OF_CONTAINERS
    ramp_numbers = generate_from_regression(data_pre_june, ramp_days, round_values=True)

    all_data = []
    # === Generate from May 24 to June 1 (ramping up)
    for i in range(ramp_days):
        current_date = ramp_start_date + timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")

        containers_list = generate_containers(
            probability_excel_path,
            number_of_containers=ramp_numbers[i],
            date=date_str
        )
        all_data.append(pd.DataFrame(containers_list))

    
    # === Generate from June 2 to June 30 (normal distribution with μ=60, σ=10)
    for i in range((end_date - start_date).days + 1):
        current_date = start_date + timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")

        # Sample number of containers, clamp to at least 10
        number_of_containers = max(10, int(np.random.normal(loc=60, scale=10)))

        containers_list = generate_containers(
            probability_excel_path,
            number_of_containers=number_of_containers,
            date=date_str
        )
        all_data.append(pd.DataFrame(containers_list))

    
    monthly_df = pd.concat(all_data, ignore_index=True)
    monthly_df.to_excel(simulation_data / "june_2025_containters_mean_60_sigma_10.xlsx", index=False)
    print(monthly_df)
