import numpy as np
import pandas as pd

import random
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union, List, Callable, Sequence, Dict

# Simulation parameters
NUMBER_OF_CONTAINERS = 110
NUMBER_OF_TRAINS = 2
NUMBER_OF_WAGONS = 55
PROBABILITY = 0.6

LOCOMOTIVE_COST_PER_HOUR = 210000 # MNT
LOCOMOTIVE_COST_PER_MINUTE = LOCOMOTIVE_COST_PER_HOUR / 60  # 3500 MNT per minute

PENALTY_PER_HOUR = 900 # Per container
PENALTY_PER_MINUTE = PENALTY_PER_HOUR / 60 # 15 MNT per minute

MINUTES_PER_HOROO = 10 # 10 minutes to do one classification

STATIONS = ['УБ', 'Ам', 'То'] # redundant

# region here refers to бүс I, II
# zuragnaas
# TERMINALS = {
#     'УБ МЧ': {'distance': 8, 'capacity': 18, 'station': 'УБ', 'region': 1},
#     'Туушин': {'distance': 20, 'capacity': 8, 'station': 'УБ'},
#     'Монгол экс': {'distance': 20, 'capacity': 13, 'station': 'УБ'},
#     'материалын импекс': {'distance': 40, 'capacity': 39, 'station': 'УБ'},
#     'Прогресс': {'distance': 60, 'capacity': 14, 'station': 'УБ'},
#     'Эрин': {'distance': 60, 'capacity': 46, 'station': 'УБ'},
#     'Техник импорт': {'distance': 40, 'capacity': 16, 'station': 'УБ'},
#     'Амгалан': {'distance': 15, 'capacity': 37, 'station': 'Ам'},
#     'Интердэс': {'distance': 20, 'capacity': 15, 'station': 'То'},
#     'Монгол тр': {'distance': 10, 'capacity': 15, 'station': 'То'},
#     'Толгойт М': {'distance': 5, 'capacity': 8, 'station': 'То'},
#     'Номин Тр': {'distance': 20, 'capacity': 8, 'station': 'То'},
# }

# excel(screenshot) Urtuunii salbar
TERMINALS = {
    'УБ МЧ': {'distance': 8, 'capacity': 33, 'station': 'УБ', 'region': 1},
    'Туушин': {'distance': 20, 'capacity': 8, 'station': 'УБ', 'region': 1},
    'Монгол экс': {'distance': 20, 'capacity': 10, 'station': 'УБ', 'region': 1},
    'материалын импекс': {'distance': 40, 'capacity': 23, 'station': 'УБ', 'region': 2},
    'Прогресс': {'distance': 60, 'capacity': 13, 'station': 'УБ', 'region': 1},
    'Эрин': {'distance': 60, 'capacity': 7, 'station': 'УБ', 'region': 2},
    'Техник импорт': {'distance': 40, 'capacity': 12, 'station': 'УБ', 'region': 2},
    'Амгалан': {'distance': 15, 'capacity': 12, 'station': 'Ам', 'region': 1},
    'Интердэсишн': {'distance': 20, 'capacity': 8, 'station': 'То', 'region': 1},
    'Монгол транс': {'distance': 10, 'capacity': 15, 'station': 'То', 'region': 1},
    'Толгойт МЧ': {'distance': 5, 'capacity': 10, 'station': 'То', 'region': 1},
    'Номин Трэйдинг': {'distance': 20, 'capacity': 8, 'station': 'То', 'region': 2}, # UB Impex
}

# TODO: Calimariin zardal


TERMINALS_TO_NUMBER = {
    'УБ МЧ': 0,
    'Туушин': 1,
    'Монгол экс': 2,
    'материалын импекс': 3,
    'Прогресс': 4, 
    'Эрин': 5,
    'Техник импорт': 6,
    'Амгалан': 7,
    'Интердэсишн': 8,
    'Монгол транс': 9,
    'Толгойт МЧ': 10,
    'Номин Трэйдинг': 11,
}

############################## Generation process #############################

def generate_containers(
        probability_excel_path: Union[str, Path], 
        number_of_containers: int, 
        date: str,
        terminal_col: str="Терминал",
        count_col: str="Count"
        ) -> List[Dict]:
    """
    Generate containers based on terminal counts using Laplace smoothing.
    """
    probability_df = pd.read_excel(probability_excel_path)
    raw_counts = probability_df[count_col].values
    smoothed_counts = raw_counts + 1
    probabilities = smoothed_counts / smoothed_counts.sum()
    terminals = probability_df[terminal_col].values
    container_terminals = np.random.choice(
        terminals,
        size=number_of_containers,
        p=probabilities
    )
    containers = [
        {'container_id': i, 'Терминал': terminal, 'Date': date}
        for i, terminal in enumerate(container_terminals)
    ]
    return containers

def generate_train(
          number_of_wagons: int=40
          ) -> List[dict]:
    """Generate trains, which is an empty container for wagons where each wagon has a container."""
    empty_container = [
        {'container_id': None, 'Терминал': None, 'Date': None}
        for _ in range(number_of_wagons)
    ]
    return empty_container

def generate_trains(
        number_of_trains: int,
        number_of_wagons: int
        ) -> List[List[dict]]:
    """Generate many trains"""
    return [generate_train(number_of_wagons) for _ in range(number_of_trains)]

############################# END OF GENERATION ###############################

# def decision_to_load_containers_(
#         containers: pd.DataFrame, 
#         train: List[dict], 
#         decision_function: Callable) -> List[dict]:
#     """
#     Fill the train (list of dicts) with containers according to the decision_function.
#     The train is a fixed-size list of dicts with 'container_id' and 'Терминал' fields.
#     Warning: Mutates the train 
#     """
#     # random.shuffle(containers)
#     ordered_indices = decision_function(containers)
#     # Only fill up to the train's capacity
#     for i, idx in enumerate(ordered_indices[:len(train)]):
#         train[i]['container_id'] = containers.loc[idx, 'container_id']
#         train[i]['Терминал'] = containers.loc[idx, 'Терминал']
#     return train
####################### DECISIONS BANK ##############################################

#
# Decision is the function that takes in List of containers and then makes a decision
# outputs an indices on which where to put where.
#
def decision_random(containers: List[dict]) -> Sequence[int]:
    """
    Randomly shuffle the containers and return their indices.
    """
    indices = list(range(len(containers)))
    np.random.shuffle(indices)
    return indices

def decision_group_by_terminal(containers: List[dict]) -> Sequence[int]:
    """
    Group containers by 'Терминал' and return their indices in grouped order.
    """
    terminals = [(i, c['Терминал']) for i, c in enumerate(containers)]
    sorted_indices = [i for i, _ in sorted(terminals, key=lambda x: x[1])]
    return sorted_indices

def decision_group_probalistic(containers: List[dict], 
                               prob_same_terminal: float = 0.5,
                               seed: int = 42) -> Sequence[int]:
    """
    Reorder containers such that adjacent ones are likely to share the same terminal,
    using a Markov-style probability model.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Count how many containers are at each terminal
    terminal_counts = Counter(c['Терминал'] for c in containers)
    container_indices_by_terminal = {
        terminal: [i for i, c in enumerate(containers) if c['Терминал'] == terminal]
        for terminal in terminal_counts
    }

    # Build the sequence
    sequence = []
    terminals = list(terminal_counts.keys())

    # Start with a randomly chosen terminal (weighted by count)
    current_terminal = random.choices(
        terminals,
        weights=[terminal_counts[t] for t in terminals]
    )[0]

    current_list = container_indices_by_terminal[current_terminal]
    chosen = current_list.pop()
    sequence.append(chosen)
    terminal_counts[current_terminal] -= 1
    if terminal_counts[current_terminal] == 0:
        del terminal_counts[current_terminal]
        del container_indices_by_terminal[current_terminal]

    for _ in range(1, len(containers)):
        if random.random() < prob_same_terminal and current_terminal in terminal_counts:
            next_terminal = current_terminal
        else:
            available_terminals = list(terminal_counts.keys())
            if not available_terminals:
                break
            weights = [terminal_counts[t] for t in available_terminals]
            next_terminal = random.choices(available_terminals, weights=weights)[0]

        current_terminal = next_terminal
        next_list = container_indices_by_terminal[current_terminal]
        chosen = next_list.pop()
        sequence.append(chosen)
        terminal_counts[current_terminal] -= 1
        if terminal_counts[current_terminal] == 0:
            del terminal_counts[current_terminal]
            del container_indices_by_terminal[current_terminal]

    return sequence

def decision_optimal(containers: List[dict]) -> Sequence[int]:
    """
    Optimal decision based on the heuristic that the containers should be loaded
    in the geographic as well as regional order.
    Could've calculated the preferred order using the region and station order.

    containers variable is a list of dictionaries.
    Each containers should have:
     - "Терминал" 
     - "container_id"
     - "Date" which means when the container first arrived at Zamiin Uud.
    """
    # Hardcoded optimal decision
    # order = [7, 1, 2, 4, 0, 3, 6, 5, 8, 9, 10, 11] # using this order preference to populate the trains

    # for i, idx in enumerate(order):
    # return order
    preferred_order = [7, 1, 2, 4, 0, 3, 6, 5, 8, 9, 10, 11]  # using TERMINALS_TO_NUMBER values
    
    # Group containers by terminal
    terminal_groups = {}
    for idx, container in enumerate(containers):
        terminal = container['Терминал']
        terminal_num = TERMINALS_TO_NUMBER[terminal]
        if terminal_num not in terminal_groups:
            terminal_groups[terminal_num] = []
        terminal_groups[terminal_num].append(idx)
    
    # Build sequence following preferred order
    sequence = []
    for terminal_num in preferred_order:
        if terminal_num in terminal_groups:
            sequence.extend(terminal_groups[terminal_num])
    
    return sequence

def decision_bujnaa():
    # TODO: Insert Bujnaa's code here.
    pass
# Wrapper to prioritize date over order
def decision_fifo(decision_fn: Callable[[List[dict]], Sequence[int]]) -> Callable[[List[dict]], Sequence[int]]:
    """
    Wrap a decision function to first sort containers by Date (ascending),
    then apply the decision function **within each date**, but finally
    mix terminals fairly in a FIFO-aware way.
    """
    def fifo_decision(containers: List[dict]) -> Sequence[int]:
        # Group containers by date
        date_to_containers = defaultdict(list)
        for i, c in enumerate(containers):
            date_to_containers[c["Date"]].append((i, c))
        
        # Sort dates
        sorted_dates = sorted(date_to_containers.keys())

        final_order = []
        seen_indices = set()

        for date in sorted_dates:
            group = date_to_containers[date]
            if not group:
                continue
            
            # Unpack original indices and containers
            group_indices, group_containers = zip(*group)
            group_indices = list(group_indices)
            group_containers = list(group_containers)
            
            # Apply the wrapped decision function
            sub_order = decision_fn(group_containers)
            # Map back to original indices
            final_order.extend([group_indices[i] for i in sub_order if group_indices[i] not in seen_indices])
            seen_indices.update(group_indices)
        
        return final_order

    return fifo_decision

# LOAD THE CONTAINERS TO THE TRAIN
def load_containers_to_trains(containers: List[dict],
                               trains: List[List[dict]], 
                               decision_function: Callable) -> List[List[dict]]:
    """
    Loads containers onto trains in-place. Each train is filled up to its capacity,
    and loaded containers are removed from the available pool for subsequent trains.
    
    Returns the updated trains list for convenience.
    """
    available_indices = set(range(len(containers)))
    for train in trains:
        # Build a list of available containers (list of dicts)
        available_containers = [containers[i] for i in available_indices]
        # Map local indices in available_containers to original indices in containers
        idx_map = {i: orig_idx for i, orig_idx in enumerate(available_indices)}
        # Get the order in which to load containers
        ordered_indices = decision_function(available_containers)
        # Select indices for this train (up to train capacity)
        selected_indices = [idx_map[i] for i in ordered_indices[:len(train)]]
        for i, idx in enumerate(selected_indices):
            train[i]['container_id'] = containers[idx]['container_id']
            train[i]['Терминал'] = containers[idx]['Терминал']
            train[i]['Date'] = containers[idx]['Date']
        available_indices -= set(selected_indices)
        if not available_indices:
            break
    return trains


############################# - END-DECISION ##########################################

########################### CALCULATE-COST-FUNCTIONS ########################
def calculate_classification_cost(horoonii_too: int) -> int:
    """Here I mean classification as shunting cost or Huruudultiin zardal"""
    #  huleelgiin_chingeleg: int = 0, huleelgiin_chingelegiin_time: int = 0, total_terminaluudiin_hureh_zai: int = 318
    horoonii_zardal =horoonii_too = horoonii_too * MINUTES_PER_HOROO * LOCOMOTIVE_COST_PER_MINUTE # tsagt 210k minuted 3.5k
    # huleelgiin_zardal = huleelgiin_chingeleg * huleelgiin_chingelegiin_time * PENALTY_PER_MINUTE # minutaar
    # terminal_hurgeh_zardal = total_terminaluudiin_hureh_zai * LOCOMOTIVE_COST_PER_MINUTE
    # niit_zardal = horoonii_zardal + huleelgiin_zardal + terminal_hurgeh_zardal
    return horoonii_zardal

def calculate_wait_minutes():
    # TODO: Implement wait_minutes
    return 60


 ###################### END-CALCULATE ##########################3

def calculate_penalty_cost(train_sequences: List[List[Dict]]) -> float:
    """Calculate total penalty cost for containers exceeding terminal capacity."""
    total_penalty = 0

    for train_idx, train in enumerate(train_sequences):
        terminal_counts = {}
        for container in train:
            terminal = container['Терминал']
            terminal_counts[terminal] = terminal_counts.get(terminal, 0) + 1

        for terminal, count in terminal_counts.items():
            capacity = TERMINALS.get(terminal, {}).get("capacity", 1)
            if count > capacity:
                excess = count - capacity
                # print(f"[Train {train_idx}] Terminal '{terminal}' exceeded by {excess} container(s).")
                # TODO: Make `calculate_wait_minutes` function
                expected_minutes = calculate_wait_minutes()
                total_penalty += excess * PENALTY_PER_MINUTE * expected_minutes

    return total_penalty


# def load_containers_to_trains_(containers: pd.DataFrame,
#                               trains: List[List[dict]],
#                               decision_function: Callable) -> None:
#     """
#     Loads containers onto trains in-place. Each train is filled up to its capacity,
#     and loaded containers are removed from the available pool for subsequent trains.
#     """
#     pass
##################### HELPER FUNCTIONS #####################
def load_container_data(containers_file_path: Union[str, Path]) -> List[dict]:
    """Load container data from Excel file."""
    print(f"Loading container data from {containers_file_path}")
    containers_df = pd.read_excel(containers_file_path)
    containers = containers_df.to_dict(orient="records")
    print(f"Loaded {len(containers)} containers")
    return containers

def train_to_terminal_sequence(train: List[dict]) -> List[int]:
    """
    Transform a train (list of dicts with 'Терминал') into a sequence of ints using TERMINALS_TO_NUMBER.
    Returns a list of ints representing the terminal sequence for the train.
    """
    return [TERMINALS_TO_NUMBER.get(wagon['Терминал'], -1) for wagon in train if wagon['Терминал'] is not None]

def count_transitions(seq: Union[List, str]) -> int:
    """
    Хөрш өөр элементүүдийн огтлолын тоог тоолно.

    :param seq: Жагсаалт (List) эсвэл тэмдэгт мөр (str)
    :return: Огтлолын тоо (int)
    """
    if isinstance(seq, str):
        seq = seq.split(",")

    transitions = 0
    for i in range(1, len(seq)):
        if seq[i] != seq[i - 1]:
            transitions += 1
    return transitions

def dataframe_to_list(df: pd.DataFrame):
    return df.to_dict(orient="records")

def list_to_dataframe(results_list: List[dict]) -> pd.DataFrame:
    """Convert a list of dictionaries to a pandas DataFrame."""
    return pd.DataFrame(results_list)
#################### END-HELPER FUNCTIONS ###################


################### SIMULATION FUNCTIONS ####################
def get_date_range(start_date: str, end_date: str) -> List[str]:
    """
    Generate a list of dates between start_date and end_date (inclusive).
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
    
    Returns:
        List of date strings in 'YYYY-MM-DD' format
    """
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    return dates

def filter_containers_by_date(containers: List[dict], target_date: str) -> List[dict]:
    """
    Filter containers that arrived on or before the target date.
    
    Args:
        containers: List of container dictionaries
        target_date: Target date in 'YYYY-MM-DD' format
    
    Returns:
        Filtered list of containers
    """
    return [c for c in containers if c['Date'] <= target_date]

def print_simulation_summary(results_df: pd.DataFrame) -> None:
    """Print a summary of simulation results."""
    print("\n" + "="*50)
    print("MONTHLY SIMULATION RESULTS")
    print("="*50)
    print(f"Total Classification Cost: {results_df['classification_cost'].sum():,.0f} MNT")
    print(f"Total Penalty Cost: {results_df['penalty_cost'].sum():,.0f} MNT")
    print(f"Total Cost: {results_df['total_cost'].sum():,.0f} MNT")
    print(f"Total Containers Loaded: {results_df['containers_loaded'].sum():,}")
    print(f"Average Daily Cost: {results_df['total_cost'].mean():,.0f} MNT")
    print(f"Average Containers per Day: {results_df['containers_loaded'].mean():.1f}")
    print(f"Days simulated: {len(results_df)}")

# has got an error
def save_simulation_results(results_df: pd.DataFrame, 
                          start_date: str, 
                          end_date: str,
                          output_dir: Union[str, Path] = None) -> None:
    """Save simulation results to files."""
    if output_dir is None:
        output_dir = Path("simulation_results")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    # Save as CSV
    csv_path = output_dir / f"simulation_results_{start_date}_to_{end_date}.csv"
    results_df.to_csv(csv_path, index=False)
    
    # Save summary stats
    summary_stats = {
        'total_classification_cost': results_df['classification_cost'].sum(),
        'total_penalty_cost': results_df['penalty_cost'].sum(),
        'total_cost': results_df['total_cost'].sum(),
        'total_containers_loaded': results_df['containers_loaded'].sum(),
        'avg_daily_cost': results_df['total_cost'].mean(),
        'avg_containers_per_day': results_df['containers_loaded'].mean()
    }
    
    summary_path = output_dir / f"summary_{start_date}_to_{end_date}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"Results saved to {output_dir}")

def run_daily_simulation(containers: List[dict], 
                        decision_function: Callable,
                        number_of_trains: int = NUMBER_OF_TRAINS,
                        number_of_wagons: int = NUMBER_OF_WAGONS) -> Dict:
    """
    Run simulation for a single day with given containers.
    
    Args:
        containers: List of available containers
        decision_function: Function to decide container loading order
        number_of_trains: Number of trains to generate
        number_of_wagons: Number of wagons per train
    
    Returns:
        Dictionary containing simulation results
    """
    if not containers:
        return {
            'trains': [],
            'total_classification_cost': 0,
            'total_penalty_cost': 0,
            'total_cost': 0,
            'containers_loaded': 0,
            'containers_available': 0,
            'train_sequences': []
        }
    
    # Generate trains
    trains = generate_trains(number_of_trains=number_of_trains,
                           number_of_wagons=number_of_wagons)
    
    # Load containers to trains
    trains = load_containers_to_trains(containers, trains, decision_function)
    
    # Calculate costs
    total_classification_cost = 0
    train_sequences = []
    
    for train in trains:
        train_terminal_sequence = train_to_terminal_sequence(train)
        train_sequences.append(train_terminal_sequence)
        
        horoonii_too = count_transitions(train_terminal_sequence)
        classification_cost = calculate_classification_cost(horoonii_too=horoonii_too)
        total_classification_cost += classification_cost
    
    # Calculate penalty cost
    total_penalty_cost = calculate_penalty_cost(trains)
    
    # Count loaded containers
    containers_loaded = sum(1 for train in trains for wagon in train if wagon['container_id'] is not None)
    
    return {
        'trains': trains,
        'total_classification_cost': total_classification_cost,
        'total_penalty_cost': total_penalty_cost,
        'total_cost': total_classification_cost + total_penalty_cost,
        'containers_loaded': containers_loaded,
        'containers_available': len(containers),
        'train_sequences': train_sequences
    }

def run_monthly_simulation(containers_file_path: Union[str, Path],
                          start_date: str = "2025-06-01",
                          end_date: str = "2025-06-30",
                          decision_function: Callable = None,
                          save_results: bool = True,
                          output_dir: Union[str, Path] = None) -> pd.DataFrame:
    """
    Run simulation for a full month, day by day.
    
    Returns:
        DataFrame with daily simulation results
    """
    # Load data
    containers = load_container_data(containers_file_path)
    
    # Set default decision function
    if decision_function is None:
        def default_decision_function(containers):
            return decision_group_probalistic(containers, prob_same_terminal=PROBABILITY, seed=42)
        decision_function = default_decision_function
    
    # Get date range
    dates = get_date_range(start_date, end_date)
    print(f"Simulating from {start_date} to {end_date} ({len(dates)} days)")
    
    # Run simulation for each day
    daily_results = []
    
    for day_idx, date in enumerate(dates):
        print(f"Simulating day {day_idx + 1}/{len(dates)}: {date}")
        
        # Get containers available up to this date
        available_containers = filter_containers_by_date(containers, date)
        
        # Run daily simulation
        daily_result = run_daily_simulation(available_containers, decision_function)
        
        # Store results in a flat structure
        result_row = {
            'date': date,
            'classification_cost': daily_result['total_classification_cost'],
            'penalty_cost': daily_result['total_penalty_cost'],
            'total_cost': daily_result['total_cost'],
            'containers_loaded': daily_result['containers_loaded'],
            'containers_available': daily_result['containers_available']
        }
        daily_results.append(result_row)
        
        print(f"  Available: {daily_result['containers_available']}, "
              f"Loaded: {daily_result['containers_loaded']}, "
              f"Cost: {daily_result['total_cost']:,.0f} MNT")
    
    # Convert to DataFrame
    results_df = list_to_dataframe(daily_results)
    
    # Save results if requested
    if save_results:
        save_simulation_results(results_df, start_date, end_date, output_dir)
    
    return results_df

def main():
    """Example of how to run the monthly simulation."""
    
    # Set up paths
    simulation_data_dir = Path("simulation_artifacts")
    containers_file = simulation_data_dir / "june_2025_containters_extended_smoothed.xlsx"
    
    # Define decision function
    def decision_function(containers):
        return decision_fifo(decision_group_probalistic(containers, prob_same_terminal=PROBABILITY, seed=42))
    
    # Alternative decision functions you can use:
    # decision_function = decision_random
    # decision_function = decision_group_by_terminal
    # decision_function = decision_optimal
    # decision_function = decision_fifo(decision_optimal)
    
    print("Starting monthly simulation...")
    output_dir = Path("simulation_results")
    # Run the simulation
    results_df = run_monthly_simulation(
        containers_file_path=containers_file,
        start_date="2025-06-01",
        end_date="2025-06-30",
        decision_function=decision_function,
        save_results=False,
        output_dir=output_dir
    )
    
    # Print summary
    print_simulation_summary(results_df)
    print(f"Saved to {output_dir}")

    return results_df.to_csv(output_dir / "results_df.csv")

if __name__ == "__main__":
    # Run the simulation
    results_df = main()



# if __name__ == "__main__":
#     simulation_data_dir = Path("simulation_artifacts")
#     input_dir = Path("input")
#     probability_excel_path = input_dir / "UBmagad.xlsx"

#     date="2025-06-01"
#     # containers = generate_containers(
#     #     probability_excel_path,
#     #     number_of_containers=NUMBER_OF_CONTAINERS,
#     #     date=date
#     # )
#     # For reproducing the results
#     # containers.to_excel(simulation_data / f"containers_{date}.xlsx", index=False)
#     #containers = pd.read_excel(simulation_data_dir / f"containers_{date}.xlsx")
#     #containers = dataframe_to_list(containers)
#     containers = pd.read_excel( simulation_data_dir / "june_2025_containters_extended_smoothed.xlsx", index=False)
#     containers = dataframe_to_list(containers)
#     # Save containers to csv
    
#     # print(containers)
#     # Generate trains
#     trains = generate_trains(number_of_trains=NUMBER_OF_TRAINS,
#                              number_of_wagons=NUMBER_OF_WAGONS)
#     # print(trains)
#     # Save trains to csv
#     # pd.DataFrame(trains).to_csv(simulation_data / "trains.csv", index=False)
    
#     def decision_function(containers):
#         return decision_group_probalistic(containers, prob_same_terminal=PROBABILITY, seed=42)
#     # decision_function = decision_random
#     # decision_function = decision_group_by_terminal
#     # decision_function = decision_optimal

#     trains = load_containers_to_trains(containers, trains, decision_function)
#     print(f"Trains: {[wagon['Терминал'] for wagon in trains[0]]}")
#     print('LENGTH1=',len(trains[0]), "LENGTH2=",len(trains[1]))
#     trains_terminal_sequences = [train_to_terminal_sequence(train) for train in trains]
#     print(trains_terminal_sequences)
#     calculate_penalty_cost(trains_terminal_sequences)
#     # horoonii_too = count_transitions(trains_terminal_sequences)
#     # print('HOROODOLT',horoonii_too)
#     expense_sum = 0.0
#     for train_terminal_sequence in trains_terminal_sequences:
#         horoonii_too = count_transitions(train_terminal_sequence)
#         # horoonii_too_dict[train_terminal_sequence] = horoonii_too
#         classification_cost = calculate_classification_cost(horoonii_too=horoonii_too)
#         print(f"Huruunii zardal: {classification_cost}")
#         expense_sum += classification_cost
#         print('HOROODOLT',horoonii_too)

#     print('expense_sum',expense_sum)

