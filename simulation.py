import numpy as np
import pandas as pd

import random
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union, List, Callable, Sequence

# Simulation parameters
NUMBER_OF_CONTAINERS = 110
NUMBER_OF_TRAINS = 2
NUMBER_OF_WAGONS = 55
PROBABILITY = 0.6

LOCOMOTIVE_COST_PER_HOUR = 210000 # MNT
LOCOMOTIVE_COST_PER_MINUTE = LOCOMOTIVE_COST_PER_HOUR / 60  # 3500 MNT per minute

PENALTY_PER_HOUR = 900
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

# TODO: Make the Terminals a class
# class Terminal:
#     def __init__(self, name, distance=0, capacity=0, storage_capacity=0):
#         self.name = name
#         self.distance = distance # minutes
#         self.capacity = capacity # wagons per day
#         self.storage_capacity = storage_capacity # wagons
#         self.wagons = []

#     def is_full(self):
#         return len(self.wagons) >= self.capacity  # just an example

# # Dictionary of terminal objects
# # TERMINALS = {
# #     'УБ МЧ': Terminal('УБ МЧ'),
# #     'Туушин': Terminal('Туушин'),
# #     # ...
# # }

############################## Generation process #############################

def generate_containers(
        probability_excel_path: Union[str, Path], 
        number_of_containers: int, 
        date: str,
        terminal_col: str="Терминал",
        count_col: str="Count"
        ) -> pd.DataFrame:
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



def decision_fifo(decision_fn: Callable[[List[dict]], Sequence[int]]) -> Callable[[List[dict]], Sequence[int]]:
    """
    Returns a wrapped decision function that applies the original decision function
    after sorting containers in FIFO order (by 'Date').
    """
    def wrapped(containers: List[dict]) -> Sequence[int]:
        # Sort containers by date (FIFO order)
        sorted_with_idx = sorted(enumerate(containers), key=lambda x: x[1]['Date'])
        sorted_indices, sorted_containers = zip(*sorted_with_idx) if sorted_with_idx else ([], [])

        # Get order from original decision function
        reordered_indices = decision_fn(list(sorted_containers))

        # Map reordered local indices back to original indices
        return [sorted_indices[i] for i in reordered_indices]
    
    return wrapped

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

def calculate_classification_cost(horoonii_too: int) -> int:

    #  huleelgiin_chingeleg: int = 0, huleelgiin_chingelegiin_time: int = 0, total_terminaluudiin_hureh_zai: int = 318
    horoonii_zardal =horoonii_too = horoonii_too * MINUTES_PER_HOROO * LOCOMOTIVE_COST_PER_MINUTE # tsagt 210k minuted 3.5k
    # huleelgiin_zardal = huleelgiin_chingeleg * huleelgiin_chingelegiin_time * PENALTY_PER_MINUTE # minutaar
    # terminal_hurgeh_zardal = total_terminaluudiin_hureh_zai * LOCOMOTIVE_COST_PER_MINUTE
    # niit_zardal = horoonii_zardal + huleelgiin_zardal + terminal_hurgeh_zardal
    return horoonii_zardal



# def load_containers_to_trains_(containers: pd.DataFrame,
#                               trains: List[List[dict]],
#                               decision_function: Callable) -> None:
#     """
#     Loads containers onto trains in-place. Each train is filled up to its capacity,
#     and loaded containers are removed from the available pool for subsequent trains.
#     """
#     pass
##################### HELPER FUNCTIONS #####################

def calculate_penalty_cost(train_sequences: List[int]) -> float:
    unique_terminals = []
    for train_sequence in train_sequences:
        unique_terminals.extend(set(train_sequence))
    
    terminal_counts = {terminal: 0 for terminal in unique_terminals}
    for terminal in train_sequence:
        terminal_counts[terminal] += 1
    for terminal, count in terminal_counts.items():
        if count > 1:
            print(f"Terminal {terminal} appears {count} times")
    # TODO: For
    
def dataframe_to_list(df: pd.DataFrame):
    return df.to_dict(orient="records")

#################### END-HELPER FUNCTIONS ###################

def step_():
    pass

if __name__ == "__main__":
    simulation_data = Path("simulation_artifacts")
    input_dir = Path("input")
    probability_excel_path = input_dir / "UBmagad.xlsx"

    date="2025-06-01"
    # containers = generate_containers(
    #     probability_excel_path,
    #     number_of_containers=NUMBER_OF_CONTAINERS,
    #     date=date
    # )
    # For reproducing the results
    # containers.to_excel(simulation_data / f"containers_{date}.xlsx", index=False)
    containers = pd.read_excel(simulation_data / f"containers_{date}.xlsx")
    containers = dataframe_to_list(containers)
    # Save containers to csv
    
    # print(containers)
    # Generate trains
    trains = generate_trains(number_of_trains=NUMBER_OF_TRAINS,
                             number_of_wagons=NUMBER_OF_WAGONS)
    # print(trains)
    # Save trains to csv
    # pd.DataFrame(trains).to_csv(simulation_data / "trains.csv", index=False)
    
    def decision_function(containers):
        return decision_group_probalistic(containers, prob_same_terminal=PROBABILITY, seed=42)
    # decision_function = decision_random
    # decision_function = decision_group_by_terminal
    # decision_function = decision_optimal

    trains = load_containers_to_trains(containers, trains, decision_function)
    print(f"Trains: {[wagon['Терминал'] for wagon in trains[0]]}")
    print('LENGTH1=',len(trains[0]), "LENGTH2=",len(trains[1]))
    trains_terminal_sequences = [train_to_terminal_sequence(train) for train in trains]
    print(trains_terminal_sequences)
    calculate_penalty_cost(trains_terminal_sequences)
    # horoonii_too = count_transitions(trains_terminal_sequences)
    # print('HOROODOLT',horoonii_too)
    expense_sum = 0.0
    for train_terminal_sequence in trains_terminal_sequences:
        horoonii_too = count_transitions(train_terminal_sequence)
        # horoonii_too_dict[train_terminal_sequence] = horoonii_too
        classification_cost = calculate_classification_cost(horoonii_too=horoonii_too)
        print(f"Huruunii zardal: {classification_cost}")
        expense_sum += classification_cost
        print('HOROODOLT',horoonii_too)

    print('expense_sum',expense_sum)

