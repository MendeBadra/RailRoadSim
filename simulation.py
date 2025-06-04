import numpy as np
import pandas as pd

import random
from collections import Counter
from pathlib import Path
from typing import Union, List, Callable, Sequence

# Simulation parameters
NUMBER_OF_CONTAINERS = 110
NUMBER_OF_TRAINS = 2
NUMBER_OF_WAGONS = 55
PROBABILITY = 0.7

LOCOMOTIVE_COST_PER_HOUR = 210000 # MNT
LOCOMOTIVE_COST_PER_MINUTE = LOCOMOTIVE_COST_PER_HOUR / 60  # 3500 MNT per minute

PENALTY_PER_HOUR = 900
PENALTY_PER_MINUTE = PENALTY_PER_HOUR / 60 # 15 MNT per minute

MINUTES_PER_HOROO = 10 # 10 minutes to do one classification

STATIONS = ['УБ', 'Ам', 'То'] # redundant

TERMINALS = {
    'УБ МЧ': {'distance': 8, 'capacity': 18, 'station': 'УБ'},
    'Туушин': {'distance': 20, 'capacity': 8, 'station': 'УБ'},
    'Монгол экс': {'distance': 20, 'capacity': 13, 'station': 'УБ'},
    'материалын импекс': {'distance': 40, 'capacity': 39, 'station': 'УБ'},
    'Прогресс': {'distance': 60, 'capacity': 14, 'station': 'УБ'},
    'Эрин': {'distance': 60, 'capacity': 46, 'station': 'УБ'},
    'Техник импорт': {'distance': 40, 'capacity': 16, 'station': 'УБ'},
    'Амгалан': {'distance': 15, 'capacity': 37, 'station': 'Ам'},
    'Интердэс': {'distance': 20, 'capacity': 15, 'station': 'То'},
    'Монгол тр': {'distance': 10, 'capacity': 15, 'station': 'То'},
    'Толгойт М': {'distance': 5, 'capacity': 8, 'station': 'То'},
    'Номин Тр': {'distance': 20, 'capacity': 8, 'station': 'То'},
}


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



def generate_containers(
        probability_excel_path: Union[str, Path], 
        number_of_containers: int, 
        date: str) -> pd.DataFrame:
    """
    Generate containers based on probability distribution from excel file.
    """
    probability_df = pd.read_excel(probability_excel_path)
    # Extract probabilities from the "Хувь" column
    probabilities = probability_df["Хувь"].values
    
    # Normalize probabilities to sum to 1
    probabilities = probabilities / probabilities.sum()
    
    # Get terminal names
    terminals = probability_df["Терминал"].values
    
    # Generate random terminal assignments based on probabilities
    container_terminals = np.random.choice(
        terminals,
        size=number_of_containers,
        p=probabilities
    )
    
    # Create DataFrame with container IDs and terminals
    containers_df = pd.DataFrame({
        'container_id': range(number_of_containers),
        'Терминал': container_terminals
    })

    containers_df["Date"] = date
    
    return containers_df

def generate_train(
          number_of_wagons: int=40
          ) -> List[dict]:
    """Generate trains, which is an empty container for wagons where each wagon has a container."""
    empty_container = [
        {'container_id': None, 'Терминал': None}
        for _ in range(number_of_wagons)
    ]
    return empty_container

def generate_trains(
        number_of_trains: int,
        number_of_wagons: int
        ) -> List[List[dict]]:
    """Generate trains, which is an empty container for wagons where each wagon has a container."""
    return [generate_train(number_of_wagons) for _ in range(number_of_trains)]




def decision_to_load_containers_(
        containers: pd.DataFrame, 
        train: List[dict], 
        decision_function: Callable) -> List[dict]:
    """
    Fill the train (list of dicts) with containers according to the decision_function.
    The train is a fixed-size list of dicts with 'container_id' and 'Терминал' fields.
    Warning: Mutates the train 
    """
    # random.shuffle(containers)
    ordered_indices = decision_function(containers)
    # Only fill up to the train's capacity
    for i, idx in enumerate(ordered_indices[:len(train)]):
        train[i]['container_id'] = containers.loc[idx, 'container_id']
        train[i]['Терминал'] = containers.loc[idx, 'Терминал']
    return train

def decision_random(containers: pd.DataFrame) -> Sequence[int]:
    """
    Randomly shuffle the containers and return their indices.
    """
    indices = containers.index.to_numpy()
    np.random.shuffle(indices)
    return indices

def decision_group_by_terminal(containers: pd.DataFrame) -> Sequence[int]:
    """
    Group containers by 'Терминал' and return their indices in grouped order.
    """
    return containers.sort_values('Терминал').index.to_numpy()

def decision_group_probalistic(containers: pd.DataFrame, 
                               prob_same_terminal: float = 0.5,
                               seed: int = 42) -> Sequence[int]:
    """
    Reorder containers such that adjacent ones are likely to share the same terminal,
    using a Markov-style probability model.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Count how many containers are at each terminal
    terminal_counts = Counter(containers["Терминал"])
    container_indices_by_terminal = {
        terminal: containers[containers["Терминал"] == terminal].index.tolist()
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

def count_out(horoonii_too: int, huleelgiin_chingeleg: int = 0, huleelgiin_chingelegiin_time: int = 0, total_terminaluudiin_hureh_zai: int = 318) -> int:
    horoonii_zardal =horoonii_too = horoonii_too * MINUTES_PER_HOROO * LOCOMOTIVE_COST_PER_MINUTE # tsagt 210k minuted 3.5k
    huleelgiin_zardal = huleelgiin_chingeleg * huleelgiin_chingelegiin_time * PENALTY_PER_MINUTE # minutaar
    terminal_hurgeh_zardal = total_terminaluudiin_hureh_zai * LOCOMOTIVE_COST_PER_MINUTE
    niit_zardal = horoonii_zardal + huleelgiin_zardal + terminal_hurgeh_zardal
    return niit_zardal

def load_containers_to_trains(containers: pd.DataFrame,
                               trains: List[List[dict]], 
                               decision_function: Callable) -> None:
    """
    Loads containers onto trains in-place. Each train is filled up to its capacity,
    and loaded containers are removed from the available pool for subsequent trains.
    """
    available_indices = set(containers.index)
    for train in trains:
        # Only consider containers that haven't been loaded yet
        available_containers = containers.loc[list(available_indices)]
        ordered_indices = decision_function(available_containers)
        # Select up to the train's capacity
        selected_indices = ordered_indices[:len(train)]
        for i, idx in enumerate(selected_indices):
            train[i]['container_id'] = containers.loc[idx, 'container_id']
            train[i]['Терминал'] = containers.loc[idx, 'Терминал']
        # Remove loaded containers from the pool
        available_indices -= set(selected_indices)
        # If no more containers, break
        if not available_indices:
            break





def step_():
    pass


if __name__ == "__main__":
    simulation_data = Path("simulation_artifacts")
    input_dir = Path("input")
    probability_excel_path = input_dir / "UBmagad.xlsx"
    containers = generate_containers(
        probability_excel_path,
        number_of_containers=NUMBER_OF_CONTAINERS,
        date="2025-01-01"
    )
    # For reproducing the results
    # containers.to_excel(simulation_data / "containers.xlsx", index=False)
    containers = pd.read_excel(simulation_data / "containers.xlsx")
    # Save containers to csv
    
    # print(containers)
    # Generate trains
    trains = generate_trains(number_of_trains=NUMBER_OF_TRAINS,
                             number_of_wagons=NUMBER_OF_WAGONS)
    # print(trains)
    # Save trains to csv
    # pd.DataFrame(trains).to_csv(simulation_data / "trains.csv", index=False)
    
    decision_function = lambda containers: decision_group_probalistic(containers,
                                                                      prob_same_terminal=PROBABILITY,
                                                                      seed=42)
    # decision_function = decision_random
    # decision_function = decision_group_by_terminal

    load_containers_to_trains(containers, trains, decision_function)
    print(f"Trains: {[wagon['Терминал'] for wagon in trains[0]]}")
    print('LENGTH=',NUMBER_OF_CONTAINERS)
    trains_terminal_sequences = [train_to_terminal_sequence(train) for train in trains]
    print(trains_terminal_sequences)
    # horoonii_too = count_transitions(trains_terminal_sequences)
    # print('HOROODOLT',horoonii_too)
    expense_sum = 0.0
    for train_terminal_sequence in trains_terminal_sequences:
        horoonii_too = count_transitions(train_terminal_sequence)
        # horoonii_too_dict[train_terminal_sequence] = horoonii_too
        expense_sum += count_out(horoonii_too=horoonii_too)
        print('HOROODOLT',horoonii_too)

    print('expense_sum',expense_sum)
