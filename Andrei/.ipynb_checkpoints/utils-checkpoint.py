from matplotlib import pyplot as plt
import networkx as nx


### We start with functions used to extract data from the df

def get_states_sequence_from_df(df, column='train_state'):
    assert column in df.columns, f'Unrecognised column name. The available columns are {list(df.columns)}'
    latent_states_sequence = df[column].tolist()
    return latent_states_sequence


def get_state_sleep_stage_relationship(df, state_col='train_state'):
    """Get a sleep stage frequency dictionary for each train state"""
    grouped = df.groupby(state_col)['sleep_stage'].value_counts()
    sleep_stage_mapping_frequency = grouped.unstack(fill_value=0).to_dict(orient='index')
    for state, freq in sleep_stage_mapping_frequency.items():
        sleep_stage_mapping_frequency[state] = dict(sorted(freq.items(), key=lambda item: item[1], reverse=True))  # to order each dict by frequency
    return sleep_stage_mapping_frequency


def get_manual_sleep_stage_mapping(df, state_col='train_state'):
    """Return a dictionary with the latent states as keys and the sleep stages assigned to them most often as values."""
    nested_dict = get_state_sleep_stage_relationship(df, state_col=state_col)
    return {state: max(stages, key=stages.get) for state, stages in nested_dict.items()}


def get_activation_pattern_train_state_mapping(df, key='train_state', val='binary_pattern'):
    mapping = dict(zip(df[key], df[val]))
    val_from_mapping = df[key].map(mapping)
    assert val_from_mapping.equals(df[val]), 'There is something wrong with the df mapping between the 2 columns'
    mapping = dict(sorted(mapping.items(), key=lambda item: item[0]))
    return mapping


### Functions to get the raw data in a useful format (simplified states sequence, transition dict etc.)

def simplify_states_sequence(latent_states_sequence):
    """Take a raw states sequence (as in the rat files) and simplify it by counting how often the same state appears in a subsequence.
    The result will be a list where each element is a tuple with the first entry being a state and the second being a count.
    e.g., the sequence 22 → 71 → 71 → 71 → 15 → 15 → 2 becomes (22, 1) → (71, 3) → (15, 2) → (2, 1)"""
    prev_state = latent_states_sequence[0]
    duration = 1
    simplified_states_sequence = []
    for next_state in latent_states_sequence[1:]:
        if next_state == prev_state:
            duration += 1
        else:
            simplified_states_sequence.append((prev_state, duration))
            prev_state = next_state
            duration = 1
    simplified_states_sequence.append((prev_state, duration))
    assert sum(x[1] for x in simplified_states_sequence) == len(latent_states_sequence), 'States count in the simplified sequence is incorrect'
    return simplified_states_sequence


def build_transition_dict(simplified_states_sequence, include_self_loops=False):
    """Take a simplified states sequence and build a dictionary for the state (and associated frequency) transitions.
    This dictionary will contain no cycles by default (as we are working on the simplified sequence).
    The count corresponds to how often a state transitions into another."""
    transition_dict = {}
    for idx in range(len(simplified_states_sequence) - 1):
        cur_state, next_state = simplified_states_sequence[idx][0], simplified_states_sequence[idx + 1][0]
        if cur_state in transition_dict:
            transition_dict[cur_state][next_state] = transition_dict[cur_state].get(next_state, 0) + 1
        else:
            transition_dict[cur_state] = {next_state: 1}

    if include_self_loops:  # then we also add how often self transitions happen (inferred from the state duration)
        for state, duration in simplified_states_sequence:
            if duration > 1:  # then there is at least 1 self-transition
                if state in transition_dict:
                    transition_dict[state][state] = transition_dict[state].get(state, 0) + duration - 1
                else:
                    transition_dict[state] = {state: duration - 1}
    for node, transitions in transition_dict.items():  # to also sort the transitions based on frequency
        transition_dict[node] = dict(sorted(transitions.items(), key=lambda item: item[1], reverse=True))
    return transition_dict


def build_transition_graph(transition_dict, min_edge_frequency=None, use_raw_edge_weights=False):
    G = nx.DiGraph()
    for src, targets in transition_dict.items():
        out_degree = 1 if use_raw_edge_weights else sum(targets.values())
        for dst, weight in targets.items():
            if min_edge_frequency is None or weight >= min_edge_frequency:  # filter out infrequent edges if the parameter is provided
                G.add_edge(src, dst, weight=weight / out_degree)
    return G

### Various utility functions to help with analysis/putting the data in a different format

def get_state_frequency(simplified_states_sequence, transform_to_ratios=False):
    """Return a dictionary with the frequency of each state"""
    state_frequency = {}
    for (state, duration) in simplified_states_sequence:
        state_frequency[state] = state_frequency.get(state, 0) + duration
    state_frequency = dict(sorted(state_frequency.items(), key=lambda item: item[1], reverse=True))
    assert sum(x[1] for x in simplified_states_sequence) == sum(state_frequency.values())
    if transform_to_ratios:
        state_frequency = {state: round(duration / sum(state_frequency.values()), 3) for state, duration in state_frequency.items()}
    return state_frequency

def get_states_list_ordered_by_freq(simplified_states_sequence, reverse=True):
    state_frequency = get_state_frequency(simplified_states_sequence)
    states_ordered_by_freq = sorted([(key, val) for key, val in state_frequency.items()], key=lambda x: x[1], reverse=reverse)
    return [x[0] for x in states_ordered_by_freq]

def build_community_sequence_from_simplified_sequence(simplified_states_sequence, node_mapping):
    result = []
    prev = None
    current_duration = 0
    for state, duration in simplified_states_sequence:
        mapped = node_mapping[state]
        if mapped != prev:
            if prev is not None:
                result.append((prev, current_duration))
            current_duration = duration
            prev = mapped
        else:
            current_duration += duration
    result.append((prev, current_duration))
    return result

def get_communities_mapping_from_node_mapping(node_community_mapping, print_result=False):
    """Transform a mapping from {node: community} to {community: nodes}"""
    communities_dict = {}
    for node, com in node_community_mapping.items():
        communities_dict.setdefault(com, []).append(node)
    for com in communities_dict:
        communities_dict[com].sort()
    if print_result:
        foo = [(com, nodes) for com, nodes in communities_dict.items()]
        foo.sort(key=lambda x: x[0])
        for com, nodes in foo:
            print(f'{com}: {nodes}')
    return communities_dict
    
def resimplify_simplified_states_sequence(simplified_states_sequence):
    """Resimplify a states sequence where consecutive entries might be on the same node (following a relabelling)
    e.g., (7, 2), (7, 3) becomes (7, 5)"""
    result = []
    prev = None
    current_duration = 0
    for state, duration in simplified_states_sequence:
        if state != prev:
            if prev is not None:
                result.append((prev, current_duration))
            current_duration = duration
            prev = state
        else:
            current_duration += duration
    result.append((prev, current_duration))
    assert sum(x[1] for x in result) == sum(x[1] for x in simplified_states_sequence)
    return result

def compute_distance_between_states(state1, state2, state_to_pattern_dict):
    """Compute the Hamming distance between the activation patterns corresponding to 2 states"""
    return sum(a != b for a, b in zip(state_to_pattern_dict[state1], state_to_pattern_dict[state2]))

def produce_exit_pattern(node_community, transition_dict, transform_to_ratios=False):
    """Return 2 dictionaries: one for exit nodes inside the community and one for entry nodes outside the community.
    This function looks at random walks exiting the community and remembering the associated nodes."""
    exit_dict, entry_dict = {}, {}
    for node in node_community:
        for transition_node, freq in transition_dict[node].items():
            if transition_node not in node_community:  # then the random walk exits the community through this node
                exit_dict[node] = exit_dict.get(node, 0) + freq
                entry_dict[transition_node] = entry_dict.get(transition_node, 0) + freq
    
    exit_dict = dict(sorted(exit_dict.items(), key=lambda item: item[1], reverse=True))
    entry_dict = dict(sorted(entry_dict.items(), key=lambda item: item[1], reverse=True))
    if transform_to_ratios:
        exit_dict = {node: freq / sum(exit_dict.values()) for node, freq in exit_dict.items()}
        entry_dict = {node: freq / sum(entry_dict.values()) for node, freq in entry_dict.items()}
    return exit_dict, entry_dict


def assign_colours_to_communities(communities: list):
    """Return a dictionary with community numbers as keys and colours as values"""
    assert len(communities) <= 20, f'At most 20 colours are accepted'

    # This is a hacky way to make sure Wake, NREM, REM always appear in this order 
    target_values = ['Wake', 'NREM', 'REM']
    communities = [x for x in target_values if x in communities] + [x for x in communities if x not in target_values]
    
    if len(communities) <= 10:
        colours = plt.colormaps['tab10']  # tab10_r
    else:
        colours = plt.colormaps['tab20']
    return {com:colours(idx) for idx, com in enumerate(communities)}