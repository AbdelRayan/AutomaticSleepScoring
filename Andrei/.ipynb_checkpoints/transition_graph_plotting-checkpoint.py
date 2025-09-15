import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Wedge
import networkx as nx
from utils import (build_transition_dict, build_transition_graph, assign_colours_to_communities, build_community_sequence_from_simplified_sequence, get_communities_mapping_from_node_mapping, compute_distance_between_states, get_state_frequency, get_activation_pattern_train_state_mapping, produce_exit_pattern)

### These are functions used for plotting the (simple) transition graphs

def get_node_sizes(simplified_states_sequence, min_node_size=150, max_node_size=1500):
    """Return a dictionary for the size of each node in the graph"""
    #todo should this use the state frequency or the node frequency?
    #todo make it work with other types of states/nodes as well
    state_frequency = get_state_frequency(simplified_states_sequence)
    min_freq = min(state_frequency.values())
    max_freq = max(state_frequency.values())
    node_sizes = {node: (max_node_size - min_node_size) * ((freq - min_freq) / (max_freq - min_freq)) + min_node_size for node, freq in state_frequency.items()}
    return node_sizes


def draw_transition_graph(simplified_states_sequence, community_mapping=None, min_edge_frequency=None, draw_raw_edge_weights=False, edge_weight_font_size=7, node_label_mapping=None, custom_title=None, seed=None):
    transition_dict = build_transition_dict(simplified_states_sequence)
    G = build_transition_graph(transition_dict, min_edge_frequency=min_edge_frequency)
    pos = nx.spring_layout(G, seed=seed)  # 43
    
    plt.figure(figsize=(15, 10))

    node_sizes_dict = get_node_sizes(simplified_states_sequence)
    node_sizes = [node_sizes_dict[node] for node in G.nodes()]
    
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

    if community_mapping is None:
        communities = sorted(set(x[0] for x in simplified_states_sequence))
        community_to_colour = assign_colours_to_communities(communities)
        node_colours = [community_to_colour[node] for node in G.nodes()]
    else:
        communities = sorted(set(community_mapping.values()))
        community_to_colour = assign_colours_to_communities(communities)
        node_colours = [community_to_colour[community_mapping[node]] for node in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colours, node_size=node_sizes)

    nx.draw_networkx_edges(
        G, pos,
        edgelist=G.edges(),
        edge_color='gray',
        width=[w * 3 for w in edge_weights],
        arrows=True,
        connectionstyle='arc3,rad=0.1'
    )
    if draw_raw_edge_weights:
        G_raw = build_transition_graph(transition_dict=transition_dict, use_raw_edge_weights=True, min_edge_frequency=min_edge_frequency)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): int(G_raw[u][v]['weight']) for u, v in G.edges()}, font_size=edge_weight_font_size, label_pos=0.2, rotate=False)

    if node_label_mapping:
        labels = {node: node_label_mapping.get(node, node) for node in G.nodes()}
    else:
        labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels)

    legend_handles = [patches.Patch(color=color, label=community) for community, color in community_to_colour.items()]
    plt.legend(handles=legend_handles, title="Communities", loc="upper right")

    if custom_title is not None:
        title = custom_title
    else:
        title = 'Transition Graph of Latent States'
    if min_edge_frequency is not None:
        title += f' (min edge frequency {min_edge_frequency})'
    plt.title(title)
    plt.axis('off')
    plt.show()


def draw_community_transition_graph(simplified_states_sequence, community_mapping, draw_raw_edge_weights=True, seed=None):
    simplified_community_sequence = build_community_sequence_from_simplified_sequence(simplified_states_sequence, community_mapping)
    draw_transition_graph(simplified_community_sequence, draw_raw_edge_weights=draw_raw_edge_weights, edge_weight_font_size=12, seed=seed)


### Here we deal with plotting stuff with pie chart nodes (to accurately represent the human labelling)

def get_circular_positions(nodes, radius=0.1):
    angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
    return {node: [radius * np.cos(a), radius * np.sin(a)] for node, a in zip(nodes, angles)}


def draw_pie_node(ax, center, radius, proportions, colours):
    """Draw a pie chart node at a given center."""
    start_angle = 0
    for frac, colour in zip(proportions, colours):
        if frac == 0:
            continue
        end_angle = start_angle + frac * 360
        wedge = Wedge(center, radius, start_angle, end_angle, facecolor=colour, edgecolor='none', linewidth=0)
        ax.add_patch(wedge)
        start_angle = end_angle


def draw_transition_graph_with_pie_chart_nodes(simplified_states_sequence, sleep_stage_mapping_frequency, node_community=None, min_edge_frequency=None,
                                               draw_raw_edge_weights=False, edge_weight_font_size=7, node_label_mapping=None, 
                                               custom_title=None, node_size_adjustment_factor=800, seed=None):
    transition_dict = build_transition_dict(simplified_states_sequence)
    if node_community is not None:  # then we want to only draw the exit from a specific community
        transition_dict = {node: transition_dict[node] for node in node_community}
    G = build_transition_graph(transition_dict, min_edge_frequency=min_edge_frequency)

    if node_community is not None:
        exit_dict, entry_dict = produce_exit_pattern(node_community=node_community, transition_dict=transition_dict)
        exit_positions = get_circular_positions(list(exit_dict), radius=3)
        entry_positions = get_circular_positions(list(entry_dict), radius=5)
        central_positions = get_circular_positions([node for node in node_community if node not in exit_positions], radius=1)
        pos = nx.spring_layout(G, seed=seed, pos=central_positions|exit_positions|entry_positions, fixed=list(node_community)+list(entry_positions))
    else:
        pos = nx.spring_layout(G, seed=seed)
    
    plt.figure(figsize=(15, 10))
    plt.axis('equal')
    ax = plt.gca()

    node_sizes_dict = get_node_sizes(simplified_states_sequence)
    node_sizes = [node_sizes_dict[node] for node in G.nodes()]
    
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

    sleep_stages = list({k for inner in sleep_stage_mapping_frequency.values() for k in inner})
    community_to_colour = assign_colours_to_communities(sleep_stages)

    for node in G.nodes():
        centre = pos[node]
        size = node_sizes_dict[node] ** 0.5 / node_size_adjustment_factor

        freqs = sleep_stage_mapping_frequency[node]
        total = sum(freqs.values())
        proportions = [v / total for v in freqs.values()]
        labels = list(freqs.keys())
    
        colours = [community_to_colour[label] for label in labels] 
    
        draw_pie_node(ax, centre, radius=size, proportions=proportions, colours=colours)

    nx.draw_networkx_edges(
        G, pos,
        edgelist=G.edges(),
        edge_color='gray',
        width=[w * 3 for w in edge_weights],
        arrows=True,
        connectionstyle='arc3,rad=0.1'
    )
    
    if draw_raw_edge_weights:
        G_raw = build_transition_graph(transition_dict=transition_dict, use_raw_edge_weights=True, min_edge_frequency=min_edge_frequency)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): int(G_raw[u][v]['weight']) for u, v in G.edges()}, font_size=edge_weight_font_size, label_pos=0.2, rotate=False)

    if node_label_mapping:
        labels = {node: node_label_mapping.get(node, node) for node in G.nodes()}
    else:
        labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels)

    legend_handles = [patches.Patch(color=color, label=community) for community, color in community_to_colour.items()]
    plt.legend(handles=legend_handles, title="Communities", loc="upper right")

    if custom_title is not None:
        title = custom_title
    elif node_community is not None:
        title = 'Node Community Exit Pattern'
    else:
        title = 'Transition Graph of Latent States'
    if min_edge_frequency is not None:
        title += f' (min edge frequency {min_edge_frequency})'
    plt.title(title)
    plt.axis('off')
    plt.show()


def draw_community_pie_chart_nodes(node_community_mapping, sleep_stage_mapping_frequency, state_frequency=None, pie_radius=0.25):
    """I vibe coded this one. It will not respect actual pie chart sizes, but will create a clear division into communities"""
    communities_dict = get_communities_mapping_from_node_mapping(node_community_mapping)
    sleep_stages = list({k for inner in sleep_stage_mapping_frequency.values() for k in inner})
    
    fig, ax = plt.subplots(figsize=(12, len(communities_dict) * 1.5))
    ax.axis("off")
    ax.set_aspect('equal')

    y_positions = np.arange(len(communities_dict), 0, -1)
    stage_to_colour = assign_colours_to_communities(sleep_stages)

    max_width = 0  # track max row width

    for i, (community, nodes) in enumerate(communities_dict.items()):
        y = y_positions[i]
        ax.text(-0.5, y, f"Community {community}", va="center", ha="right", fontsize=12)

        current_x = 0
        for node in nodes:
            freqs = sleep_stage_mapping_frequency.get(node)
            if not freqs:
                continue

            total = sum(freqs.values())
            proportions = [v / total for v in freqs.values()]
            labels = list(freqs.keys())
            colours = [stage_to_colour[label] for label in labels]

            draw_pie_node(ax, (current_x, y), pie_radius, proportions, colours)
            ax.text(current_x, y, str(node), ha="center", va="center", fontsize=8, color="black")
            if state_frequency is not None:
                ax.text(current_x, y - pie_radius - 0.05, str(state_frequency[node]), ha="center", va="top", fontsize=7, color="gray")

            current_x += pie_radius * 2.5  # horizontal spacing

        max_width = max(max_width, current_x)

    ax.set_xlim(-1, max_width)
    ax.set_ylim(0, len(communities_dict) + 1)
    legend_handles = [patches.Patch(color=color, label=stage) for stage, color in stage_to_colour.items()]
    plt.legend(handles=legend_handles, title="Communities", loc="center right")
    plt.tight_layout()
    plt.show()


def draw_activation_pattern_difference_with_pie_chart_nodes(simplified_states_sequence, sleep_stage_mapping_frequency,
                                                            nodes_subset=None, min_edge_frequency=None,
                                                            custom_title=None, node_size_adjustment_factor=800, edge_weight_font_size=7, seed=None):
    transition_dict = build_transition_dict(simplified_states_sequence)
    if nodes_subset is not None:
        transition_dict = {key: val for key, val in transition_dict.items() if key in nodes_subset}
        for node, transitions in transition_dict.items():
            transition_dict[node] = {key: val for key, val in transitions.items() if key in nodes_subset}
    print(transition_dict)
    
    G = build_transition_graph(transition_dict, min_edge_frequency=min_edge_frequency)

    pos = nx.spring_layout(G, seed=seed)
    
    plt.figure(figsize=(15, 10))
    plt.axis('equal')
    ax = plt.gca()

    node_sizes_dict = get_node_sizes(simplified_states_sequence)
    node_sizes = [node_sizes_dict[node] for node in G.nodes()]
    
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

    sleep_stages = list({k for inner in sleep_stage_mapping_frequency.values() for k in inner})
    community_to_colour = assign_colours_to_communities(sleep_stages)

    for node in G.nodes():
        centre = pos[node]
        size = node_sizes_dict[node] ** 0.5 / node_size_adjustment_factor

        freqs = sleep_stage_mapping_frequency[node]
        total = sum(freqs.values())
        proportions = [v / total for v in freqs.values()]
        labels = list(freqs.keys())
    
        colours = [community_to_colour[label] for label in labels] 
    
        draw_pie_node(ax, centre, radius=size, proportions=proportions, colours=colours)

    nx.draw_networkx_edges(
        G, pos,
        edgelist=G.edges(),
        edge_color='gray',
        width=[w * 3 for w in edge_weights],
        arrows=True,
        connectionstyle='arc3,rad=0.1'
    )

    state_to_pattern_dict = get_activation_pattern_train_state_mapping(df)  # todo you can precompute this below whole thing if latency becomes an issue
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): compute_distance_between_states(u, v, state_to_pattern_dict=state_to_pattern_dict) for u, v in G.edges()}, font_size=edge_weight_font_size, rotate=False)

    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels)

    legend_handles = [patches.Patch(color=color, label=community) for community, color in community_to_colour.items()]
    plt.legend(handles=legend_handles, title="Communities", loc="upper right")

    if custom_title is not None:
        plt.title(custom_title)
    else:
        plt.title('Transition Graph of Latent States')
    plt.axis('off')
    plt.show()