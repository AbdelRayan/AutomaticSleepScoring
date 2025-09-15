from matplotlib import pyplot as plt
import matplotlib.patches as patches
from utils import get_states_list_ordered_by_freq, build_community_sequence_from_simplified_sequence, assign_colours_to_communities


def draw_simplified_sleep_slab(simplified_states_sequence, xlim=None, custom_title=None):
    total_time = sum(duration for _, duration in simplified_states_sequence)
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.set_xlim(0, total_time)
    ax.set_yticks([])

    states = get_states_list_ordered_by_freq(simplified_states_sequence)
    colour_map = assign_colours_to_communities(states)

    start = 0
    for state, duration in simplified_states_sequence:
        rect = patches.Rectangle((start, 0), duration, 1, facecolor=colour_map[state])
        ax.add_patch(rect)
        start += duration

    handles = [patches.Patch(color=colour_map[state], label=state) for state in states]
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel("Time (epochs)")
    if custom_title is not None:
        plt.title(custom_title)
    else:
        plt.title("Sleep Plot")
    plt.tight_layout()
    if xlim is not None:
        plt.xlim(*xlim)
    plt.show()

    
def draw_clustered_sleep(simplified_states_sequence, node_mapping, add_latent_state_breakdown=False, xlim=None):
    # Step 1: Convert state_sequence into community-segmented blocks
    community_sequence = []
    prev_community = None
    current_duration = 0

    for state, duration in simplified_states_sequence:
        community = node_mapping[state]
        if community != prev_community:
            if prev_community is not None:
                community_sequence.append((prev_community, current_duration))
            current_duration = duration
            prev_community = community
        else:
            current_duration += duration
    community_sequence.append((prev_community, current_duration))  # append last

    communities = get_states_list_ordered_by_freq(community_sequence)
    colour_map = assign_colours_to_communities(communities)

    # Step 3: Set up plot
    total_time = sum(duration for _, duration in simplified_states_sequence)
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.set_xlim(0, total_time)
    ax.set_yticks([])

    # Step 4: Plot community blocks
    start = 0
    for community, duration in community_sequence:
        rect = patches.Rectangle((start, 0), duration, 1, facecolor=colour_map[community])
        ax.add_patch(rect)
        start += duration

    # Step 5: Optionally add dotted lines and state labels
    if add_latent_state_breakdown:
        time_cursor = 0
        for state, duration in simplified_states_sequence:
            mid = time_cursor + duration / 2
            ax.text(mid, .47, str(state), ha='center', va='bottom', fontsize=5)
            time_cursor += duration
            if time_cursor < total_time:
                ax.axvline(time_cursor, color='black', linestyle=':', linewidth=0.8)

    # Step 6: Add legend and plot
    handles = [patches.Patch(color=colour_map[c], label=f'Community {c}') for c in communities]
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel("Time (epochs)")
    plt.title("Clustered Sleep Plot" + (" with Latent State Breakdown" if add_latent_state_breakdown else ""))
    plt.tight_layout()
    if xlim is not None:
        plt.xlim(*xlim)
    plt.show()


def draw_all_sleep_slabs(simplified_states_sequence, sleep_stages, node_community_mapping, manually_labelled_dict=None, xlim=None):
    # Step 1: Build both sequences in one pass
    community_sequence = build_community_sequence_from_simplified_sequence(simplified_states_sequence, node_community_mapping)
    if manually_labelled_dict is not None:
        manual_sequence = build_community_sequence_from_simplified_sequence(simplified_states_sequence, manually_labelled_dict)

    # Step 2: Define colour maps
    communities = get_states_list_ordered_by_freq(community_sequence)
    community_colour_map = assign_colours_to_communities(communities)
    
    labels = get_states_list_ordered_by_freq(sleep_stages)
    label_colour_map = assign_colours_to_communities(labels)

    # Step 3: Set up subplots
    total_time = sum(duration for _, duration in simplified_states_sequence)
    if manually_labelled_dict is None:
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 3.2), sharex=True, height_ratios=[1, 1])
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 4.2), sharex=True, height_ratios=[1, 1, 1])
    fig.subplots_adjust(hspace=0.3)

    # Step 4: Plot the original sleep stages
    start = 0
    for stage, duration in sleep_stages:
        rect = patches.Rectangle((start, 0), duration, 1, facecolor=label_colour_map[stage])
        ax1.add_patch(rect)
        start += duration
    ax1.set_xlim(0, total_time)
    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    ax1.set_title("Original Sleep Labels")
    sleep_stage_handles = [patches.Patch(color=label_colour_map[s], label=s) for s in labels]
    ax1.legend(handles=sleep_stage_handles, bbox_to_anchor=(1.02, 0.5), loc='center left', title='Sleep Stages')
    
    # Step 5: (Optionally) plot manual labels (once you associate one stage to each train_state)
    if manually_labelled_dict is not None:
        start = 0
        for label, duration in manual_sequence:
            rect = patches.Rectangle((start, 0), duration, 1, facecolor=label_colour_map[label])
            ax2.add_patch(rect)
            start += duration
        ax2.set_xlim(0, total_time)
        ax2.set_ylim(0, 1)
        ax2.set_yticks([])
        ax2.set_title("Manual Labels (assign one sleep stage to each latent state)")
        manual_handles = [patches.Patch(color=label_colour_map[l], label=l) for l in labels]
        ax2.legend(handles=manual_handles, bbox_to_anchor=(1.02, 0.5), loc='center left', title='Manual Labels')

    # Step 6: Plot community labels as resulting from clustering algo
    start = 0
    for community, duration in community_sequence:
        rect = patches.Rectangle((start, 0), duration, 1, facecolor=community_colour_map[community])
        ax3.add_patch(rect)
        start += duration
    ax3.set_xlim(0, total_time)
    ax3.set_ylim(0, 1)
    ax3.set_yticks([])
    ax3.set_title("Clustered Communities")
    community_handles = [patches.Patch(color=community_colour_map[c], label=f'Community {c}') for c in communities]
    ax3.legend(handles=community_handles, bbox_to_anchor=(1.02, 0.5), loc='center left', title='Communities')
    
    ax3.set_xlabel("Time (epochs)")

    # Step 7: Optional xlim
    if xlim is not None:
        ax1.set_xlim(*xlim)
        ax2.set_xlim(*xlim)
        ax3.set_xlim(*xlim)

    # Step 8: Plot
    plt.tight_layout()
    plt.subplots_adjust(right=0.78)  # Shrinks plotting area to leave room
    plt.show()