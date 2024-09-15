import re
from pathlib import Path
import yaml

def get_metadata(metaname):
  """
  Parameters
  ----------
  metaname: str
    metaname consists of rat number, study day, condition, treatment value and trial number
    each separated by underscore.
    Example: Rat2_SD4_HC_2_posttrial1

  Returns
  -------
  metadata: dict
   Dictionary object which contains each conditions.
  """
  metadata = {}

  metaname  = metaname.split('_')
  metadata["rat_id"]    = int(metaname[0][3:])
  metadata["study_day"] = int(metaname[1][2:])
  metadata["condition"] = metaname[2]
  metadata["treatment"] = int(metaname[3])
  metadata["trial_num"] = metaname[4][9:]

  return metadata

def create_title(metadata):
    treatment = {0: "Negative CBD", 1: "Positive CBD", 
                 2: "Negative RGS14", 3:"Positive RGS14", 4:"OS basic"}
    
    title = "Rat " + str(metadata["rat_id"])
    title += " Study Day: " + str(metadata["study_day"])
    title += " Treatment: " + treatment[metadata["treatment"]]
    title += " Post-trial: " + str(metadata["trial_num"])
    return title

def create_name_cbd(file, overview_df):
    #pattern for matching the information on the rat
    pattern = r'Rat(\d+)_.*_SD(\d+)_([A-Z]+).*posttrial(\d+)'

    # extract the information from the file path
    match = re.search(pattern, file)
    rat_num = int(match.group(1))
    sd_num = int(match.group(2))
    condition = str(match.group(3))
    posttrial_num = int(match.group(4))

    mask = (overview_df['Rat no.'] == rat_num) & (overview_df['Study Day'] == sd_num) & (overview_df['Condition'] == condition)

    # use boolean indexing to extract the Treatment value
    treatment_value = overview_df.loc[mask, 'Treatment'].values[0]
    
    # Extract the value from the "treatment" column of the matching row
    if treatment_value == 0:
        treatment = '0'
    else:
        treatment = '1'
       
    title_name = 'Rat' + str(rat_num) +'_' + 'SD' + str(sd_num) + '_' + condition + '_' + treatment + '_' + 'posttrial' + str(posttrial_num)
    #RatID,StudyDay,condition,conditionfull, treatment, treatmentfull, posstrial number

    return title_name

def create_name_rgs(fname):
    #pattern for matching the information on the rat
    pattern = r'Rat(\d+)_.*_SD(\d+)_([A-Z]+).*post[\w-]+trial(\d+)'
    
    # extract the information from the file path
    match = re.search(pattern, fname, flags=re.IGNORECASE)
    rat_num = int(match.group(1))
    sd_num = int(match.group(2))
    condition = str(match.group(3))
    posttrial_num = int(match.group(4))

    # Extract the value from the "treatment" column of the matching row
    if (rat_num == 1) or (rat_num == 2) or (rat_num == 6) or (rat_num == 9) :
        treatment = '2'
    else:
        treatment = '3'
    
    title_name = 'Rat' + str(rat_num) +'_' + 'SD' + str(sd_num) + '_' + condition + '_' + treatment + '_' + 'posttrial' + str(posttrial_num)
    #RatID,StudyDay,condition,conditionfull, treatment, treatmentfull, posstrial number

    return title_name

def create_name_os(hpc_fname):
    metadata = str(Path(hpc_fname).parent.parent.name).split("_")
    title = metadata[1] + "_" + metadata[2] + "_" + metadata[3]
    
    pattern = r"post_trial(\d+)"
    match = re.search(pattern, hpc_fname, re.IGNORECASE)
    title += "_4_" + "posttrial" + match.group(1)

    #RatID,StudyDay,condition,conditionfull, treatment, treatmentfull, posstrial number
    return title

def partition_to_4(rem_dict):
    # rem_dict: dictionary with keys as tuples and values as numpy arrays
    keys = sorted(rem_dict.keys())
    partitions = [{} for _ in range(4)]  # Create a list of 4 empty dictionaries

    for rem_idx in keys:
        _, end = rem_idx
        if end < 2700:  # First region
            partitions[0][rem_idx] = rem_dict[rem_idx]
        elif end < 5400:  # Second region
            partitions[1][rem_idx] = rem_dict[rem_idx]
        elif end < 8100:  # Third region
            partitions[2][rem_idx] = rem_dict[rem_idx]
        else:  # Fourth region
            partitions[3][rem_idx] = rem_dict[rem_idx]

    return partitions

def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config