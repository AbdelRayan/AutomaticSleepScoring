
# Input / Output of the Analyses

## SleepInvestigatoR Analysis

### Input
- **File**: `string_analysis_hypno_latencies_tonic_phasic.xlsx`
- **Contents**: Sleep sequences and other information about the recordings.

### Preprocessing
- **Script**: `preprocessing_sleep_sequences.R`
- **Function**: Preprocesses the input file and creates separate files for each REM-included sequence.
- **Output**: Files are saved in a designated folder (modifiable in the script).

### Main Analysis
- **Tool**: `SleepInvestigatoR`
- **Steps**:
  1. Open the folder containing the preprocessed REM sequence files.
  2. Process each file individually.
  3. Compile features for every sequence into a table.
  4. Save the table as `SleepInvestigatoR_output_table.xlsx`.

### Visualization
- **Notebook**: `SleepInvestigatoR_data_plotting.ipynb`
- **Input**: `SleepInvestigatoR_output_table.xlsx`
- **Function**: Visualizes the desired features.

---

## Markov Chain Analyses

### Input
- **File**: `string_analysis_hypno_latencies_tonic_phasic.xlsx`

### Scripts
- `First-order_markov_chain.ipynb`
- `Second-order_markov_chain.ipynb`

### Notes
- Each script reads the input file twice:
  - Once for REM sequences.
  - Once for Phasic, Tonic, and Intermediate sequences.
- **Important**: Filepaths must be updated accordingly.

---

## Calling SleepInvestigatoR

```r
SleepInvestigatoR(
    FileNames, 
    file.type = "csv",
    epoch.size = 1, 
    max.hours = 0.75, 
    byBlocks = NULL, 
    byTotal = TRUE, 
    score.checker = F,
    id.factor = TRUE, 
    group.factor = TRUE, 
    normalization = NULL, 
    time.stamp = F,                   
    scores.column = 1, 
    lights.on = NULL,
    lights.off = NULL,
    time.format = "hh:mm:ss", 
    date.format = 'm/d/y',                              
    date.time.separator = '', 
    score.value.changer = FALSE, 
    sleep.adjust = "NREM.Onset", 
    NREM.cutoff = 60,                            
    REM.cutoff = 60, 
    Wake.cutoff = 60, 
    Sleep.cutoff = 60,
    Cycle.cutoff = 120, 
    Propensity.cutoff = 60,
    data.format = "long", 
    save.name = "C:/Users/woutp/OneDrive/Donders_instituut/SleepInvestigatoR_output/"
)
