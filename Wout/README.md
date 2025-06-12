# Cannabidiol and Regulators of G protein Signaling effect on the sleep architecture in rats


## Flowchart dataflow
![SleepInvestigatoR Pipeline](data_flow_flowchart.png)

*Flowchart illustrating the data flow and script execution order. Blue elements represent input folders or files, while yellow elements denote scripts. The branches correspond to different analyses: the left branch shows first-order Markov chains, the middle represents the SleepInvestigatoR analysis, and the right branch depicts second-order Markov chains.*


## SleepInvestigatoR Analysis

### Preprocessing
- **Performed by:** `preprocessing_sleep_sequences.R`
  - **Input:**  
    - `string_analysis_hypno_latencies_tonic_phasic.xlsx`
    - Contains sleep sequences and other recording information.
  - **Functionality:**
    - Preprocesses the data.
    - Creates a separate file for each REM-included sequence.
    - Saves these files in a designated folder (path modifiable in the script).

### SleepInvestigatoR
- **Performed by:** `SleepInvestigatoR_V4.R`
  - **Input:**
    - Location of the preprocessing script's output folder containing the sleep sequences.
      - Update the filepath in the `SleepInvestigatoR` script.
  - **Usage:**
    - Run the `SleepInvestigatoR` function in preferred R environment to initialize the function as a variable.
    - Afterwards, call `SleepInvestigatoR` with the input on the next page of this document.
      - Set the `save_name` input variable as the desired output folder path.
    - `SleepInvestigatoR` processes each file and compiles a table of features.
  - **Output:**
    - `SleepInvestigatoR_output_table.xlsx`


#### Example: Running `SleepInvestigatoR`

Use the code below to analyze your sleep data with `SleepInvestigatoR`. To run it with the parameters, manually select and copy the code below, don’t use the GitHub copy button (it inserts unwanted characters):

```r
SleepInvestigatoR(
  FileNames,                          # List of input files
  file.type            = "csv",       # File format input
  epoch.size           = 1,           # the number of seconds you scored in
  max.hours            = 0.75,        # will truncate all file lengths so they equal this number of hours (also adds values of lower than this number)
  byBlocks             = NULL,        # Optional block-based analysis
  byTotal              = TRUE,        # Analyze total sleep sequence
  score.checker        = FALSE,       # Flags W->R transitions
  id.factor            = TRUE,        # First word in the file name of each file name is the animal id
  group.factor         = TRUE,        # The second word in the filename indicates the grouping factor, such as treatment or condition
  normalization        = NULL,        # Normalization based on baseline values
  time.stamp           = FALSE,       # Timestamp column present
  scores.column        = 1,           # Column number with sleep scores (can only be 1st or 2nd column)
  lights.on            = NULL,        # Optional lights-on time
  lights.off           = NULL,        # Optional lights-off time
  time.format          = "hh:mm:ss",  # Time format
  date.format          = "m/d/y",     # Date format
  date.time.separator  = "",          # No separator for the dates
  score.value.changer  = FALSE,       # Uses 1 = Wake, 2 = NREMS, 3 = REMS. Set to FALSE if already the case; otherwise, specify as c(Wake, NREM, REM)
  sleep.adjust         = "NREM.Onset",# Crop off the beginning of a scored sleep file to the first minimum length NREM bout
  NREM.cutoff          = 60,          # Mininum number of uninterrupted epochs to be consider a formal bout of NREM
  REM.cutoff           = 60,          # Mininum number of uninterrupted epochs to be consider a formal bout of REM
  Wake.cutoff          = 60,          # Mininum number of uninterrupted epochs to be consider a formal bout of Wake
  Sleep.cutoff         = 60,          # Mininum number of uninterrupted epochs to be consider a formal bout of sleep (NREM+REM)
  Cycle.cutoff         = 120,         # Mininum number of uninterrupted epochs to be consider a formal sleep cycle
  Propensity.cutoff    = 60,          # Minimum bout length of two states in propensity measurements.
  data.format          = "long",      # Long format data
  save.name            = "C:filepath_to/output_folder" # Output path
)
```

If parameters not clear check [SleepInvestigatoR on GitHub](https://github.com/mgamble1023/SleepInvestigatoR)
>  **Tip:** Make sure that the packages are installed and that your `FileNames` object is a list of valid file paths before calling this function.


### Visualization
- **Performed by:** `SleepInvestigatoR_data_plotting.ipynb`
  - **Input:**
    - `SleepInvestigatoR_output_table.xlsx`
  - **Processing:**
    - Reads the file once and creates a dataframe of it.
    - Extracts necessary features for visualization per subscript.
  - **Output:**
    - Print statements containing the statistical tests
    - Print statements containing the specific values
    - Visualizations of those values

---

## Markov Chain Analysis

- **Input File:**
  - `string_analysis_hypno_latencies_tonic_phasic.xlsx`

- **Performed by:**
  - `First-order_markov_chain.ipynb`
  - `Second-order_markov_chain.ipynb`

- **Processing**
  - Each script reads the input file twice:
    - Once for REM sequences.
    - Once for Phasic, Tonic, and Intermediate sequences.
      - Filepaths in the scripts need to be updated accordingly.
