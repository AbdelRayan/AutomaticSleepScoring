install.packages("xlsx")
library(xlsx)

# Reads the excel sheet
dataset <- read.xlsx("C:/Users/woutp/Downloads/string_analysis_hypno_latencies_tonic_phasic.xlsx", sheetName = "Feuil1")

# Only uses the part of the excel sheet that is necessary
dataset <- dataset[1:388,]

# Removes the column_id column. Comment out of the rat ID is already the first 
# column
dataset <- dataset[, -1]

combined_replacement <- function(treatment_col, sleep_states_col) {
  # Define the treatment replacements
  treatments <- c("0" = "Control", "1" = "CBD", "2" = "Control", "3" = "RGS", 
                  "4" = "Control")
  
  # Define the sleep stage replacements
  sleep_states <- c("W" = "1", "N" = "2", "R" = "3", "I" = "", "_" = "")
  
  # Replace treatments in the treatment column
  for (key in names(treatments)) {
    treatment_col <- gsub(key, treatments[key], treatment_col)
  }
  
  # Replace sleep stages in the sleep stage column
  for (key in names(sleep_states)) {
    sleep_states_col <- gsub(key, sleep_states[key], sleep_states_col)
  }
  
  # Return the modified columns separately
  return(list(treatment_col, sleep_states_col))
}


# Initialize an empty list to store new rows
new_rows <- list()

# Loop to separate trail_number 5
for (i in 1:nrow(dataset)) {  
  # Extract the current row from the dataset
  row <- dataset[i,]

  if (row$trial_num == 5) {
    
    # Obtains the sleep stage sequence and calculate the length of the sequence
    # to determine the size of the groups
    sequence <- dataset$string_rem[i]
    length_of_sequence_rem <- nchar(sequence)
    size_group_rem <- length_of_sequence_rem %/% 4
    
    # Creates four groups that each have 1/4 of the sleep stage sequence
    group1 <- substr(sequence, 1, size_group_rem)
    group2 <- substr(sequence, size_group_rem + 1, 2 * size_group_rem)
    group3 <- substr(sequence, 2 * size_group_rem + 1, 3 * size_group_rem)
    group4 <- substr(sequence, 3 * size_group_rem + 1, length_of_sequence_rem)
    
    # Create new rows for each group
    new_row1 <- row
    new_row1$trial_num <- "5_1"
    new_row1$string_rem <- group1
    
    new_row2 <- row
    new_row2$trial_num <- "5_2"
    new_row2$string_rem <- group2
    
    new_row3 <- row
    new_row3$trial_num <- "5_3"
    new_row3$string_rem <- group3
    
    new_row4 <- row
    new_row4$trial_num <- "5_4"
    new_row4$string_rem <- group4
    
    # Append new rows to the list
    new_rows <- append(new_rows, list(new_row1, new_row2, new_row3, new_row4))
  }
}

# Removes the rows with a trial number of 5, since these are now separated
dataset <- dataset[dataset$trial_num != 5, ]


# Convert the list of new rows to a data frame and add it to the original dataset
if (length(new_rows) > 0) {
  new_rows_df <- do.call(rbind, lapply(new_rows, as.data.frame))
  dataset <- rbind(dataset, new_rows_df)
}

for (i in 1:nrow(dataset)) {  
  # Extract the current row from the data set
  row <- dataset[i,]
  
  result <- combined_replacement(row$treatment, row$string_rem)
  row$treatment <- result[[1]]
  row$string_rem <- result[[2]]
  
  # Split the numerical sleep stages into a vector, with each stage as a separate character
  split_sequence <- unlist(strsplit(row$string_rem, ""))
  
  # Create a dataframe containing only the numerical sleep stages
  temp_data <- data.frame(Sleep_stage = as.numeric(split_sequence))
  
  # Remove column name. Otherwise the name will be counted as a sleep stage
  colnames(temp_data) <- NULL
  
  # Checks empty dataframes
  if (nrow(temp_data) == 0) {
    cat(paste("Skipping row", i, "because temp_data is empty\n"))
    next
  }
  # Write the dataframe to a CSV file, with the filename constructed from various elements of the row
  write.csv(temp_data, paste0("C:/Users/woutp/OneDrive/Donders_instituut/example_dataset_sleepstages/", 
                              row$rat_id, "_", 
                              row$treatment, "_",
                              "NA", "_", 
                              "NA", "_", 
                              "NA","_", 
                             "StudyDay",row$study_day, "_", 
                              "Condition", row$condition, "_",
                              row$trial_num, 
                              ".csv"), 
            row.names = FALSE)
}

rm(group1, group2, group3, group4, sequence, size_group_rem, new_data,original_data, 
   modified_data, combined_data, new_row1, new_row2, new_row3, new_row4, new_rows_df)
