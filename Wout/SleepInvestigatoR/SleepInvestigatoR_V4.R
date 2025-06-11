##Necessary Packages

#First time install only:

#install.packages('data.table') #remove # to run
#install.packages('tidyverse') #remove # to run
#install.packages('chron') #remove # to run
#install.packages("openxlsx")

#Load

library(data.table)
library(tidyverse)
library(chron)
library(openxlsx)

################################################################

##Set working directory as filepath containing sleep scores to be analyzed


FilePath <- "C:/Users/woutp/OneDrive/Donders_instituut/example_dataset_sleepstages"
# FilePath <- "C:/Users/woutp/OneDrive/Donders_instituut/trial_number5"

setwd(FilePath)

FileNames <- list.files(path = FilePath)

#################################################################

SleepInvestigatoR <- function(FileNames, file.type = "csv", epoch.size = 4, max.hours = NULL, byBlocks = NULL, byTotal = TRUE, 
                              score.checker = T, id.factor = TRUE, group.factor = TRUE, normalization = NULL, time.stamp = F, 
                              scores.column = 1, lights.on = NULL, lights.off = NULL, time.format = "hh:mm:ss", date.format = 'm/d/y', 
                              date.time.separator = '', score.value.changer = NULL, sleep.adjust = NULL, NREM.cutoff = 15, 
                              REM.cutoff = 15, Wake.cutoff = 15, Sleep.cutoff = 15, Propensity.cutoff = 15, Cycle.cutoff =0, data.format = "long", 
                              save.name = NULL){
  
  ###Pre-allocate Matrix###
  
  if (byTotal == TRUE) {
    #byTotal Matrix
    mat.byTotal <- matrix(NA_real_, ncol = 98, nrow = length(FileNames))
    colnames(mat.byTotal) <- c("Animal.ID", "Grouping.Factor","Trial.Number", 'Sex', 'Cohort','Age',"Total.Hours", "Wake.Epoch.Count", "NREMS.Epoch.Count", "REMS.Epoch.Count", "Wake.Seconds", "NREMS.Seconds", 
                               "REMS.Seconds", "NREMS.Onset", "REMS.Onset", "Sleep.Onset", "NREMS.Offset", "REMS.Offset", "Sleep.Offset",  "Latency.to.NREMS", "Latency.to.REMS", "Latency.to.Sleep", 
                               "Wake.Percent", "NREMS.Percent", "REMS.Percent", "Arousal.Count", "Microarousal.Count", "Wake.Bout.Count", "NREMS.Bout.Count", "REMS.Bout.Count", 'Sleep.Bout.Count', 'Sleep.Cycle.Count',
                               "Avg.Wake.Bout.Duration", "Avg.NREMS.Bout.Duration","Avg.REMS.Bout.Duration", 'Avg.Sleep.Bout.Duration', "Wake.NREMS.Transition.Count", "Wake.REMS.Transition.Count",
                               "NREMS.Wake.Transition.Count", "NREMS.REMS.Transition.Count", "REMS.Wake.Transition.Count", "REMS.NREMS.Transition.Count", 'Propensity.to.enter.NREMS', 'Propensity.to.Wake.from.NREMS', 
                               'Propensity.to.enter.REMS', 'Propensity.to.Wake.from.REMS', 'Propensity.to.enter.Sleep', 'Propensity.to.Wake.from.Sleep', "Sleep.Fragmentation.Index", "Total.Sleep.Time", 
                               "Sleep.Efficiency", "WASO", "WASF", "Latency.to.Wake.from.NREMS","Latency.to.Wake.from.REMS", "Latency.to.Wake.from.Sleep", "NREMS.Onset.Bout.Length", "REMS.Onset.Bout.Length", 
                               "Sleep.Onset.Bout.Length", "NREMS.Offset.Bout.Length","REMS.Offset.Bout.Length", "Sleep.Offset.Bout.Length", "Post-NREMS.Wake.Onset", "Post-NREMS.Wake.Onset.Bout.Length", 
                               "Post-REMS.Wake.Onset", "Post-REMS.Wake.Onset.Bout.Length", "Post-Sleep.Wake.Onset", "Post-Sleep.Wake.Onset.Bout.Length", "Avg.Wake.Interbout.Interval", "Avg.NREMS.Interbout.Interval", 
                               "Avg.REMS.Interbout.Interval", "WBC.group1", "WBC.group2","WBC.group3","WBC.group4","WBC.group5", "WBC.group6","WBC.group7","WBC.group8", "WBC.group9","NBC.group1","NBC.group2","NBC.group3",
                               "NBC.group4","NBC.group5","NBC.group6","NBC.group7","NBC.group8", "NBC.group9","RBC.group1","RBC.group2","RBC.group3","RBC.group4","RBC.group5","RBC.group6","RBC.group7","RBC.group8","RBC.group9")
    
  }
  
  
  
  
  if (!is.null(byBlocks)) {
    # byBlocks Matrix
    mat.byBlocks <- matrix(NA_real_, ncol = 102, nrow = byBlocks*length(FileNames))
    colnames(mat.byBlocks) <- c("Animal.ID","Grouping.Factor", "Trial.Number", 'Sex', 'Cohort','Age','Blocks',"Dates","ZT","Days.Elapsed.from.Recording.Start", "Light.Dark","Total.Hours",  "Wake.Epoch.Count", "NREMS.Epoch.Count", 
                                "REMS.Epoch.Count", "Wake.Seconds", "NREMS.Seconds", "REMS.Seconds", "NREMS.Onset", "REMS.Onset", "Sleep.Onset", "NREMS.Offset", "REMS.Offset", "Sleep.Offset",  "Latency.to.NREMS", 
                                "Latency.to.REMS", "Latency.to.Sleep", "Wake.Percent", "NREMS.Percent", "REMS.Percent", "Arousal.Count", "Microarousal.Count", "Wake.Bout.Count", "NREMS.Bout.Count", "REMS.Bout.Count", 'Sleep.Bout.Count', 'Sleep.Cycle.Count',
                                "Avg.Wake.Bout.Duration", "Avg.NREMS.Bout.Duration", "Avg.REMS.Bout.Duration", 'Avg.Sleep.Bout.Duration', "Wake.NREMS.Transition.Count", "Wake.REMS.Transition.Count","NREMS.Wake.Transition.Count", 
                                "NREMS.REMS.Transition.Count", "REMS.Wake.Transition.Count", "REMS.NREMS.Transition.Count", 'Propensity.to.enter.NREMS', 'Propensity.to.Wake.from.NREMS', 'Propensity.to.enter.REMS', 'Propensity.to.Wake.from.REMS',
                                'Propensity.to.enter.Sleep', 'Propensity.to.Wake.from.Sleep', "Sleep.Fragmentation.Index", "Total.Sleep.Time", "Sleep.Efficiency", "WASO", "WASF", "Latency.to.Wake.from.NREMS", "Latency.to.Wake.from.REMS",
                                "Latency.to.Wake.from.Sleep", "NREMS.Onset.Bout.Length", "REMS.Onset.Bout.Length", "Sleep.Onset.Bout.Length", "NREMS.Offset.Bout.Length", "REMS.Offset.Bout.Length", "Sleep.Offset.Bout.Length", "Post-NREMS.Wake.Onset", 
                                "Post-NREMS.Wake.Onset.Bout.Length", "Post-REMS.Wake.Onset","Post-REMS.Wake.Onset.Bout.Length", "Post-Sleep.Wake.Onset", "Post-Sleep.Wake.Onset.Bout.Length", "Avg.Wake.Interbout.Interval", 
                                "Avg.NREMS.Interbout.Interval","Avg.REMS.Interbout.Interval", "WBC.group1", "WBC.group2","WBC.group3", "WBC.group4","WBC.group5","WBC.group6","WBC.group7","WBC.group8", "WBC.group9",
                                "NBC.group1","NBC.group2","NBC.group3","NBC.group4","NBC.group5","NBC.group6","NBC.group7","NBC.group8","NBC.group9","RBC.group1","RBC.group2","RBC.group3","RBC.group4","RBC.group5","RBC.group6",
                                "RBC.group7","RBC.group8","RBC.group9")
  }
  
  # row range byBlocks
  if (!is.null(byBlocks)) {
    x.range1 <- seq(from = 1, to = byBlocks * length(FileNames), by = byBlocks)
    x.range2 <- x.range1 + byBlocks - 1
  }
  
  
  
  #Multiples (see WBC,NBC, and RBC)
  a = epoch.size
  
  for(i in 1:9){
    a[i+1] = 2*a[i] 
  }
  
  a <- a/epoch.size
  
  
  for (FileName in FileNames) {
    names <-  strsplit(FileName, "[[:punct:]]")
    names <- sapply(names, function(x) return(x))
    
    id <- names[1]
    group <- names[2]
    sex <- names[3]
    cohort <- names[4]
    age <- names[5]
    
    if (length(names) == 10) {
      trial_number <- (paste(names[8], names[9], sep="_"))
    } else{
      trial_number <- (names[8])
    }
    
    #Iteration Count
    counter <- match(FileName,FileNames)
    
    #Load in scores
    if(time.stamp == T & scores.column == 1){
      
      dat <- read.csv(FileName, header = FALSE, sep = ",")
      scores <- dat$V1[1:(max.hours*3600/epoch.size)] #needs to match original code
      dt <- dat$V2 #prevents dropping of zeros
      dt <- dat[1:(max.hours*3600/epoch.size)]
      
      rm(dat)
      
    } else if(time.stamp == T & scores.column == 2){
      
      dat <- read.csv(FileName, header = FALSE, sep = ",")
      scores <- dat$V2[1:(max.hours*3600/epoch.size)]
      dt <- dat$V1 #prevents dropping of zeros
      dt <- dt[1:(max.hours*3600/epoch.size)]
      
      rm(dat)
      
    } else if (time.stamp == F){
      
      scores <- read.csv(FileName, header = FALSE, sep = ",")$V1[1:(max.hours*3600/epoch.size)]
      print(FileName)
      
    }
    
    
    if(length(scores[!is.na(scores)]) < (max.hours*3600/epoch.size)){
      print(paste("Warning:", FileName, "contains less than the max.hours set. Averages will be based on its shorter length."))
    }
    
    if (is.null(score.value.changer) == FALSE) {
      scores[scores == score.value.changer[1]] <- 1
      scores[scores == score.value.changer[2]] <- 2
      scores[scores == score.value.changer[3]] <- 3
      
    }
    
    scores.sleepwake <- scores
    scores.sleepwake[scores.sleepwake == 2] <- 4
    scores.sleepwake[scores.sleepwake == 3] <- 4
    value.check <- any(scores > 3 | scores < 1)
    
    # Check if value.check is not NA and TRUE
    if (!is.na(value.check) && value.check == TRUE) {
      stop(print(paste("Scores in", FileName, "are not in the required 1,2,3 format.")))
    }
    
    rle_scores <- rle(scores.sleepwake)  # Run-length encoding to identify sequences
    short_wake <- rle_scores$values == 1 & rle_scores$lengths < Wake.cutoff
    rle_scores$values[short_wake] <- 0  # Convert short wake sequences to 0
    scores.noshortwake <- inverse.rle(rle_scores)  # Decode back to the original format
    
    # Remove NA values
    scores.noshortwake <- na.omit(scores.noshortwake)
    
    
    # Ensure scores.noshortwake is a numeric vector
    scores.noshortwake <- as.numeric(scores.noshortwake)
    
    modified.scores <- scores.noshortwake
    rle_scores <- rle(scores.noshortwake)
    for (i in seq_along(rle_scores$values)) {
      # Check if 0s are surrounded by 4s and ensure no NA values again
      if (!is.na(rle_scores$values[i]) && rle_scores$values[i] == 0 &&
          i > 1 && i < length(rle_scores$values) &&
          !is.na(rle_scores$values[i - 1]) && rle_scores$values[i - 1] == 4 &&
          !is.na(rle_scores$values[i + 1]) && rle_scores$values[i + 1] == 4) {
        
        # Replace all 0s in this run with 4s
        start_index <- sum(rle_scores$lengths[1:(i - 1)]) + 1
        end_index <- start_index + rle_scores$lengths[i] - 1
        modified.scores[start_index:end_index] <- 4
      }
    }
    
    
    
    #Check for Wake-REM Transitions
    if(score.checker == T){
      
      WR <- with(rle(scores), {
        y <- data.frame(values = values, lead = data.table::shift(values, type ="lead"),starts = cumsum(lengths))
        y[y$values == 1 & y$lead == 3,]
      })
      
      if(!is.na(WR$starts[1])){
        print(paste("Warning:", FileName, "contains a Wake-REMS Transition at epoch", WR$starts[1],"of the file.","If you are studying Wake to REMS transitions please set score.checker to FALSE."))
      }
    }
    
    
    #Adjust score length
    if(!is.null(sleep.adjust)) {
      if (sleep.adjust == "NREMS.Onset") {
        
        NREM.Lat <- with(rle(scores), {
          y <- data.frame(lengths = lengths, values = values, ends = cumsum(lengths), starts = cumsum(lengths) - lengths + 1)
          y[y$lengths >= NREM.cutoff & y$values ==2,]
        })
        
        if (length(NREM.Lat$starts) == 0) {
          print("NREM.Lat$starts is empty. Please check your data and preprocessing steps.")
          next
        }
        scores <- scores[-c(1:(NREM.Lat$starts[1]-1))]
        
      } else if (sleep.adjust == "REMS.Onset"){
        
        REM.Lat <- with(rle(scores), {
          y <- data.frame(lengths = lengths, values = values, ends = cumsum(lengths), starts = cumsum(lengths) - lengths + 1)
          y[y$lengths >= NREM.cutoff & y$values ==3,]
        })
        
        scores <- scores[-c(1:(REM.Lat$starts[1]-1))]
        
      } else if (sleep.adjust == "Sleep.Onset"){
        
        Sleep.Lat <- with(rle(scores.sleepwake), {
          y <- data.frame(lengths = lengths, values = values, ends = cumsum(lengths), starts = cumsum(lengths) - lengths + 1)
          y[y$lengths >= Sleep.cutoff & y$values == 4,]
        })
        
        scores <- scores[-c(1:(Sleep.Lat$starts[1]-1))]
        
      } else if (is.numeric(sleep.adjust))
        
        scores <- scores[-c(1:sleep.adjust)]
    }
    
    #byBlocks 
    if (!is.null(byBlocks)) {
      
      if (id.factor == TRUE & group.factor == TRUE)  {
        mat.byBlocks[x.range1[counter]:x.range2[counter],'Animal.ID'] <- id
        mat.byBlocks[x.range1[counter]:x.range2[counter],'Grouping.Factor'] <- group
        mat.byBlocks[x.range1[counter]:x.range2[counter],'Sex'] <- sex
        mat.byBlocks[x.range1[counter]:x.range2[counter],'Cohort'] <- cohort
        mat.byBlocks[x.range1[counter]:x.range2[counter],'Age'] <- age
        mat.byBlocks[x.range1[counter]:x.range2[counter],'Blocks'] <- seq(from = round((length(scores)*epoch.size/3600))/byBlocks, to = round(length(scores)*epoch.size/3600),
                                                                          by = round((length(scores)*epoch.size/3600))/byBlocks)
        mat.byBlocks[,'Total.Hours'] <- length(scores)*epoch.size/3600
      } else if (id.factor == TRUE & group.factor == FALSE) {
        mat.byBlocks[x.range1[counter]:x.range2[counter],'Animal.ID'] <- id
        mat.byBlocks[x.range1[counter]:x.range2[counter],'Grouping.Factor'] <- NA
        mat.byBlocks[x.range1[counter]:x.range2[counter],'Sex'] <- sex
        mat.byBlocks[x.range1[counter]:x.range2[counter],'Cohort'] <- cohort
        mat.byBlocks[x.range1[counter]:x.range2[counter],'Age'] <- age
        mat.byBlocks[x.range1[counter]:x.range2[counter],'Blocks'] <- seq(from = round((length(scores)*epoch.size/3600))/byBlocks, to = round(length(scores)*epoch.size/3600),
                                                                          by = round((length(scores)*epoch.size/3600))/byBlocks)
        mat.byBlocks[,'Total.Hours'] <- length(scores)*epoch.size/3600
      } else if (id.factor == FALSE & group.factor == TRUE) {
        mat.byBlocks[x.range1[counter]:x.range2[counter],'Animal.ID'] <- rep(1:length(scores), each=byBlocks)
        mat.byBlocks[x.range1[counter]:x.range2[counter],'Grouping.Factor'] <- group
        mat.byBlocks[x.range1[counter]:x.range2[counter],'Sex'] <- sex
        mat.byBlocks[x.range1[counter]:x.range2[counter],'Cohort'] <- cohort
        mat.byBlocks[x.range1[counter]:x.range2[counter],'Age'] <- age
        mat.byBlocks[x.range1[counter]:x.range2[counter],'Blocks'] <- seq(from = round((length(scores)*epoch.size/3600))/byBlocks, to = round(length(scores)*epoch.size/3600),
                                                                          by = round((length(scores)*epoch.size/3600))/byBlocks)
        mat.byBlocks[,'Total.Hours'] <- length(scores)*epoch.size/3600
      }
    } 
    
    #byTotal
    if (byTotal == TRUE){
      if (id.factor == TRUE & group.factor == TRUE)  {
        mat.byTotal[counter:counter,"Animal.ID"] <- id
        mat.byTotal[counter:counter,"Grouping.Factor"] <- group
        mat.byTotal[counter:counter,"Sex"] <- sex
        mat.byTotal[counter:counter,'Cohort'] <- cohort
        mat.byTotal[counter:counter,'Age'] <- age
        mat.byTotal[counter:counter, 'Trial.Number'] <- trial_number
        mat.byTotal[,'Total.Hours'] <- length(scores)*epoch.size/3600
      } else if (id.factor == TRUE & group.factor == FALSE) {
        mat.byTotal[counter:counter,'Animal.ID'] <- id
        mat.byTotal[counter:counter,'Grouping.Factor'] <- NA
        mat.byTotal[counter:counter,"Sex"] <- sex
        mat.byTotal[counter:counter,'Cohort'] <- cohort
        mat.byTotal[counter:counter,'Age'] <- age
        mat.byTotal[counter:counter,'Trial.Number'] <- trial_number
        mat.byTotal[,'Total.Hours'] <- length(scores)*epoch.size/3600
      } else if (id.factor == FALSE & group.factor == TRUE) {
        mat.byTotal[counter:counter,'Animal.ID'] <- counter
        mat.byTotal[counter:counter,'Grouping.Factor'] <- group
        mat.byTotal[counter:counter,"Sex"] <- sex
        mat.byTotal[counter:counter,'Cohort'] <- cohort
        mat.byTotal[counter:counter,'Age'] <- age
        mat.byTotal[counter:counter, 'Trial.Number'] <- trial_number
        mat.byTotal[,'Total.Hours'] <- length(scores)*epoch.size/3600
      }
    }
    
    
    if(!is.null(byBlocks)){
      scores.list <- split(scores, ceiling(seq_along(scores)/floor(length(scores)/byBlocks))) 
      scores.rle.list <- lapply(scores.list, function(x) rle(x))
      
      scores.sleepwake.list <- split(scores.sleepwake, ceiling(seq_along(scores)/floor(length(scores)/byBlocks))) 
      scores.sleepwake.rle.list <- lapply(scores.sleepwake.list, function(x) rle(x))
    }
    
    
    if(byTotal == TRUE){
      
      
      ###byTotal Latencies###
      
      
      #Latency to NREM, NREM Onset, & NREM Onset Bout length byTotal
      NREM.Lat <- with(rle(scores), {
        y <- data.frame(lengths = lengths, values = values, ends = cumsum(lengths), starts = cumsum(lengths) - lengths + 1)
        y[y$lengths >= NREM.cutoff & y$values ==2,]
      })
      
      mat.byTotal[counter:counter,'NREMS.Onset'] <- NREM.Lat$starts[1] #NREM Onset
      mat.byTotal[counter:counter,'Latency.to.NREMS'] <- ((NREM.Lat$starts[1]-1)*epoch.size)/60 #Latency to NREM
      mat.byTotal[counter:counter,'NREMS.Onset.Bout.Length'] <- (NREM.Lat$lengths[1]*epoch.size)/60 #NREM Onset Bout Length
      
      #Latency to REM, REM Onset, & REM Onset Bout length byTotal
      REM.Lat <- with(rle(scores), {
        y <- data.frame(lengths = lengths, values = values, ends = cumsum(lengths), starts = cumsum(lengths) - lengths + 1)
        y[y$lengths >= REM.cutoff & y$values ==3,]
      })
      
      mat.byTotal[counter:counter,'REMS.Onset'] <- REM.Lat$starts[1] #REM Onset
      mat.byTotal[counter:counter,'Latency.to.REMS'] <- ((REM.Lat$starts[1]-1)*epoch.size)/60 #Latency to REM
      mat.byTotal[counter:counter,'REMS.Onset.Bout.Length'] <- (REM.Lat$lengths[1]*epoch.size)/60 #REM Onset Bout Length
      
      #Latency to Sleep, Sleep Onset, & Sleep Onset Bout length byTotal
      Sleep.Lat <- with(rle(scores.sleepwake), {
        y <- data.frame(lengths = lengths, values = values, ends = cumsum(lengths), starts = cumsum(lengths) - lengths + 1)
        y[y$lengths >= Sleep.cutoff & y$values == 4,]
      })
      
      mat.byTotal[counter:counter,'Sleep.Onset'] <- Sleep.Lat$starts[1] #Sleep Onset
      mat.byTotal[counter:counter,'Latency.to.Sleep'] <- ((Sleep.Lat$starts[1]-1)*epoch.size)/60 #Latency to Sleep
      mat.byTotal[counter:counter,'Sleep.Onset.Bout.Length'] <- (Sleep.Lat$lengths[1]*epoch.size)/60 #Sleep Onset Bout Length
      
      #Latency to Wake from NREM, Post-NREM Wake Onset Bout Length byTotal
      Wake.from.NREM.Lat <- with(rle(scores), {
        y <- data.frame(lengths = lengths, values = values, value.lag = data.table::shift(values), lengths.lag = data.table::shift(lengths),
                        ends = cumsum(lengths), starts = cumsum(lengths) - lengths + 1)
        y[y$lengths >= Wake.cutoff & y$lengths.lag >= NREM.cutoff & y$values ==1 & y$value.lag == 2, ]
      })
      
      Wake.from.NREM.Lat <- Wake.from.NREM.Lat[complete.cases(Wake.from.NREM.Lat), ]
      
      mat.byTotal[counter:counter,'Post-NREMS.Wake.Onset'] <- Wake.from.NREM.Lat$starts[1] #Post-NREMS Wake Onset
      mat.byTotal[counter:counter,'Latency.to.Wake.from.NREMS'] <- ((Wake.from.NREM.Lat$starts[1]-1)*epoch.size)/60 #Latency to Wake from NREM
      mat.byTotal[counter:counter,'Post-NREMS.Wake.Onset.Bout.Length'] <- (Wake.from.NREM.Lat$lengths[1]*epoch.size)/60 #Post-NREMS Wake Onset Bout Length
      
      #Latency to Wake from REM, Post-REM Wake Onset Bout Length byTotal
      Wake.from.REM.Lat <- with(rle(scores), {
        y <- data.frame(lengths = lengths, values = values, value.lag = data.table::shift(values), lengths.lag = data.table::shift(lengths),
                        ends = cumsum(lengths), starts = cumsum(lengths) - lengths + 1)
        y[y$lengths >= Wake.cutoff & y$lengths.lag >= REM.cutoff & y$values ==1 & y$value.lag == 3,]
      })
      
      Wake.from.REM.Lat <- Wake.from.REM.Lat[complete.cases(Wake.from.REM.Lat), ]
      
      mat.byTotal[counter:counter,'Post-REMS.Wake.Onset'] <- Wake.from.REM.Lat$starts[1] #Post-REMS Wake Onset
      mat.byTotal[counter:counter,'Latency.to.Wake.from.REMS'] <- ((Wake.from.REM.Lat$starts[1]-1)*epoch.size)/60 #Latency to Wake from REM
      mat.byTotal[counter:counter,'Post-REMS.Wake.Onset.Bout.Length'] <- (Wake.from.REM.Lat$lengths[1]*epoch.size)/60 #Post-REM Wake Onset Bout Length
      
      #Latency to Wake from Sleep, Post-Sleep Wake Onset Bout Length byTotal
      Wake.from.Sleep.Lat <- with(rle(scores.sleepwake), {
        y <- data.frame(lengths = lengths, values = values, value.lag = data.table::shift(values), lengths.lag = data.table::shift(lengths),
                        ends = cumsum(lengths), starts = cumsum(lengths) - lengths + 1)
        y[y$lengths >= Wake.cutoff & y$lengths.lag >= Sleep.cutoff & y$values ==1 & y$value.lag == 4,]
      })
      
      Wake.from.Sleep.Lat <- Wake.from.Sleep.Lat[complete.cases(Wake.from.Sleep.Lat), ]
      
      mat.byTotal[counter:counter,'Post-Sleep.Wake.Onset'] <- Wake.from.Sleep.Lat$starts[1] #Post-Sleep Wake Onset
      mat.byTotal[counter:counter,'Latency.to.Wake.from.Sleep'] <- ((Wake.from.Sleep.Lat$starts[1]-1)*epoch.size)/60 #Latency to Wake from Sleep
      mat.byTotal[counter:counter,'Post-Sleep.Wake.Onset.Bout.Length'] <- (Wake.from.Sleep.Lat$lengths[1]*epoch.size)/60 #Post-Sleep Wake Onset Bout Length
      
      
      ###byTotal Sleep Offset###
      
      
      #NREM Offset, NREM Offset Bout Length byTotal
      NREM.Offset <- with(rle(scores), {
        y <- data.frame(lengths = lengths, values = values, ends = cumsum(lengths))    
        y[y$lengths >= NREM.cutoff & y$values ==2,]
      })
      
      if(is.data.frame(NREM.Offset) && nrow(NREM.Offset)==0){ #Confirms if REMS is present (checks for empty df), if TRUE it fills it with NA values
        NREM.Offset <- data.frame(matrix(ncol = 2, nrow = 1))
        colnames(NREM.Offset) <- c('lengths', 'ends')
        NREM.Offset$lengths <- NA
        NREM.Offset$ends <- NA
      }
      
      NREM.Offset.Lengths <- tail(NREM.Offset$lengths, n = 1L)
      NREM.Offset <- tail(NREM.Offset$ends, n = 1L)
      
      mat.byTotal[counter:counter,'NREMS.Offset'] <- NREM.Offset #NREM Offset
      mat.byTotal[counter:counter,'NREMS.Offset.Bout.Length'] <- (NREM.Offset.Lengths*epoch.size)/60 #NREM Offset Bout Length
      
      #REM Offset, REM Offset Bout Length byTotal
      REM.Offset <- with(rle(scores), {
        y <- data.frame(lengths = lengths, values = values, ends = cumsum(lengths))    
        y[y$lengths >= REM.cutoff & y$values ==3,]
      })
      
      if(is.data.frame(REM.Offset) && nrow(REM.Offset)==0){ #Confirms if REMS is present (checks for empty df), if TRUE it fills it with NA values
        REM.Offset <- data.frame(matrix(ncol = 2, nrow = 1))
        colnames(REM.Offset) <- c('lengths', 'ends')
        REM.Offset$lengths <- NA
        REM.Offset$ends <- NA
      }
      
      REM.Offset.Lengths <- tail(REM.Offset$lengths, n = 1L)
      REM.Offset <- tail(REM.Offset$ends, n = 1L) 
      
      
      mat.byTotal[counter:counter,'REMS.Offset'] <- REM.Offset #REM Offset
      mat.byTotal[counter:counter,'REMS.Offset.Bout.Length'] <- (REM.Offset.Lengths*epoch.size)/60 #REM Offset Bout Length
      
      #Sleep Offset, Sleep Offset Bout Length byTotal
      Sleep.Offset <- with(rle(scores.sleepwake), {
        y <- data.frame(lengths = lengths, values = values, ends = cumsum(lengths))    
        y[y$lengths >= Sleep.cutoff & y$values == 4,]
      })
      
      if(is.data.frame(Sleep.Offset) && nrow(Sleep.Offset)==0){ #Confirms if REMS is present (checks for empty df), if TRUE it fills it with NA values
        Sleep.Offset <- data.frame(matrix(ncol = 2, nrow = 1))
        colnames(Sleep.Offset) <- c('lengths', 'ends')
        Sleep.Offset$lengths <- NA
        Sleep.Offset$ends <- NA
      }
      
      Sleep.Offset.Lengths <- tail(Sleep.Offset$lengths, n = 1L)
      Sleep.Offset <- tail(Sleep.Offset$ends, n = 1L)
      
      mat.byTotal[counter:counter,'Sleep.Offset'] <- Sleep.Offset #Sleep Offset
      mat.byTotal[counter:counter,'Sleep.Offset.Bout.Length'] <- (Sleep.Offset.Lengths*epoch.size)/60 #Sleep Offset Bout Length
      
      
      ###byTotal Transitions###
      
      
      #Wake-NREM Transition byTotal
      WN <- with(rle(scores), {
        y <- data.frame(values = values, lead = data.table::shift(values, type ="lead"))
        y[y$values == 1 & y$lead == 2,]
      })
      
      mat.byTotal[counter:counter,'Wake.NREMS.Transition.Count'] <- as.integer(length(WN$values[!is.na(WN$values)]))
      
      WR <- with(rle(scores), {
        y <- data.frame(values = values, lead = data.table::shift(values, type ="lead"),starts = cumsum(lengths))
        y[y$values == 1 & y$lead == 3,]
      })
      
      mat.byTotal[counter:counter,'Wake.REMS.Transition.Count'] <- as.integer(length(WR$values[!is.na(WR$values)]))
      
      NW <- with(rle(scores), {
        y <- data.frame(values = values, lead = data.table::shift(values, type ="lead"))
        y[y$values == 2 & y$lead == 1,]
      })
      
      mat.byTotal[counter:counter,'NREMS.Wake.Transition.Count'] <- as.integer(length(NW$values[!is.na(NW$values)]))
      
      NR <- with(rle(scores), {
        y <- data.frame(values = values, lead = data.table::shift(values, type ="lead"))
        y[y$values == 2 & y$lead == 3,]
      })
      
      mat.byTotal[counter:counter,'NREMS.REMS.Transition.Count'] <- as.integer(length(NR$values[!is.na(NR$values)]))
      
      RW <- with(rle(scores), {
        y <- data.frame(values = values, lead = data.table::shift(values, type ="lead"))
        y[y$values == 3 & y$lead == 1,]
      })
      
      mat.byTotal[counter:counter,'REMS.Wake.Transition.Count'] <- as.integer(length(RW$values[!is.na(RW$values)]))
      
      RN <- with(rle(scores), {
        y <- data.frame(values = values, lead = data.table::shift(values, type ="lead"))
        y[y$values == 3 & y$lead == 2,]
      })
      
      mat.byTotal[counter:counter,'REMS.NREMS.Transition.Count'] <- as.integer(length(RN$values[!is.na(RN$values)]))
      
      
      ###byTotal Epoch Counts###
      
      
      #Wake Epoch Count byTotal
      Wake.Epoch.Count <- sum(scores == 1, na.rm = TRUE)
      
      mat.byTotal[counter:counter,'Wake.Epoch.Count'] <- as.integer(Wake.Epoch.Count)
      
      #NREM Epoch Count byTotal
      NREM.Epoch.Count <- sum(scores == 2, na.rm = TRUE)
      
      mat.byTotal[counter:counter,'NREMS.Epoch.Count'] <- as.integer(NREM.Epoch.Count)
      
      #REM Epoch Count byTotal
      REM.Epoch.Count <- sum(scores == 3, na.rm=TRUE)
      
      mat.byTotal[counter:counter,'REMS.Epoch.Count'] <- as.integer(REM.Epoch.Count)
      
      
      ###byTotal Vigilance State Seconds###
      
      
      #Wake Seconds byTotal
      Wake.Seconds <- Wake.Epoch.Count*epoch.size
      
      mat.byTotal[counter:counter,'Wake.Seconds'] <- as.integer(Wake.Seconds)
      
      #NREM Seconds byTotal
      NREM.Seconds <- NREM.Epoch.Count*epoch.size
      
      mat.byTotal[counter:counter,'NREMS.Seconds'] <- as.integer(NREM.Seconds)
      
      #REM Seconds byTotal
      REM.Seconds <- REM.Epoch.Count*epoch.size
      
      mat.byTotal[counter:counter,'REMS.Seconds'] <- as.integer(REM.Seconds)
      
      
      ###byTotal Vigilance State Percentages###
      
      #Wake Percentage byTotal
      Wake.Percent <- mean(scores == 1, na.rm=TRUE)
      
      mat.byTotal[counter:counter,'Wake.Percent'] <- as.numeric(Wake.Percent)
      
      #NREM Percentage byTotal
      NREM.Percent <- mean(scores == 2, na.rm=TRUE)
      
      mat.byTotal[counter:counter,'NREMS.Percent'] <- as.numeric(NREM.Percent)
      
      #REM Percentage byTotal
      REM.Percent <- mean(scores == 3, na.rm=TRUE)
      
      mat.byTotal[counter:counter,'REMS.Percent'] <- as.numeric(REM.Percent)
      
      
      ###byTotal Episode/Bout Counts###
      
      
      #Arousal Count byTotal
      Arousal.Count <- with(rle(scores), {
        sum(values == 1, na.rm = TRUE)
      })
      
      mat.byTotal[counter:counter,'Arousal.Count'] <- as.integer(Arousal.Count)
      
      #Microarousal Count byTotal
      if (Wake.cutoff <= 1) {
        
        Microarousal.Count <- with(rle(scores), {
          sum(lengths == 1 & values == 1, na.rm = TRUE)
        })
        
      } else {
        
        Microarousal.Count <- with(rle(scores), {
          sum(lengths < Wake.cutoff & values == 1, na.rm = TRUE)
        })
      }
      
      mat.byTotal[counter:counter,'Microarousal.Count'] <- as.integer(Microarousal.Count)
      
      #Wake Bout Count byTotal
      Wake.Bout.Count <- with(rle(scores), {
        
        sum(lengths >= Wake.cutoff & values == 1, na.rm = TRUE)
      })
      mat.byTotal[counter:counter, 'Wake.Bout.Count'] <- as.integer(Wake.Bout.Count)
      
      #Wake Bout Counts by group byTotal
      WBC.group1 <- with(rle(scores), {
        sum(values == 1 & lengths <= a[1])
      })
      
      mat.byTotal[counter:counter, 'WBC.group1'] <- as.integer(WBC.group1)
      
      WBC.group2 <- with(rle(scores), {
        sum(values == 1 & lengths >= a[2] & lengths < a[3])
      })
      
      mat.byTotal[counter:counter,'WBC.group2'] <- as.integer(WBC.group2)
      
      WBC.group3 <- with(rle(scores), {
        sum(values == 1 & lengths >= a[3] & lengths < a[4])
      })
      
      mat.byTotal[counter:counter,'WBC.group3'] <- as.integer(WBC.group3)
      
      WBC.group4 <- with(rle(scores), {
        sum(values == 1 & lengths >= a[4] & lengths < a[5])
      })
      
      mat.byTotal[counter:counter,'WBC.group4'] <- as.integer(WBC.group4)
      
      WBC.group5 <- with(rle(scores), {
        sum(values == 1 & lengths >= a[5] & lengths < a[6])
      })
      
      mat.byTotal[counter:counter,'WBC.group5'] <- as.integer(WBC.group5)
      
      WBC.group6 <- with(rle(scores), {
        sum(values == 1 & lengths >= a[6] & lengths < a[7])
      })
      
      mat.byTotal[counter:counter,'WBC.group6'] <- as.integer(WBC.group6)
      
      WBC.group7 <- with(rle(scores), {
        sum(values == 1 & lengths >= a[7] & lengths < a[8])
      })
      
      mat.byTotal[counter:counter,'WBC.group7'] <- as.integer(WBC.group7)
      
      WBC.group8 <- with(rle(scores), {
        sum(values == 1 & lengths >= a[8] & lengths < a[9])
      })
      
      mat.byTotal[counter:counter,'WBC.group8'] <- as.integer(WBC.group8)
      
      WBC.group9 <- with(rle(scores), {
        sum(values == 1 & lengths >= a[9])
      })
      
      mat.byTotal[counter:counter,'WBC.group9'] <- as.integer(WBC.group9)
      
      
      #NREMS Bout Count byTotal
      NREMS.Bout.Count <- with(rle(scores), {
        sum(lengths >= NREM.cutoff & values == 2, na.rm = TRUE)
      })
      
      mat.byTotal[counter:counter,'NREMS.Bout.Count'] <- as.integer(NREMS.Bout.Count)
      
      #NREMS Bout Counts by group byTotal
      NBC.group1 <- with(rle(scores), {
        sum(values == 2 & lengths <= a[1])
      })
      
      mat.byTotal[counter:counter,'NBC.group1'] <- as.integer(NBC.group1)
      
      NBC.group2 <- with(rle(scores), {
        sum(values == 2 & lengths >= a[2] & lengths < a[3])
      })
      
      mat.byTotal[counter:counter,'NBC.group2'] <- as.integer(NBC.group2)
      
      NBC.group3 <- with(rle(scores), {
        sum(values == 2 & lengths >= a[3] & lengths < a[4])
      })
      
      mat.byTotal[counter:counter,'NBC.group3'] <- as.integer(NBC.group3)
      
      NBC.group4 <- with(rle(scores), {
        sum(values == 2 & lengths >= a[4] & lengths < a[5])
      })
      
      mat.byTotal[counter:counter,'NBC.group4'] <- as.integer(NBC.group4)
      
      NBC.group5 <- with(rle(scores), {
        sum(values == 2 & lengths >= a[5] & lengths < a[6])
      })
      
      mat.byTotal[counter:counter,'NBC.group5'] <- as.integer(NBC.group5)
      
      NBC.group6 <- with(rle(scores), {
        sum(values == 2 & lengths >= a[6] & lengths < a[7])
      })
      
      mat.byTotal[counter:counter,'NBC.group6'] <- as.integer(NBC.group6)
      
      NBC.group7 <- with(rle(scores), {
        sum(values == 2 & lengths >= a[7] & lengths < a[8])
      })
      
      mat.byTotal[counter:counter,'NBC.group7'] <- as.integer(NBC.group7)
      
      NBC.group8 <- with(rle(scores), {
        sum(values == 2 & lengths >= a[8] & lengths < a[9])
      })
      
      mat.byTotal[counter:counter,'NBC.group8'] <- as.integer(NBC.group8)
      
      NBC.group9 <- with(rle(scores), {
        sum(values == 2 & lengths >= a[9])
      })
      
      mat.byTotal[counter:counter,'NBC.group9'] <- as.integer(NBC.group9)
      
      
      #REMS Bout Count byTotal
      REM.Bout.Count <- with(rle(scores), {
        sum(lengths >= REM.cutoff & values == 3, na.rm = TRUE)
      })
      
      mat.byTotal[counter:counter,'REMS.Bout.Count'] <- as.integer(REM.Bout.Count)
      
      #REM Bout Counts by group byTotal
      RBC.group1 <- with(rle(scores), {
        sum(values == 3 & lengths <= a[1])
      })
      
      mat.byTotal[counter:counter,'RBC.group1'] <- as.integer(RBC.group1)
      
      RBC.group2 <- with(rle(scores), {
        sum(values == 3 & lengths >= a[2] & lengths < a[3])
      })
      
      mat.byTotal[counter:counter,'RBC.group2'] <- as.integer(RBC.group2)
      
      RBC.group3 <- with(rle(scores), {
        sum(values == 3 & lengths >= a[3] & lengths < a[4])
      })
      
      mat.byTotal[counter:counter,'RBC.group3'] <- as.integer(RBC.group3)
      
      RBC.group4 <- with(rle(scores), {
        sum(values == 3 & lengths >= a[4] & lengths < a[5])
      })
      
      mat.byTotal[counter:counter,'RBC.group4'] <- as.integer(RBC.group4)
      
      RBC.group5 <- with(rle(scores), {
        sum(values == 3 & lengths >= a[5] & lengths < a[6])
      })
      
      mat.byTotal[counter:counter,'RBC.group5'] <- as.integer(RBC.group5)
      
      RBC.group6 <- with(rle(scores), {
        sum(values == 3 & lengths >= a[6] & lengths < a[7])
      })
      
      mat.byTotal[counter:counter,'RBC.group6'] <- as.integer(RBC.group6)
      
      RBC.group7 <- with(rle(scores), {
        sum(values == 3 & lengths >= a[7] & lengths < a[8])
      })
      
      mat.byTotal[counter:counter,'RBC.group7'] <- as.integer(RBC.group7)
      
      RBC.group8 <- with(rle(scores), {
        sum(values == 3 & lengths >= a[8] & lengths < a[9])
      })
      
      mat.byTotal[counter:counter,'RBC.group8'] <- as.integer(RBC.group8)
      
      RBC.group9 <- with(rle(scores), {
        sum(values == 3 & lengths >= a[9])
      })
      
      mat.byTotal[counter:counter,'RBC.group9'] <- as.integer(RBC.group9)
      
      #Sleep Bout Count byTotal
      Sleep.Bout.Count <- with(rle(scores.sleepwake), {
        sum(lengths >= Sleep.cutoff & values == 4, na.rm = TRUE)
      })
      
      mat.byTotal[counter:counter,'Sleep.Bout.Count'] <- as.integer(Sleep.Bout.Count)
      
      
      #Sleep Cycle Count byTotal
      Sleep.Cycle.Count <- with(rle(modified.scores), {  
        sum(values == 4 & lengths >= Cycle.cutoff)
      })
      
      mat.byTotal[counter:counter,'Sleep.Cycle.Count'] <- as.integer(Sleep.Cycle.Count)
      
      
      ###byTotal Bout Duration Average###
      
      #Wake Bout Duration Average byTotal
      Wake.Bout.Duration <- with(rle(scores), {
        y <- data.frame(lengths = lengths, values = values)
        y <- y[y$lengths >= Wake.cutoff & y$values ==1,]
        mean(y$lengths, na.rm = TRUE)
      })
      
      mat.byTotal[counter:counter,'Avg.Wake.Bout.Duration'] <- (as.numeric((Wake.Bout.Duration*epoch.size)))/60
      
      #NREM Bout Duration Average byTotal
      NREM.Bout.Duration <- with(rle(scores), {
        y <- data.frame(lengths = lengths, values = values)
        y <- y[y$lengths >= NREM.cutoff & y$values ==2,]
        mean(y$lengths, na.rm = TRUE)
      })
      
      mat.byTotal[counter:counter,'Avg.NREMS.Bout.Duration'] <- (as.numeric((NREM.Bout.Duration*epoch.size)))/60
      
      #REM Bout Duration Average byTotal
      REM.Bout.Duration <- with(rle(scores), {
        y <- data.frame(lengths = lengths, values = values)
        y <- y[y$lengths >= REM.cutoff & y$values ==3,]
        mean(y$lengths, na.rm=TRUE)
      })
      
      mat.byTotal[counter:counter,'Avg.REMS.Bout.Duration'] <- (as.numeric((REM.Bout.Duration*epoch.size)))/60
      
      #Sleep Bout Duration Average byTotal
      Sleep.Bout.Duration <- with(rle(scores.sleepwake), {
        y <- data.frame(lengths = lengths, values = values)
        y <- y[y$lengths >= Sleep.cutoff & y$values ==4,]
        mean(y$lengths, na.rm = TRUE)
      })
      
      mat.byTotal[counter:counter,'Avg.Sleep.Bout.Duration'] <- (as.numeric((Sleep.Bout.Duration*epoch.size)))/60
      
      
      ###byTotal Interbout Interval Average###
      
      
      #Wake Interbout Interval byTotal
      Wake.dat <- with(rle(scores),{
        y <- data.frame(lengths = lengths, values = values, ends = cumsum(lengths), starts = cumsum(lengths) - lengths + 1)
        y[y$lengths >= Wake.cutoff & y$values ==1,]
      })
      
      Wake.Interbout.Intervals <- data.frame(IBI = data.table::shift(Wake.dat$starts, type = "lead") - Wake.dat$ends - 1)
      Wake.Interbout.Intervals <- Wake.Interbout.Intervals[!is.na(Wake.Interbout.Intervals)]
      
      Wake.Interbout.Interval.Average <- data.frame(IBIA = mean(Wake.Interbout.Intervals))
      
      
      mat.byTotal[counter:counter,'Avg.Wake.Interbout.Interval'] <- as.numeric((Wake.Interbout.Interval.Average$IBIA[1]*epoch.size)/60)
      
      #NREM Interbout Interval byTotal
      NREM.dat <- with(rle(scores),{
        y <- data.frame(lengths = lengths, values = values, ends = cumsum(lengths), starts = cumsum(lengths) - lengths + 1)
        y[y$lengths >= NREM.cutoff & y$values ==2,]
      })
      
      NREM.Interbout.Intervals <- data.frame(IBI = data.table::shift(NREM.dat$starts, type = "lead") - NREM.dat$ends - 1)
      NREM.Interbout.Intervals <- NREM.Interbout.Intervals[!is.na(NREM.Interbout.Intervals)]
      
      NREM.Interbout.Interval.Average <- data.frame(IBIA = mean(NREM.Interbout.Intervals))
      
      
      mat.byTotal[counter:counter,'Avg.NREMS.Interbout.Interval'] <- as.numeric((NREM.Interbout.Interval.Average$IBIA[1]*epoch.size)/60)
      
      #REM Interbout Interval byTotal
      REM.dat <- with(rle(scores),{
        y <- data.frame(lengths = lengths, values = values, ends = cumsum(lengths), starts = cumsum(lengths) - lengths + 1)
        y[y$lengths >= REM.cutoff & y$values ==3,]
      })
      
      REM.Interbout.Intervals <- data.frame(IBI = data.table::shift(REM.dat$starts, type = "lead") - REM.dat$ends - 1)
      REM.Interbout.Intervals <- REM.Interbout.Intervals[!is.na(REM.Interbout.Intervals)]
      
      REM.Interbout.Interval.Average <- data.frame(IBIA = mean(REM.Interbout.Intervals))
      
      
      mat.byTotal[counter:counter,'Avg.REMS.Interbout.Interval'] <- as.numeric((REM.Interbout.Interval.Average$IBIA[1]*epoch.size)/60)
      
      
      ###byTotal Sleep-Wake Propensity Measures###
      
      #Propensity to enter NREMS
      Wake.NREMS.dat <- with(rle(scores),{
        y <- data.frame(lengths = lengths, values = values, ends = cumsum(lengths), starts = cumsum(lengths) - lengths + 1)
        y[y$lengths >= Propensity.cutoff & (y$values == 1 | y$values == 2),]
      })
      
      Wake.NREMS.dat['lead'] <- c(data.table::shift(Wake.NREMS.dat$values, type = "lead"))
      Wake.NREMS.dat2 <- Wake.NREMS.dat[Wake.NREMS.dat$values == 1 & Wake.NREMS.dat$lead == 2,]
      Wake.NREMS.dat2  <- Wake.NREMS.dat2 [complete.cases(Wake.NREMS.dat2 ), ]
      Propensity.to.enter.NREMS <- (mean((Wake.NREMS.dat2$ends+1) - Wake.NREMS.dat2$starts))*epoch.size
      
      mat.byTotal[counter:counter,'Propensity.to.enter.NREMS'] <- as.numeric(Propensity.to.enter.NREMS)/60
      
      #Propensity to wake from NREMS
      NREMS.Wake.dat <- Wake.NREMS.dat[Wake.NREMS.dat$values == 2 & Wake.NREMS.dat$lead == 1,]
      NREMS.Wake.dat <- NREMS.Wake.dat[complete.cases(NREMS.Wake.dat), ]
      Propensity.to.Wake.from.NREMS <- (mean((NREMS.Wake.dat$ends+1) - NREMS.Wake.dat$starts))*epoch.size
      
      mat.byTotal[counter:counter,'Propensity.to.Wake.from.NREMS'] <- as.numeric(Propensity.to.Wake.from.NREMS)/60
      
      #Propensity to enter REMS
      Wake.REMS.dat <- with(rle(scores),{
        y <- data.frame(lengths = lengths, values = values, ends = cumsum(lengths), starts = cumsum(lengths) - lengths + 1)
        y[y$lengths >= Propensity.cutoff & (y$values == 1 | y$values == 3),]
      })
      
      Wake.REMS.dat['lead'] <- c(data.table::shift(Wake.REMS.dat$values, type = "lead"))
      Wake.REMS.dat2 <- Wake.REMS.dat[Wake.REMS.dat$values == 1 & Wake.REMS.dat$lead == 3,]
      Wake.REMS.dat2  <- Wake.REMS.dat2 [complete.cases(Wake.REMS.dat2 ), ]
      Propensity.to.enter.REMS <- (mean((Wake.REMS.dat2$ends+1) - Wake.REMS.dat2$starts))*epoch.size
      
      mat.byTotal[counter:counter,'Propensity.to.enter.REMS'] <- as.numeric(Propensity.to.enter.REMS)/60
      
      #Propensity to wake from REMS
      REMS.Wake.dat <- Wake.REMS.dat[Wake.REMS.dat$values == 3 & Wake.REMS.dat$lead == 1,]
      REMS.Wake.dat <- REMS.Wake.dat[complete.cases(REMS.Wake.dat), ]
      Propensity.to.Wake.from.REMS <- (mean((REMS.Wake.dat$ends+1) - REMS.Wake.dat$starts))*epoch.size
      
      mat.byTotal[counter:counter,'Propensity.to.Wake.from.REMS'] <- as.numeric(Propensity.to.Wake.from.REMS)/60
      
      #Propensity to enter sleep
      Wake.Sleep.dat <- with(rle(scores.sleepwake),{
        y <- data.frame(lengths = lengths, values = values, ends = cumsum(lengths), starts = cumsum(lengths) - lengths + 1)
        y[y$lengths >= Propensity.cutoff & (y$values == 1 | y$values == 4),]
      })
      
      Wake.Sleep.dat['lead'] <- c(data.table::shift(Wake.Sleep.dat$values, type = "lead"))
      Wake.Sleep.dat2 <- Wake.Sleep.dat[Wake.Sleep.dat$values == 1 & Wake.Sleep.dat$lead == 4,]
      Wake.Sleep.dat2  <- Wake.Sleep.dat2 [complete.cases(Wake.Sleep.dat2 ), ]
      Propensity.to.enter.Sleep <- (mean((Wake.Sleep.dat2$ends+1) - Wake.Sleep.dat2$starts))*epoch.size
      
      mat.byTotal[counter:counter,'Propensity.to.enter.Sleep'] <- as.numeric(Propensity.to.enter.Sleep)/60
      
      #Propensity to wake from sleep
      Sleep.Wake.dat <- Wake.Sleep.dat[Wake.Sleep.dat$values == 4 & Wake.Sleep.dat$lead == 1,]
      Sleep.Wake.dat <- Sleep.Wake.dat[complete.cases(Sleep.Wake.dat), ]
      Propensity.to.Wake.from.Sleep <- (mean((Sleep.Wake.dat$ends+1) - Sleep.Wake.dat$starts))*epoch.size
      
      mat.byTotal[counter:counter,'Propensity.to.Wake.from.Sleep'] <- as.numeric(Propensity.to.Wake.from.Sleep)/60
      
      
    } #End of byTotal (in loop)
    
    
    if(!is.null(byBlocks)) {
      
      #Time.Stamp.Measures
      if(time.stamp == T){
        dtparts = t(as.data.frame(strsplit(dt, date.time.separator))) # have to tell it what the separator is
        row.names(dtparts) = NULL
        
        #Dates
        dates=dtparts[,1]
        Dates <- chron(dates = dates, format = date.format)
        
        mat.byBlocks[x.range1[counter]:x.range2[counter],'Dates'] <- Dates
        
        #LD
        times <- dtparts[,2]
        times <- chron(times = times, format = c(time.format))
        
        l.on <- chron(times = lights.on, format = time.format)
        l.off <- chron(times = lights.off, format = time.format)
        
        LD.all <- ifelse(times > l.on & times < l.off, 1, 2)
        
        seqs <- seq_along(LD.all)
        LD.collapsed <- tapply(LD.all,rep(seqs,each=((max.hours*3600/epoch.size)/byBlocks))[seqs],FUN=mean) #match main sheet
        
        LD <- dplyr::case_when(LD.collapsed == 1 ~ "Light",
                               LD.collapsed == 2 ~ "Dark",
                               LD.collapsed != round(LD.collapsed) ~ "Mixed")
        
        mat.byBlocks[x.range1[counter]:x.range2[counter],'Light.Dark'] <- LD
        
        #ZT
        Time.elapsed.from.lights.on <- times - l.on
        
        seconds.elapsed <- seconds(Time.elapsed.from.lights.on)
        minutes.elapsed <- minutes(Time.elapsed.from.lights.on)*60 #in seconds
        hours.elapsed <- hours(Time.elapsed.from.lights.on)*3600 #in seconds
        
        hms.elapsed <- hours.elapsed + minutes.elapsed + seconds.elapsed
        
        
        ZT.all <- round(hms.elapsed/3600, digits = 2)
        ZT_seqs <- seq_along(ZT.all)
        ZT.collapsed <- tapply(ZT.all,rep(ZT_seqs,each=((max.hours*3600/epoch.size)/byBlocks))[ZT_seqs],FUN=mean)
        ZT.collapsed <- ZT.all[seq(1, max.hours*3600/epoch.size, ((max.hours*3600/epoch.size)/byBlocks))] #dont want first element
        
        mat.byBlocks[x.range1[counter]:x.range2[counter],'ZT'] <- ZT.collapsed
        
        #Days Elapsed from Recording Start
        Time.elapsed.from.file.start <- times - times[1]
        
        seconds.elapsed <- seconds(Time.elapsed.from.file.start)
        minutes.elapsed <- minutes(Time.elapsed.from.file.start)*60 #in seconds
        hours.elapsed <- hours(Time.elapsed.from.file.start)*3600 #in seconds
        
        hms.elapsed <- hours.elapsed + minutes.elapsed + seconds.elapsed
        
        days.elapsed <- round(hms.elapsed/86400, digits = 2)
        number.of.days.elapsed <- length(which(days.elapsed == 0))
        replace.zeros.with.days <- seq.int(0, number.of.days.elapsed-1, 1)
        
        d <- replace(days.elapsed, which(days.elapsed == 0), replace.zeros.with.days)
        add <- rep(0:(which(d == 1) - which(d == 0)), times = length(days.elapsed)/(which(d == 1) - which(d == 0)), each = (which(d == 1) - which(d == 0)))
        
        dhms.elapsed <- days.elapsed + add
        
        mat.byBlocks[x.range1[counter]:x.range2[counter],'Days.Elapsed.From.Recording.Start'] <- dhms.elapsed
        
      }
      
      
      ###byBlocks Latencies###
      
      
      #NREM Latency byBlocks
      NREM.lats <- lapply(scores.rle.list,function(x){
        y <- data.frame(lengths = x$lengths, values = x$values, ends = cumsum(x$lengths), starts = cumsum(x$lengths) - x$lengths + 1)
        y[y$lengths >= NREM.cutoff & y$values ==2,]})
      
      NREM.lats.lengths <- purrr::map(NREM.lats, 1)
      NREM.lats.lengths <- purrr::map_dfr(NREM.lats.lengths, 1, .default = NA) #length
      NREM.lats <- purrr::map(NREM.lats, 4)
      NREM.lats <- purrr::map(NREM.lats, 1, .default = NA) #lats
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'NREMS.Onset'] <- as.integer(unlist(NREM.lats[1:byBlocks])) #NREM Onset
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Latency.to.NREMS'] <- (as.integer((unlist(NREM.lats[1:byBlocks])-1)*epoch.size))/60 #Latency to NREM
      mat.byBlocks[x.range1[counter]:x.range2[counter],'NREMS.Onset.Bout.Length'] <- (as.integer(unlist(NREM.lats.lengths[1:byBlocks])*epoch.size))/60 #NREM Onset Bout Length
      
      #REM Latency byBlocks
      REM.lats <- lapply(scores.rle.list,function(x){
        y <- data.frame(lengths = x$lengths, values = x$values, ends = cumsum(x$lengths), starts = cumsum(x$lengths) - x$lengths + 1)
        y[y$lengths >= REM.cutoff & y$values ==3,]})
      
      REM.lats.lengths <- purrr::map(REM.lats, 1)
      REM.lats.lengths <- purrr::map_dfr(REM.lats.lengths, 1, .default = NA) #length
      REM.lats <- purrr::map(REM.lats, 4)
      REM.lats <- purrr::map(REM.lats, 1, .default = NA) #lats
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'REMS.Onset'] <- as.integer(unlist(REM.lats[1:byBlocks])) #REM Onset
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Latency.to.REMS'] <- (as.integer((unlist(REM.lats[1:byBlocks])-1)*epoch.size))/60 #Latency to REM
      mat.byBlocks[x.range1[counter]:x.range2[counter],'REMS.Onset.Bout.Length'] <- (as.integer(unlist(REM.lats.lengths[1:byBlocks])*epoch.size))/60 #REM Onset Bout Length
      
      #Sleep Latency byBlocks
      Sleep.lats <- lapply(scores.sleepwake.rle.list,function(x){
        y <- data.frame(lengths = x$lengths, values = x$values, ends = cumsum(x$lengths), starts = cumsum(x$lengths) - x$lengths + 1)
        y[y$lengths >= Sleep.cutoff & y$values == 4,]})
      
      Sleep.lats.lengths <- purrr::map(Sleep.lats, 1)
      Sleep.lats.lengths <- purrr::map_dfr(Sleep.lats.lengths, 1, .default = NA) #length
      Sleep.lats <- purrr::map(Sleep.lats, 4)
      Sleep.lats <- purrr::map(Sleep.lats, 1, .default = NA) #lats
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Sleep.Onset'] <- as.integer(unlist(Sleep.lats[1:byBlocks])) #Sleep Onset
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Latency.to.Sleep'] <- (as.integer((unlist(Sleep.lats[1:byBlocks])-1)*epoch.size))/60 #Latency to Sleep
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Sleep.Onset.Bout.Length'] <- (as.integer(unlist(Sleep.lats.lengths[1:byBlocks])*epoch.size))/60 #Sleep Onset Bout Length
      
      #Wake Latency from NREMS byBlocks
      Wake.NREMS.dat <- lapply(scores.rle.list, function(x){
        y <- data.frame(lengths = x$lengths, values = x$values, ends = cumsum(x$lengths), 
                        starts = cumsum(x$lengths) - x$lengths + 1)
        y[y$lengths >= Wake.cutoff & (y$values == 1 | y$values == 2),]
      })
      
      Wake.NREMS.dat2 <- lapply(Wake.NREMS.dat, function (x){
        y <- data.frame(lead.values = data.table::shift(x$values, type = "lead"), lag.values = data.table::shift(x$values, type = "lag"), 
                        lead.starts = data.table::shift(x$starts, type = "lead" ))
        y <- replace(y, is.na(y), 0)
      })
      
      Wake.NREMS.dat3 <- mapply(c, Wake.NREMS.dat, Wake.NREMS.dat2, SIMPLIFY = F)
      
      Wake.NREMS.dat4 <- lapply(Wake.NREMS.dat3, function(x) {
        y <- data.frame(lengths = x$lengths, values = x$values, ends = x$ends, 
                        starts = x$starts, lead.values = x$lead.values, 
                        lag.values = x$lag.values, lead.starts = x$lead.starts)
        y[!(y$values == 1 & y$lag.values == 1),]
      })
      
      Wake.NREMS.dat5 <- lapply(Wake.NREMS.dat4, function(x) {
        y <- data.frame(lengths = x$lengths, values = x$values, ends = x$ends, lag.values = data.table::shift(x$values, type = "lag"),
                        starts = x$starts, lead.values = data.table::shift(x$lead.values, type = "lead"), #These have to be calculated again bc of rows being removed
                        lead.starts = data.table::shift(x$starts, type = 'lead'))
        y[y$values == 1 & y$lag.values == 2,]
      })
      
      Wake.NREMS.dat6 <- purrr::map(Wake.NREMS.dat5, 5, .default = NA)
      Wake.NREMS.dat7 <- purrr::map(Wake.NREMS.dat5, 1)
      
      Latency.to.Wake.from.NREMS <- purrr::map(Wake.NREMS.dat6, 1, .default = NA) 
      Post.NREMS.Wake.Onset.Bout.Length <- purrr::map_dfr(Wake.NREMS.dat7, 1, .default = NA)
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Post-NREMS.Wake.Onset'] <- as.integer((unlist(Latency.to.Wake.from.NREMS[1:byBlocks]))) 
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Latency.to.Wake.from.NREMS'] <- (as.integer((unlist(Latency.to.Wake.from.NREMS[1:byBlocks])-1)*epoch.size))/60 #Latency to Wake from NREM
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Post-NREMS.Wake.Onset.Bout.Length'] <- (as.integer(unlist(Post.NREMS.Wake.Onset.Bout.Length[1:byBlocks])*epoch.size))/60 #Post-NREM Wake Onset Bout Length
      
      #Wake Latency from REMS byBlocks
      Wake.REMS.dat <- lapply(scores.rle.list, function(x){
        y <- data.frame(lengths = x$lengths, values = x$values, ends = cumsum(x$lengths), 
                        starts = cumsum(x$lengths) - x$lengths + 1)
        y[y$lengths >= Wake.cutoff & (y$values == 1 | y$values == 3),]
      })
      
      Wake.REMS.dat2 <- lapply(Wake.REMS.dat, function (x){
        y <- data.frame(lead.values = data.table::shift(x$values, type = "lead"), lag.values = data.table::shift(x$values, type = "lag"), 
                        lead.starts = data.table::shift(x$starts, type = "lead" ))
        y <- replace(y, is.na(y), 0)
      })
      
      Wake.REMS.dat3 <- mapply(c, Wake.REMS.dat, Wake.REMS.dat2, SIMPLIFY = F)
      
      Wake.REMS.dat4 <- lapply(Wake.REMS.dat3, function(x) {
        y <- data.frame(lengths = x$lengths, values = x$values, ends = x$ends, 
                        starts = x$starts, lead.values = x$lead.values, 
                        lag.values = x$lag.values, lead.starts = x$lead.starts)
        y[!(y$values == 1 & y$lag.values == 1),]
      })
      
      Wake.REMS.dat5 <- lapply(Wake.REMS.dat4, function(x) {
        y <- data.frame(lengths = x$lengths, values = x$values, ends = x$ends, lag.values = data.table::shift(x$values, type = "lag"),
                        starts = x$starts, lead.values = data.table::shift(x$lead.values, type = "lead"), #These have to be calculated again bc of rows being removed
                        lead.starts = data.table::shift(x$starts, type = 'lead'))
        y[y$values == 1 & y$lag.values == 3,]
      })
      
      Wake.REMS.dat6 <- purrr::map(Wake.REMS.dat5, 5, .default = NA)
      Wake.REMS.dat7 <- purrr::map(Wake.REMS.dat5, 1)
      
      Latency.to.Wake.from.REMS <- purrr::map(Wake.REMS.dat6, 1, .default = NA) 
      Post.REMS.Wake.Onset.Bout.Length <- purrr::map_dfr(Wake.REMS.dat7, 1, .default = NA)
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Post-REMS.Wake.Onset'] <- as.integer((unlist(Latency.to.Wake.from.REMS[1:byBlocks]))) 
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Latency.to.Wake.from.REMS'] <- (as.integer((unlist(Latency.to.Wake.from.REMS[1:byBlocks])-1)*epoch.size))/60 #Latency to Wake from NREM
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Post-REMS.Wake.Onset.Bout.Length'] <- (as.integer(unlist(Post.REMS.Wake.Onset.Bout.Length[1:byBlocks])*epoch.size))/60 #Post-NREM Wake Onset Bout Length
      
      
      #Wake Latency from Sleep byBlocks
      Wake.Sleep.dat <- lapply(scores.sleepwake.rle.list, function(x){
        y <- data.frame(lengths = x$lengths, values = x$values, ends = cumsum(x$lengths), 
                        starts = cumsum(x$lengths) - x$lengths + 1)
        y[y$lengths >= Wake.cutoff & (y$values == 1 | y$values == 4),]
      })
      
      Wake.Sleep.dat2 <- lapply(Wake.Sleep.dat, function (x){
        y <- data.frame(lead.values = data.table::shift(x$values, type = "lead"), lag.values = data.table::shift(x$values, type = "lag"), 
                        lead.starts = data.table::shift(x$starts, type = "lead" ))
        y <- replace(y, is.na(y), 0)
      })
      
      Wake.Sleep.dat3 <- mapply(c, Wake.Sleep.dat, Wake.Sleep.dat2, SIMPLIFY = F)
      
      Wake.Sleep.dat4 <- lapply(Wake.Sleep.dat3, function(x) {
        y <- data.frame(lengths = x$lengths, values = x$values, ends = x$ends, 
                        starts = x$starts, lead.values = x$lead.values, 
                        lag.values = x$lag.values, lead.starts = x$lead.starts)
        y[!(y$values == 1 & y$lag.values == 1),]
      })
      
      Wake.Sleep.dat5 <- lapply(Wake.Sleep.dat4, function(x) {
        y <- data.frame(lengths = x$lengths, values = x$values, ends = x$ends, lag.values = data.table::shift(x$values, type = "lag"),
                        starts = x$starts, lead.values = data.table::shift(x$lead.values, type = "lead"), #These have to be calculated again bc of rows being removed
                        lead.starts = data.table::shift(x$starts, type = 'lead'))
        y[y$values == 1 & y$lag.values == 4,]
      })
      
      Wake.Sleep.dat6 <- purrr::map(Wake.Sleep.dat5, 5, .default = NA)
      Wake.Sleep.dat7 <- purrr::map(Wake.Sleep.dat5, 1)
      
      Latency.to.Wake.from.Sleep <- purrr::map(Wake.Sleep.dat6, 1, .default = NA) 
      Post.Sleep.Wake.Onset.Bout.Length <- purrr::map_dfr(Wake.Sleep.dat7, 1, .default = NA)
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Post-Sleep.Wake.Onset'] <- as.integer((unlist(Latency.to.Wake.from.Sleep[1:byBlocks]))) 
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Latency.to.Wake.from.Sleep'] <- (as.integer((unlist(Latency.to.Wake.from.Sleep[1:byBlocks])-1)*epoch.size))/60 #Latency to Wake from NREM
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Post-Sleep.Wake.Onset.Bout.Length'] <- (as.integer(unlist(Post.Sleep.Wake.Onset.Bout.Length[1:byBlocks])*epoch.size))/60 #Post-NREM Wake Onset Bout Length
      
      
      ###byBlocks Sleep Offset###
      
      
      #NREM Offset byBlocks
      NREM.Offset <- lapply(scores.rle.list,function(x){
        y <- data.frame(lengths = x$lengths, values = x$values, ends = cumsum(x$lengths))
        y[y$lengths >= NREM.cutoff & y$values ==2,]})
      
      NREM.Offset.Lengths <- purrr::map(NREM.Offset, 1)
      NREM.Offset.Lengths <- lapply(NREM.Offset.Lengths, function(x) tail(x, n = 1L)) #length
      NREM.Offset.Lengths[NREM.Offset.Lengths == "NULL"] <- NA
      
      NREM.Offset <- purrr::map(NREM.Offset, 3)
      NREM.Offset <- lapply(NREM.Offset, function(x) tail(x, n = 1L)) #lats
      NREM.Offset[NREM.Offset == "NULL"] <- NA
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'NREMS.Offset'] <- as.integer(unlist(NREM.Offset[1:byBlocks])) #NREM Offset
      mat.byBlocks[x.range1[counter]:x.range2[counter],'NREMS.Offset.Bout.Length'] <- (as.integer(unlist(NREM.Offset.Lengths[1:byBlocks])*epoch.size))/60 #NREM Offset Bout Length
      
      #REM Offset byBlocks
      REM.Offset <- lapply(scores.rle.list,function(x){
        y <- data.frame(lengths = x$lengths, values = x$values, ends = cumsum(x$lengths))
        y[y$lengths >= REM.cutoff & y$values ==3,]})
      
      REM.Offset.Lengths <- purrr::map(REM.Offset, 1)
      REM.Offset.Lengths <- lapply(REM.lats.lengths, function(x) tail(x, n = 1L)) #length
      REM.Offset.Lengths[REM.Offset.Lengths == "NULL"] <- NA
      
      REM.Offset <- purrr::map(REM.Offset, 3)
      REM.Offset <- lapply(REM.Offset, function(x) tail(x, n = 1L)) #lats
      REM.Offset[REM.Offset == "NULL"] <- NA
      
      # Check if REM.Offset contains non-empty and useful elements
      if (length(REM.Offset) > 0) {
        
        # Check if the specific subset is not empty and contains useful data
        if (length(REM.Offset[[1]]) > 0 && length(unlist(REM.Offset[1:byBlocks])) > 0) {
          mat.byBlocks[x.range1[counter]:x.range2[counter], "REMS.Offset"] <- as.integer(unlist(REM.Offset[1:byBlocks]))
        } 
        
        else {
          warning("REM.Offset[1:byBlocks] is empty or does not contain useful data.")
        }
      } 
      
      else {
        warning("REM.Offset is empty.")
      }
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'REMS.Offset.Bout.Length'] <- (as.integer(unlist(REM.Offset.Lengths[1:byBlocks])*epoch.size))/60 #REM Offset Bout Length
      
      #Sleep Offset byBlocks
      Sleep.Offset <- lapply(scores.sleepwake.rle.list,function(x){
        y <- data.frame(lengths = x$lengths, values = x$values, ends = cumsum(x$lengths))
        y[y$lengths >= Sleep.cutoff & y$values == 4,]})
      
      Sleep.Offset.Lengths <- purrr::map(Sleep.Offset, 1)
      Sleep.Offset.Lengths <- lapply(Sleep.Offset.Lengths, function(x) tail(x, n = 1L)) #length
      Sleep.Offset.Lengths[Sleep.Offset.Lengths == "NULL"] <- NA
      
      Sleep.Offset <- purrr::map(Sleep.Offset, 3)
      Sleep.Offset <- lapply(Sleep.Offset, function(x) tail(x, n = 1L)) #lats
      Sleep.Offset[Sleep.Offset == "NULL"] <- NA
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Sleep.Offset'] <- as.integer(unlist(Sleep.Offset[1:byBlocks])) #Sleep Offset
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Sleep.Offset.Bout.Length'] <- (as.integer(unlist(Sleep.Offset.Lengths[1:byBlocks])*epoch.size))/60 #Sleep Offset Bout Length
      
      
      ###byBlocks Transitions###
      
      
      #Wake-NREM Transition byBlocks
      WN <- lapply(scores.rle.list, function(x) {
        
        y <- data.frame(values = x$values, lead = data.table::shift(x$values, type ="lead"))
        y[y$values == 1 & y$lead == 2,]
        
      })
      
      WN <- purrr::map(WN, 1)
      WN <- lapply(WN, function(x) x[!is.na(x)])
      WN <- lapply(WN, function(x) length(x))
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Wake.NREMS.Transition.Count'] <- as.integer(unlist(WN[1:byBlocks]))
      
      WR <- lapply(scores.rle.list, function(x) {
        
        y <- data.frame(values = x$values, lead = data.table::shift(x$values, type ="lead"))
        y[y$values == 1 & y$lead == 3,]
        
      })
      
      WR <- purrr::map(WR, 1)
      WR <- lapply(WR, function(x) x[!is.na(x)])
      WR <- lapply(WR, function(x) length(x))
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Wake.REMS.Transition.Count'] <- as.integer(unlist(WR[1:byBlocks]))
      
      #NREM-Wake Transition byBlocks
      NW <- lapply(scores.rle.list, function(x) {
        
        y <- data.frame(values = x$values, lead = data.table::shift(x$values, type ="lead"))
        y[y$values == 2 & y$lead == 1,]
        
      })
      
      NW <- purrr::map(NW, 1)
      NW <- lapply(NW, function(x) x[!is.na(x)])
      NW <- lapply(NW, function(x) length(x))
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'NREMS.Wake.Transition.Count'] <- as.integer(unlist(NW[1:byBlocks]))
      
      
      #NREM-REM Transition byBlocks
      NR <- lapply(scores.rle.list, function(x) {
        
        y <- data.frame(lengths = x$lengths, values = x$values, lead = data.table::shift(x$values, type ="lead"))
        y[y$values == 2 & y$lead == 3,]
        
      })
      
      NR <- purrr::map(NR, 1)
      NR <- lapply(NR, function(x) x[!is.na(x)])
      NR <- lapply(NR, function(x) length(x))
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'NREMS.REMS.Transition.Count'] <- as.integer(unlist(NR[1:byBlocks]))
      
      #REM-Wake Transition byBlocks
      RW <- lapply(scores.rle.list, function(x) {
        
        y <- data.frame(lengths = x$lengths, values = x$values, lead = data.table::shift(x$values, type ="lead"))
        y[y$values == 3 & y$lead == 1,]
        
      })
      
      RW <- purrr::map(RW, 1)
      RW <- lapply(RW, function(x) x[!is.na(x)])
      RW <- lapply(RW, function(x) length(x))
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'REMS.Wake.Transition.Count'] <- as.integer(unlist(RW[1:byBlocks]))
      
      #REM-NREM Transition byBlocks
      RN <- lapply(scores.rle.list, function(x) {
        
        y <- data.frame(lengths = x$lengths, values = x$values, lead = data.table::shift(x$values, type ="lead"))
        y[y$values == 3 & y$lead == 2,]
        
      })
      
      RN <- purrr::map(RN, 1)
      RN <- lapply(RN, function(x) x[!is.na(x)])
      RN <- lapply(RN, function(x) length(x))
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'REMS.NREMS.Transition.Count'] <- as.integer(unlist(RN[1:byBlocks]))
      
      
      ###byBlocks Epoch Counts###
      
      
      #Wake Epoch Count byBlocks
      Wake.Epoch.Count <- sapply(scores.list, function(x) sum(x == 1, na.rm = TRUE))
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Wake.Epoch.Count'] <- as.integer(unlist(Wake.Epoch.Count[1:byBlocks]))
      
      #NREM Epoch Count byBlocks
      NREM.Epoch.Count <- sapply(scores.list, function(x) sum(x == 2, na.rm = TRUE))
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'NREMS.Epoch.Count'] <- as.integer(unlist(NREM.Epoch.Count[1:byBlocks]))
      
      #REM Epoch Count byBlocks
      REM.Epoch.Count <- sapply(scores.list, function(x) sum(x == 3, na.rm = TRUE))
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'REMS.Epoch.Count'] <- as.integer(unlist(REM.Epoch.Count[1:byBlocks]))
      
      
      ###byBlocks Vigilance State Seconds###
      
      
      #Wake Seconds byBlocks
      Wake.Seconds <- Wake.Epoch.Count*epoch.size
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Wake.Seconds'] <- as.integer(unlist(Wake.Seconds[1:byBlocks]))
      
      #NREM Seconds byBlocks
      NREM.Seconds <- NREM.Epoch.Count*epoch.size
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'NREMS.Seconds'] <- as.integer(unlist(NREM.Seconds[1:byBlocks]))
      
      #REM Seconds byBlocks
      REM.Seconds <- REM.Epoch.Count*epoch.size
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'REMS.Seconds'] <- as.integer(unlist(REM.Seconds[1:byBlocks]))
      
      ###byBlocks Vigilance State Percentages###
      
      
      #Wake Percentage byBlocks
      Wake.Percent <- sapply(scores.list, function(x) mean(x == 1, na.rm = TRUE))
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Wake.Percent'] <- as.numeric(unlist(Wake.Percent[1:byBlocks]))
      
      #NREM Percentage byBlocks
      NREM.Percent <- sapply(scores.list, function(x) mean(x == 2, na.rm = TRUE))
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'NREMS.Percent'] <- as.numeric(unlist(NREM.Percent[1:byBlocks]))
      
      #REM Percentage byBlocks
      REM.Percent <- sapply(scores.list, function(x) mean(x == 3, na.rm = TRUE))
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'REMS.Percent'] <- as.numeric(unlist(REM.Percent[1:byBlocks]))
      
      
      ###Episode/Bout Counts###
      
      
      #Arousal Count
      Arousal.Count <- sapply(scores.list, function(x) {  
        r <- rle(x)
        sum(r$values == 1)
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Arousal.Count'] <- as.integer(unlist(Arousal.Count[1:byBlocks]))
      
      #Microarousal Count
      if (Wake.cutoff <= 1) {
        
        Microarousal.Count <- sapply(scores.list, function(x) {  
          r <- rle(x)
          sum(r$lengths == 1 & r$values == 1)
        })
        
      } else {
        
        Microarousal.Count <- sapply(scores.list, function(x) {  
          r <- rle(x)
          sum(r$lengths < Wake.cutoff & r$values == 1)
        })
      }
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Microarousal.Count'] <- as.integer(unlist(Microarousal.Count[1:byBlocks]))
      
      #Wake Bout Count byBlocks
      Wake.Bout.Count <- sapply(scores.list, function(x) {  
        r <- rle(x)
        sum(r$lengths >= Wake.cutoff & r$values == 1)
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Wake.Bout.Count'] <- as.integer(unlist(Wake.Bout.Count[1:byBlocks]))
      
      #Wake Bout Counts by group byBlocks
      WBC.group1 <- sapply(scores.list, function(x) {  
        r <- rle(x)
        sum(r$values == 1 & r$lengths <= a[1])
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'WBC.group1'] <- as.integer(unlist(WBC.group1[1:byBlocks]))
      
      WBC.group2 <- sapply(scores.list, function(x) {  
        r <- rle(x)
        sum(r$values == 1 & r$lengths >= a[2] & r$lengths < a[3])
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'WBC.group2'] <- as.integer(unlist(WBC.group2[1:byBlocks]))
      
      WBC.group3 <- sapply(scores.list, function(x) {  
        r <- rle(x)
        sum(r$values == 1 & r$lengths >= a[3] & r$lengths < a[4])
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'WBC.group3'] <- as.integer(unlist(WBC.group3[1:byBlocks]))
      
      WBC.group4 <- sapply(scores.list, function(x) {  
        r <- rle(x)
        sum(r$values == 1 & r$lengths >= a[4] & r$lengths < a[5])
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'WBC.group4'] <- as.integer(unlist(WBC.group4[1:byBlocks]))
      
      WBC.group5 <- sapply(scores.list, function(x) {  
        r <- rle(x)
        sum(r$values == 1 & r$lengths >= a[5] & r$lengths < a[6])
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'WBC.group5'] <- as.integer(unlist(WBC.group5[1:byBlocks]))
      
      WBC.group6 <- sapply(scores.list, function(x) {  
        r <- rle(x)
        sum(r$values == 1 & r$lengths >= a[6] & r$lengths < a[7])
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'WBC.group6'] <- as.integer(unlist(WBC.group6[1:byBlocks]))
      
      WBC.group7 <- sapply(scores.list, function(x) {  
        r <- rle(x)
        sum(r$values == 1 & r$lengths >= a[7] & r$lengths < a[8])
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'WBC.group7'] <- as.integer(unlist(WBC.group7[1:byBlocks]))
      
      WBC.group8 <- sapply(scores.list, function(x) {  
        r <- rle(x)
        sum(r$values == 1 & r$lengths >= a[8] & r$lengths < a[9])
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'WBC.group8'] <- as.integer(unlist(WBC.group8[1:byBlocks]))
      
      WBC.group9 <- sapply(scores.list, function(x) {  
        r <- rle(x)
        sum(r$values == 1 & r$lengths >= a[9])
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'WBC.group9'] <- as.integer(unlist(WBC.group9[1:byBlocks]))
      
      
      #NREMS Bout Count byBlocks
      NREMS.Bout.Count <- sapply(scores.list, function(x) {  
        r <- rle(x)
        sum(r$lengths >= NREM.cutoff & r$values == 2)
      })
      
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'NREMS.Bout.Count'] <- as.integer(unlist(NREMS.Bout.Count[1:byBlocks]))
      
      
      #NREM Bout Counts by group byBlocks
      NBC.group1 <- sapply(scores.list, function(x) {  
        r <- rle(x)
        sum(r$values == 2 & r$lengths <= a[1])
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'NBC.group1'] <- as.integer(unlist(NBC.group1[1:byBlocks]))
      
      NBC.group2 <- sapply(scores.list, function(x) {  
        r <- rle(x)
        sum(r$values == 2 & r$lengths >= a[2] & r$lengths < a[3])
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'NBC.group2'] <- as.integer(unlist(NBC.group2[1:byBlocks]))
      
      NBC.group3 <- sapply(scores.list, function(x) {  
        r <- rle(x)
        sum(r$values == 2 & r$lengths >= a[3] & r$lengths < a[4])
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'NBC.group3'] <- as.integer(unlist(NBC.group3[1:byBlocks]))
      
      NBC.group4 <- sapply(scores.list, function(x) {  
        r <- rle(x)
        sum(r$values == 2 & r$lengths >= a[4] & r$lengths < a[5])
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'NBC.group4'] <- as.integer(unlist(NBC.group4[1:byBlocks]))
      
      NBC.group5 <- sapply(scores.list, function(x) {  
        r <- rle(x)
        sum(r$values == 2 & r$lengths >= a[5] & r$lengths < a[6])
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'NBC.group5'] <- as.integer(unlist(NBC.group5[1:byBlocks]))
      
      NBC.group6 <- sapply(scores.list, function(x) {  
        r <- rle(x)
        sum(r$values == 2 & r$lengths >= a[6] & r$lengths < a[7])
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'NBC.group6'] <- as.integer(unlist(NBC.group6[1:byBlocks]))
      
      NBC.group7 <- sapply(scores.list, function(x) {  
        r <- rle(x)
        sum(r$values == 2 & r$lengths >= a[7] & r$lengths < a[8])
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'NBC.group7'] <- as.integer(unlist(NBC.group7[1:byBlocks]))
      
      NBC.group8 <- sapply(scores.list, function(x) {  
        r <- rle(x)
        sum(r$values == 2 & r$lengths >= a[8] & r$lengths < a[9])
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'NBC.group8'] <- as.integer(unlist(NBC.group8[1:byBlocks]))
      
      NBC.group9 <- sapply(scores.list, function(x) {  
        r <- rle(x)
        sum(r$values == 2 & r$lengths >= a[9])
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'NBC.group9'] <- as.integer(unlist(NBC.group9[1:byBlocks]))
      
      
      #REMS Bout Count byBlocks
      REMS.Bout.Count <- sapply(scores.list, function(x) {  
        r <- rle(x)
        sum(r$lengths >= REM.cutoff & r$values == 3)
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'REMS.Bout.Count'] <- as.integer(unlist(REMS.Bout.Count[1:byBlocks]))
      
      #REM Bout Counts by group byBlocks
      RBC.group1 <- sapply(scores.list, function(x) {  
        r <- rle(x)
        sum(r$values == 3 & r$lengths <= a[1])
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'RBC.group1'] <- as.integer(unlist(RBC.group1[1:byBlocks]))
      
      RBC.group2 <- sapply(scores.list, function(x) {  
        r <- rle(x)
        sum(r$values == 3 & r$lengths >= a[2] & r$lengths < a[3])
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'RBC.group2'] <- as.integer(unlist(RBC.group2[1:byBlocks]))
      
      RBC.group3 <- sapply(scores.list, function(x) {  
        r <- rle(x)
        sum(r$values == 3 & r$lengths >= a[3] & r$lengths < a[4])
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'RBC.group3'] <- as.integer(unlist(RBC.group3[1:byBlocks]))
      
      RBC.group4 <- sapply(scores.list, function(x) {  
        r <- rle(x)
        sum(r$values == 3 & r$lengths >= a[4] & r$lengths < a[5])
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'RBC.group4'] <- as.integer(unlist(RBC.group4[1:byBlocks]))
      
      RBC.group5 <- sapply(scores.list, function(x) {  
        r <- rle(x)
        sum(r$values == 3 & r$lengths >= a[5] & r$lengths < a[6])
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'RBC.group5'] <- as.integer(unlist(RBC.group5[1:byBlocks]))
      
      RBC.group6 <- sapply(scores.list, function(x) {  
        r <- rle(x)
        sum(r$values == 3 & r$lengths >= a[6] & r$lengths < a[7])
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'RBC.group6'] <- as.integer(unlist(RBC.group6[1:byBlocks]))
      
      RBC.group7 <- sapply(scores.list, function(x) {  
        r <- rle(x)
        sum(r$values == 3 & r$lengths >= a[7] & r$lengths < a[8])
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'RBC.group7'] <- as.integer(unlist(RBC.group7[1:byBlocks]))
      
      RBC.group8 <- sapply(scores.list, function(x) {  
        r <- rle(x)
        sum(r$values == 3 & r$lengths >= a[8] & r$lengths < a[9])
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'RBC.group8'] <- as.integer(unlist(RBC.group8[1:byBlocks]))
      
      RBC.group9 <- sapply(scores.list, function(x) {  
        r <- rle(x)
        sum(r$values == 3 & r$lengths >= a[9])
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'RBC.group9'] <- as.integer(unlist(RBC.group9[1:byBlocks]))
      
      #Sleep Bout Count byBlocks
      Sleep.Bout.Count <- sapply(scores.sleepwake.list, function(x) {  
        r <- rle(x)
        sum(r$lengths >= Sleep.cutoff & r$values == 4)
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Sleep.Bout.Count'] <- as.integer(unlist(Sleep.Bout.Count[1:byBlocks]))
      
      
      ###Bout Duration Average###
      
      
      #Wake Bout Duration Average byBlocks
      Wake.Bout.Duration <- sapply(scores.list, function(x) {
        r <- rle(x)
        mean(r$lengths[r$values == 1])
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Avg.Wake.Bout.Duration'] <- (as.numeric(unlist(Wake.Bout.Duration[1:byBlocks])*epoch.size))/60
      
      #NREM Bout Duration Average byBlocks
      NREM.Bout.Duration <- sapply(scores.list, function(x) {
        r <- rle(x)
        mean(r$lengths[r$values == 2])
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Avg.NREMS.Bout.Duration'] <- (as.numeric(unlist(NREM.Bout.Duration[1:byBlocks])*epoch.size))/60
      
      #REM Bout Duration Average byBlocks
      REM.Bout.Duration <- sapply(scores.list, function(x) {
        r <- rle(x)
        mean(r$lengths[r$values == 3])
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Avg.REMS.Bout.Duration'] <- (as.numeric(unlist(REM.Bout.Duration[1:byBlocks])*epoch.size))/60
      
      #Sleep Bout Duration Average byBlocks
      Sleep.Bout.Duration <- sapply(scores.sleepwake.list, function(x) {
        r <- rle(x)
        mean(r$lengths[r$values == 4])
      })
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Avg.Sleep.Bout.Duration'] <- (as.numeric(unlist(Sleep.Bout.Duration[1:byBlocks])*epoch.size))/60
      
      
      ###Interbout Interval Average###
      
      
      #Wake Interbout Interval byBlocks
      Wake.dat <- lapply(scores.rle.list,function(x){
        y <- data.frame(lengths = x$lengths, values = x$values, ends = cumsum(x$lengths), starts = cumsum(x$lengths) - x$lengths + 1)
        y[y$lengths >= Wake.cutoff & y$values ==1,]})
      
      Wake.Interbout.Intervals <- lapply(Wake.dat,function(x){
        y <- data.frame(IBI = data.table::shift(x$starts, type = "lead") - x$ends - 1)
        y[!is.na(y)]
      })
      
      Wake.Interbout.Interval.Average <- lapply(Wake.Interbout.Intervals, function(x){
        y <- data.frame(IBIA = mean(x))
      })
      
      Wake.Interbout.Interval.Average <- purrr::map(Wake.Interbout.Interval.Average, 1, .default = NA)
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Avg.Wake.Interbout.Interval'] <- (as.numeric(unlist(Wake.Interbout.Interval.Average[1:byBlocks])*epoch.size))/60
      
      #NREMS Interbout Interval byBlocks
      NREM.dat <- lapply(scores.rle.list,function(x){
        y <- data.frame(lengths = x$lengths, values = x$values, ends = cumsum(x$lengths), starts = cumsum(x$lengths) - x$lengths + 1)
        y[y$lengths >= NREM.cutoff & y$values ==2,]})
      
      NREM.Interbout.Intervals <- lapply(NREM.dat,function(x){
        y <- data.frame(IBI = data.table::shift(x$starts, type = "lead") - x$ends - 1)
        y[!is.na(y)]
      })
      
      NREM.Interbout.Interval.Average <- lapply(NREM.Interbout.Intervals, function(x){
        y <- data.frame(IBIA = mean(x))
      })
      
      NREM.Interbout.Interval.Average <- purrr::map(NREM.Interbout.Interval.Average, 1, .default = NA)
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Avg.NREMS.Interbout.Interval'] <- (as.numeric(unlist(NREM.Interbout.Interval.Average[1:byBlocks])*epoch.size))/60
      
      #REMS Interbout Interval byBlocks
      REM.dat <- lapply(scores.rle.list,function(x){
        y <- data.frame(lengths = x$lengths, values = x$values, ends = cumsum(x$lengths), starts = cumsum(x$lengths) - x$lengths + 1)
        y[y$lengths >= REM.cutoff & y$values ==3,]})
      
      REM.Interbout.Intervals <- lapply(REM.dat,function(x){
        y <- data.frame(IBI = data.table::shift(x$starts, type = "lead") - x$ends - 1)
        y[!is.na(y)]
      })
      
      REM.Interbout.Interval.Average <- lapply(REM.Interbout.Intervals, function(x){
        y <- data.frame(IBIA = mean(x))
      })
      
      REM.Interbout.Interval.Average <- purrr::map(REM.Interbout.Interval.Average, 1, .default = NA)
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Avg.REMS.Interbout.Interval'] <- (as.numeric(unlist(REM.Interbout.Interval.Average[1:byBlocks])*epoch.size))/60
      
      
      ###byBlocks Sleep-Wake Propensity Measures###
      
      #Propensity to enter NREMS byBlocks
      Wake.NREMS.dat <- lapply(scores.rle.list, function(x){
        y <- data.frame(lengths = x$lengths, values = x$values, ends = cumsum(x$lengths), 
                        starts = cumsum(x$lengths) - x$lengths + 1)
        y[y$lengths >= Propensity.cutoff & (y$values == 1 | y$values == 2),]
      })
      
      Wake.NREMS.dat2 <- lapply(Wake.NREMS.dat, function (x){
        y <- data.frame(lead.values = data.table::shift(x$values, type = "lead"), lag.values = data.table::shift(x$values, type = "lag"), 
                        lead.starts = data.table::shift(x$starts, type = "lead" ))
        y <- replace(y, is.na(y), 0)
      })
      
      Wake.NREMS.dat3 <- mapply(c, Wake.NREMS.dat, Wake.NREMS.dat2, SIMPLIFY = F)
      
      Wake.NREMS.dat4 <- lapply(Wake.NREMS.dat3, function(x) {
        y <- data.frame(lengths = x$lengths, values = x$values, ends = x$ends, 
                        starts = x$starts, lead.values = x$lead.values, 
                        lag.values = x$lag.values, lead.starts = x$lead.starts)
        y[!(y$values == 1 & y$lag.values == 1),]
      })
      
      Wake.NREMS.dat5 <- lapply(Wake.NREMS.dat4, function(x) {
        y <- data.frame(lengths = x$lengths, values = x$values, ends = x$ends, 
                        starts = x$starts, lead.values = data.table::shift(x$values, type = "lead"), #These have to be calculated again bc of rows being removed
                        lead.starts = data.table::shift(x$starts, type = 'lead'))
        y[y$values == 1 & (y$lead.values == 2 | y$lead.values == 0),]
      })
      
      Wake.NREMS.dat6 <- lapply(Wake.NREMS.dat5, function(x){
        y <- data.frame(dist = (x$lead.starts) - x$start)
        y <- data.frame(mean.dist = mean(y$dist))
      })
      
      Propensity.to.enter.NREMS <- purrr::map(Wake.NREMS.dat6, 1, .default = NA)
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Propensity.to.enter.NREMS'] <- (as.numeric(unlist(Propensity.to.enter.NREMS))*epoch.size)/60
      
      #Propensity to Wake from NREMS
      Wake.NREMS.dat <- lapply(scores.rle.list, function(x){
        y <- data.frame(lengths = x$lengths, values = x$values, ends = cumsum(x$lengths), 
                        starts = cumsum(x$lengths) - x$lengths + 1)
        y[y$lengths >= Propensity.cutoff & (y$values == 1 | y$values == 2),]
      })
      
      Wake.NREMS.dat2 <- lapply(Wake.NREMS.dat, function (x){
        y <- data.frame(lead.values = data.table::shift(x$values, type = "lead"), lag.values = data.table::shift(x$values, type = "lag"), 
                        lead.starts = data.table::shift(x$starts, type = "lead" ))
        y <- replace(y, is.na(y), 0)
      })
      
      Wake.NREMS.dat3 <- mapply(c, Wake.NREMS.dat, Wake.NREMS.dat2, SIMPLIFY = F)
      
      Wake.NREMS.dat4 <- lapply(Wake.NREMS.dat3, function(x) {
        y <- data.frame(lengths = x$lengths, values = x$values, ends = x$ends, 
                        starts = x$starts, lead.values = x$lead.values, 
                        lag.values = x$lag.values, lead.starts = x$lead.starts)
        y[!(y$values == 2 & y$lag.values == 2),]
      })
      
      Wake.NREMS.dat5 <- lapply(Wake.NREMS.dat4, function(x) {
        y <- data.frame(lengths = x$lengths, values = x$values, ends = x$ends, lag.values = data.table::shift(x$values, type = "lag"),
                        starts = x$starts, lead.values = data.table::shift(x$values, type = "lead"), #These have to be calculated again bc of rows being removed
                        lag.starts = data.table::shift(x$starts, type = 'lag'))
        y[y$values == 1 & (y$lag.values == 2),]
      })
      
      Wake.NREMS.dat6 <- lapply(Wake.NREMS.dat5, function(x){
        y <- data.frame(dist = (x$starts) - x$lag.starts)
        y <- data.frame(mean.dist = mean(y$dist))
      })
      
      Propensity.to.Wake.from.NREMS <- purrr::map(Wake.NREMS.dat6, 1, .default = NA)
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Propensity.to.Wake.from.NREMS'] <- (as.numeric(unlist(Propensity.to.Wake.from.NREMS))*epoch.size)/60
      
      #Propensity to enter REMS byBlocks
      Wake.REMS.dat <- lapply(scores.rle.list, function(x){
        y <- data.frame(lengths = x$lengths, values = x$values, ends = cumsum(x$lengths), 
                        starts = cumsum(x$lengths) - x$lengths + 1)
        y[y$lengths >= Propensity.cutoff & (y$values == 1 | y$values == 3),]
      })
      
      Wake.REMS.dat2 <- lapply(Wake.REMS.dat, function (x){
        y <- data.frame(lead.values = data.table::shift(x$values, type = "lead"), lag.values = data.table::shift(x$values, type = "lag"), 
                        lead.starts = data.table::shift(x$starts, type = "lead" ))
        y <- replace(y, is.na(y), 0)
      })
      
      Wake.REMS.dat3 <- mapply(c, Wake.REMS.dat, Wake.REMS.dat2, SIMPLIFY = F)
      
      Wake.REMS.dat4 <- lapply(Wake.REMS.dat3, function(x) {
        y <- data.frame(lengths = x$lengths, values = x$values, ends = x$ends, 
                        starts = x$starts, lead.values = x$lead.values, 
                        lag.values = x$lag.values, lead.starts = x$lead.starts)
        y[!(y$values == 1 & y$lag.values == 1),]
      })
      
      Wake.REMS.dat5 <- lapply(Wake.REMS.dat4, function(x) {
        y <- data.frame(lengths = x$lengths, values = x$values, ends = x$ends, 
                        starts = x$starts, lead.values = data.table::shift(x$values, type = "lead"), #These have to be calculated again bc of rows being removed
                        lead.starts = data.table::shift(x$starts, type = 'lead'))
        y[y$values == 1 & (y$lead.values == 3 | y$lead.values == 0),]
      })
      
      Wake.REMS.dat6 <- lapply(Wake.REMS.dat5, function(x){
        y <- data.frame(dist = (x$lead.starts) - x$start)
        y <- data.frame(mean.dist = mean(y$dist))
      })
      
      Propensity.to.enter.REMS <- purrr::map(Wake.REMS.dat6, 1, .default = NA)
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Propensity.to.enter.REMS'] <- (as.numeric(unlist(Propensity.to.enter.REMS))*epoch.size)/60
      
      #Propensity to Wake from REMS
      Wake.REMS.dat <- lapply(scores.rle.list, function(x){
        y <- data.frame(lengths = x$lengths, values = x$values, ends = cumsum(x$lengths), 
                        starts = cumsum(x$lengths) - x$lengths + 1)
        y[y$lengths >= Propensity.cutoff & (y$values == 1 | y$values == 3),]
      })
      
      Wake.REMS.dat2 <- lapply(Wake.REMS.dat, function (x){
        y <- data.frame(lead.values = data.table::shift(x$values, type = "lead"), lag.values = data.table::shift(x$values, type = "lag"), 
                        lead.starts = data.table::shift(x$starts, type = "lead" ))
        y <- replace(y, is.na(y), 0)
      })
      
      Wake.REMS.dat3 <- mapply(c, Wake.REMS.dat, Wake.REMS.dat2, SIMPLIFY = F)
      
      Wake.REMS.dat4 <- lapply(Wake.REMS.dat3, function(x) {
        y <- data.frame(lengths = x$lengths, values = x$values, ends = x$ends, 
                        starts = x$starts, lead.values = x$lead.values, 
                        lag.values = x$lag.values, lead.starts = x$lead.starts)
        y[!(y$values == 3 & y$lag.values == 3),]
      })
      
      Wake.REMS.dat5 <- lapply(Wake.REMS.dat4, function(x) {
        y <- data.frame(lengths = x$lengths, values = x$values, ends = x$ends, lag.values = data.table::shift(x$values, type = "lag"),
                        starts = x$starts, lead.values = data.table::shift(x$values, type = "lead"), #These have to be calculated again bc of rows being removed
                        lag.starts = data.table::shift(x$starts, type = 'lag'))
        y[y$values == 1 & (y$lag.values == 3),]
      })
      
      Wake.REMS.dat6 <- lapply(Wake.REMS.dat5, function(x){
        y <- data.frame(dist = (x$starts) - x$lag.starts)
        y <- data.frame(mean.dist = mean(y$dist))
      })
      
      Propensity.to.Wake.from.REMS <- purrr::map(Wake.REMS.dat6, 1, .default = NA)
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Propensity.to.Wake.from.REMS'] <- (as.numeric(unlist(Propensity.to.Wake.from.REMS))*epoch.size)/60
      
      #Propensity to enter Sleep byBlocks
      Wake.Sleep.dat <- lapply(scores.sleepwake.rle.list, function(x){
        y <- data.frame(lengths = x$lengths, values = x$values, ends = cumsum(x$lengths), 
                        starts = cumsum(x$lengths) - x$lengths + 1)
        y[y$lengths >= Propensity.cutoff & (y$values == 1 | y$values == 4),]
      })
      
      Wake.Sleep.dat2 <- lapply(Wake.Sleep.dat, function (x){
        y <- data.frame(lead.values = data.table::shift(x$values, type = "lead"), lag.values = data.table::shift(x$values, type = "lag"), 
                        lead.starts = data.table::shift(x$starts, type = "lead" ))
        y <- replace(y, is.na(y), 0)
      })
      
      Wake.Sleep.dat3 <- mapply(c, Wake.Sleep.dat, Wake.Sleep.dat2, SIMPLIFY = F)
      
      Wake.Sleep.dat4 <- lapply(Wake.Sleep.dat3, function(x) {
        y <- data.frame(lengths = x$lengths, values = x$values, ends = x$ends, 
                        starts = x$starts, lead.values = x$lead.values, 
                        lag.values = x$lag.values, lead.starts = x$lead.starts)
        y[!(y$values == 1 & y$lag.values == 1),]
      })
      
      Wake.Sleep.dat5 <- lapply(Wake.Sleep.dat4, function(x) {
        y <- data.frame(lengths = x$lengths, values = x$values, ends = x$ends, 
                        starts = x$starts, lead.values = data.table::shift(x$values, type = "lead"), #These have to be calculated again bc of rows being removed
                        lead.starts = data.table::shift(x$starts, type = 'lead'))
        y[y$values == 1 & (y$lead.values == 4 | y$lead.values == 0),]
      })
      
      Wake.Sleep.dat6 <- lapply(Wake.Sleep.dat5, function(x){
        y <- data.frame(dist = (x$lead.starts) - x$start)
        y <- data.frame(mean.dist = mean(y$dist))
      })
      
      Propensity.to.enter.Sleep <- purrr::map(Wake.Sleep.dat6, 1, .default = NA)
      
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Propensity.to.enter.Sleep'] <- (as.numeric(unlist(Propensity.to.enter.Sleep))*epoch.size)/60
      
      #Propensity to Wake from Sleep byBlocks
      Wake.Sleep.dat <- lapply(scores.sleepwake.rle.list, function(x){
        y <- data.frame(lengths = x$lengths, values = x$values, ends = cumsum(x$lengths), 
                        starts = cumsum(x$lengths) - x$lengths + 1)
        y[y$lengths >= Propensity.cutoff & (y$values == 1 | y$values == 4),]
      })
      
      Wake.Sleep.dat2 <- lapply(Wake.Sleep.dat, function (x){
        y <- data.frame(lead.values = data.table::shift(x$values, type = "lead"), lag.values = data.table::shift(x$values, type = "lag"), 
                        lead.starts = data.table::shift(x$starts, type = "lead" ))
        y <- replace(y, is.na(y), 0)
      })
      
      Wake.Sleep.dat3 <- mapply(c, Wake.Sleep.dat, Wake.Sleep.dat2, SIMPLIFY = F)
      
      Wake.Sleep.dat4 <- lapply(Wake.Sleep.dat3, function(x) {
        y <- data.frame(lengths = x$lengths, values = x$values, ends = x$ends, 
                        starts = x$starts, lead.values = x$lead.values, 
                        lag.values = x$lag.values, lead.starts = x$lead.starts)
        y[!(y$values == 4 & y$lag.values == 4),]
      })
      
      Wake.Sleep.dat5 <- lapply(Wake.Sleep.dat4, function(x) {
        y <- data.frame(lengths = x$lengths, values = x$values, ends = x$ends, lag.values = data.table::shift(x$values, type = "lag"),
                        starts = x$starts, lead.values = data.table::shift(x$values, type = "lead"), #These have to be calculated again bc of rows being removed
                        lag.starts = data.table::shift(x$starts, type = 'lag'))
        y[y$values == 1 & (y$lag.values == 4),]
      })
      
      Wake.Sleep.dat6 <- lapply(Wake.Sleep.dat5, function(x){
        y <- data.frame(dist = (x$starts) - x$lag.starts)
        y <- data.frame(mean.dist = mean(y$dist))
      })
      
      Propensity.to.Wake.from.Sleep <- purrr::map(Wake.Sleep.dat6, 1, .default = NA)
      
      mat.byBlocks[x.range1[counter]:x.range2[counter],'Propensity.to.Wake.from.Sleep'] <- (as.numeric(unlist(Propensity.to.Wake.from.Sleep))*epoch.size)/60
      
    } #End of byBlocks (in loop)  
  } #End of loop through FileNames
  
  
  if(byTotal == T){
    
    
    #byTotal Total Sleep Time/Period
    Total.Sleep.Time <- as.integer(mat.byTotal[,'NREMS.Seconds']) + as.integer(mat.byTotal[,'REMS.Seconds'])
    #NREM.Seconds + REM.Seconds
    
    mat.byTotal[,'Total.Sleep.Time'] <- as.integer(Total.Sleep.Time) 
    
    
    #byTotal WASO (
    WASO <- length(scores)*epoch.size - as.integer(mat.byTotal[,'Total.Sleep.Time']) - (as.integer(mat.byTotal[,'Latency.to.NREMS']))*60
    #WASO <- floor(length(scores)/byBlocks) - Total.Sleep.Time - NREM.Latency
    #(NREM.Lat$starts[1]-1)
    
    mat.byTotal[,'WASO'] <- as.numeric(WASO)/60
    
    #byTotal WASF (minutes)
    WASF <- as.integer(length(scores)*epoch.size) - (as.integer(mat.byTotal[,'Sleep.Offset'])*epoch.size)
    #Total.Recording.Time - Sleep.Offset
    
    mat.byTotal[,'WASF'] <- WASF/60
    
    #byTotal Sleep Fragmentation Index (number per minute)
    Sleep.Fragmentation.Index <- as.integer(mat.byTotal[,'Arousal.Count']) / ((as.integer(mat.byTotal[,'Total.Sleep.Time']))/60)
    
    mat.byTotal[,'Sleep.Fragmentation.Index'] <- as.numeric(Sleep.Fragmentation.Index)
    
    #byTotal Sleep Efficiency
    Sleep.Efficiency <- as.integer(mat.byTotal[,'Total.Sleep.Time']) / as.integer(length(scores)*epoch.size)
    #Total.Sleep.Time / (floor(length(scores)/byBlocks)*epoch.size)
    
    mat.byTotal[,'Sleep.Efficiency'] <- as.numeric(Sleep.Efficiency)
    
  }#End of byTotal (after loop)
  
  
  
  if(!is.null(byBlocks)){
    
    #byBlocks Total Sleep Time/Period
    Total.Sleep.Time <- as.integer(mat.byBlocks[,'NREMS.Seconds']) + as.integer(mat.byBlocks[,'REMS.Seconds']) 
    #NREM.Seconds + REM.Seconds
    
    mat.byBlocks[,'Total.Sleep.Time'] <- Total.Sleep.Time 
    
    
    #byBlocks WASO (minutes)
    WASO <- as.integer(floor(length(scores)/byBlocks)*epoch.size) - as.integer(mat.byBlocks[,'Total.Sleep.Time']) - (as.integer(mat.byBlocks[,'Latency.to.NREMS']))*60
    #WASO <- floor(length(scores)/byBlocks) - Total.Sleep.Time - NREM.Latency
    
    mat.byBlocks[,'WASO'] <- WASO/60
    
    
    #byBlocks WASF (minutes)
    WASF <- floor(length(scores)/byBlocks)*epoch.size - (as.integer(mat.byBlocks[,'Sleep.Offset'])*epoch.size)
    #- Sleep.Offset
    mat.byBlocks[,'WASF'] <- WASF/60
    
    
    #byBlocks Sleep Fragmentation Index (number per minute)
    Sleep.Fragmentation.Index <- as.integer(mat.byBlocks[,'Arousal.Count']) / (as.integer(mat.byBlocks[,'Total.Sleep.Time'])/60)
    #Wake.Bout.Count/(Wake.Bout.Count + NREM.Bout.Count + REM.Bout.Count)
    
    mat.byBlocks[,'Sleep.Fragmentation.Index'] <- Sleep.Fragmentation.Index
    
    
    #byBlocks Sleep Efficiency
    Sleep.Efficiency <- as.integer(mat.byBlocks[,'Total.Sleep.Time']) / as.integer((floor(length(scores)/byBlocks)*epoch.size))
    #Total.Sleep.Time / (floor(length(scores)/byBlocks)*epoch.size)
    
    mat.byBlocks[,'Sleep.Efficiency'] <- Sleep.Efficiency
    
  }#End of byBlocks (after loop)
  
  
  #Normalize to Baseline
  
  if(!is.null(normalization)){
    if(!is.null(byBlocks)) {
      mat.byBlocks <- mat.byBlocks %>% 
        group_by(Animal.ID, Grouping.Factor)
      
      BL <- subset(mat.byBlocks, Grouping.Factor == Normalization)
      num_cols <- unlist(lapply(BL, is.numeric))
      to.subtract <- BL[ , num_cols]
      
      minus.BL <- subset(mat.byBlocks, Grouping.Factor != Normalization)
      num_cols <- unlist(lapply(minus.BL, is.numeric))
      subtract.from <- minus.BL[ , num_cols]  
      
      name_cols <- unlist(lapply(BL, is.character))
      names <- BL[, name_cols]
      
      Norm.data <- to.subtract - subtract.from
      Norm.data <- cbind(names, Norm.Data)
      
      mat.byBlocks <- Norm.data
    } else if (byTotal == T) {
      
      mat.byTotal <- mat.byTotal %>% 
        group_by(Animal.ID, Grouping.Factor)
      
      BL <- subset(mat.byTotal, Grouping.Factor == Normalization)
      num_cols <- unlist(lapply(BL, is.numeric))
      to.subtract <- BL[ , num_cols]
      
      minus.BL <- subset(mat.byTotal, Grouping.Factor != Normalization)
      num_cols <- unlist(lapply(minus.BL, is.numeric))
      subtract.from <- minus.BL[ , num_cols]  
      
      name_cols <- unlist(lapply(BL, is.character))
      names <- BL[, name_cols]
      
      Norm.data <- to.subtract - subtract.from
      Norm.data <- cbind(names, Norm.Data)
      
      mat.byTotal <- Norm.data
    }
  }
  
  #Change from long to wide
  
  if(!is.null(byBlocks)){
    if (data.format == "long") {
      dat.byBlocks <- as.data.frame(mat.byBlocks, stringsAsFactors = F) #Matrices can only accept one data class; all classes converted to character
      dat.byBlocks[4:82] <- lapply(dat.byBlocks[4:82], function(x) as.numeric(x)) #Convert non-factor values to numeric
      dat.byBlocks$Animal.ID <- as.factor(dat.byBlocks$Animal.ID)
      dat.byBlocks$Grouping.Factor<- as.factor(dat.byBlocks$Grouping.Factor)
      dat.byBlocks$Blocks <- as.factor(dat.byBlocks$Blocks)
    } else if (data.format == "wide") {
      dat.byBlocks <- as.data.frame(mat.byBlocks, stringsAsFactors = F) #Matrices can only accept one data class; all classes converted to character
      dat.byBlocks[4:82] <- lapply(dat.byBlocks[4:82], function(x) as.numeric(x)) #Convert non-factor values to numeric
      dat.byBlocks$Animal.ID <- as.factor(dat.byBlocks$Animal.ID)
      dat.byBlocks$Grouping.Factor<- as.factor(dat.byBlocks$Grouping.Factor)
      dat.byBlocks$Total.Hours <- as.factor(dat.byBlocks$Total.Hours)
      
      dat.byBlocks <- reshape(dat.byBlocks, idvar="Animal.ID", timevar="Blocks", direction="wide")
    }
  }
  
  if(byTotal == TRUE){
    if (data.format == "long") {
      dat.byTotal <- as.data.frame(mat.byTotal, stringsAsFactors = F) #Matrices can only accept one data class; all classes converted to character
      dat.byTotal[4:76] <- lapply(dat.byTotal[4:76], function(x) as.numeric(x)) #Convert non-factor values to numeric
      dat.byTotal$Animal.ID <- as.factor(dat.byTotal$Animal.ID)
      dat.byTotal$Grouping.Factor<- as.factor(dat.byTotal$Grouping.Factor)
      dat.byTotal$Total.Hours <- as.factor(dat.byTotal$Total.Hours)
    } else if (data.format == "wide") {
      dat.byTotal <- as.data.frame(mat.byTotal, stringsAsFactors = F) #Matrices can only accept one data class; all classes converted to character
      dat.byTotal[4:76] <- lapply(dat.byTotal[4:76], function(x) as.numeric(x)) #Convert non-factor values to numeric
      dat.byTotal$Animal.ID <- as.factor(dat.byTotal$Animal.ID)
      dat.byTotal$Grouping.Factor<- as.factor(dat.byTotal$Grouping.Factor)
      dat.byTotal$Total.Hours <- as.factor(dat.byTotal$Total.Hours)
      
      dat.byTotal <- reshape(dat.byTotal, idvar="Animal.ID", timevar="Total.Hours", direction="wide")
    }
  }
  
  if(is.null(save.name)){
    
    if(byTotal == TRUE && !is.null(byBlocks)){
      
      dat.byTotal <<- dat.byTotal
      dat.byBlocks <<- dat.byBlocks
      
    } else if (byTotal == TRUE && is.null(byBlocks)){
      
      dat.byTotal <<- dat.byTotal
      
    } else if (byTotal == FALSE && !is.null(byBlocks)){
      
      dat.byBlocks <<- dat.byBlocks
      
    }
    
  } else if(!is.null(save.name)){
    
    if( byTotal == TRUE && !is.null(byBlocks)){
      
      dat.byTotal <<- dat.byTotal
      dat.byBlocks <<- dat.byBlocks
      
      write.table(dat.byTotal, file = paste0(save.name,"_byTotal",".tsv"), sep = "\t", row.names = FALSE)
      write.table(dat.byBlocks, file = paste0(save.name,"_byBlocks",".tsv"), sep = "\t", row.names = FALSE)
      
    } else if (byTotal == TRUE && is.null(byBlocks)){
      
      dat.byTotal <<- dat.byTotal
      
      #write.table(dat.byTotal, file = sprintf("%s_byTotal.tsv", save.name), row.names = FALSE, sep = "\t")
      write.xlsx(dat.byTotal, file = paste0(save.name, "_byTotal.xlsx"))
    } else if (byTotal == FALSE && !is.null(byBlocks)){
      
      dat.byBlocks <<- dat.byBlocks
      
      # write.csv(dat.byBlocks, file = paste0(save.name,"_byBlocks",".csv"), row.names = FALSE)
      write.table(dat.byBlocks, file = paste0(save.name,"_byBlocks",".tsv"), row.names = FALSE, sep = "\t")
    }
    
  }
  
  
} #End of Function 

