
  # Prompt the user to select the *.csv file
  cat("Please select the *.csv file\n")
  
  # Allow the user to choose the file
  file <- file.choose()
  
  # Read the selected file
  corr <- read.csv(file, header = TRUE)
  
  # Remove the first and last columns
  corr <- corr[, -c(1, ncol(corr))]  # This ensures both first and last columns are removed
  
  # Update column names by removing the "pearson.correlation.coefficient." prefix
  colnames(corr) <- gsub("^pearson.correlation.coefficient.", "", colnames(corr))
  
  # Print the modified data
  print(corr)
  
  # Prompt the user to enter the correlation threshold
  corr_threshold <- as.numeric(readline(prompt = "Enter correlation threshold (default 0.8): "))
  
  if (is.na(corr_threshold)) {
    corr_threshold <- 0.9  # Default value if the user does not input a threshold
  }
  
  
  # Prompt the user to choose the output directory
  cat("Please select the output directory\n")
  output_dir <- choose.dir(default = "", caption = "Select Output Directory")
  
  # Check if directory was selected
  if (is.null(output_dir)) {
    stop("No output directory selected. Exiting function.")
  }
  
  # Create output directory if it doesn't exist
  if (!dir.exists(output_dir)) {
    dir.create(output_dir)
  }
  
  # Create a function to generate combinations dynamically
  generate_combinations <- function(corr, ncols, corr_threshold, max_combo_size) {
    for (size in 2:max_combo_size) {
      # Generate all combinations for this size
      combos <- combn(ncols, size, simplify = FALSE)
      for (combo in combos) {
        # Check if the combination satisfies the correlation threshold
        is_valid <- TRUE
        for (i in 1:(length(combo) - 1)) {
          for (j in (i + 1):length(combo)) {
            if (abs(corr[combo[i], combo[j]]) >= corr_threshold) {
              is_valid <- FALSE
              break
            }
          }
          if (!is_valid) break
        }
        
        # If valid, process the combination
        if (is_valid) {
          combo_name <- paste(colnames(corr)[combo], collapse = "_")
          
          # Create a folder for the combination in the output directory
          dir.create(file.path(output_dir, combo_name))
          
          # Save the combination to a CSV file
          write(combo_name, file = file.path(output_dir, paste0("combinations_", size, ".csv")), append = TRUE)
        }
      }
    }
  }
  
  # Get the number of columns in the data frame
  ncols <- ncol(corr)
  
  # Generate combinations for sizes 2 through 7 (or any max size you prefer)
  max_combo_size <- 7
  generate_combinations(corr, ncols, corr_threshold, max_combo_size)
  
  cat("Processing complete, folders created!\n")

