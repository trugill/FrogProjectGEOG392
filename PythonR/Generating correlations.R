# Generating correlations in R

# Load the relevant library
library(raster)

# Display instructions to the user as a single block of text
cat("
===============================================================================
Please create a folder with only the *.asc files you wish to calculate 
correlations on. 

Then, make sure that each of the file names is no more than 5 characters 
in length. 

For example, you might change a file called 'aridity.asc' to 'aridi.asc'.

Once you have done this, hit <Enter>
===============================================================================
\n")

# Wait for user to press Enter before proceeding
readline(prompt = "")

# Prompt user to select the folder
cat("\nPlease select the folder containing the *.asc files in the dialog box that appears.\n")

# Specify the folder containing the .asc files
asc_folder <- "D:/Frogs/ASC/ASC_selected"

# Verify that the folder exists
if (!dir.exists(asc_folder)) {
  stop("Folder does not exist: ", asc_folder)
}

# Read in all the *.asc files from the chosen directory
list_file <- list.files(asc_folder, pattern = "*.ASC", full.names = TRUE) 

# Check if there are any .asc files in the directory
if (length(list_file) == 0) {
  stop("No .asc files found in the selected folder. Exiting program.")
}

# Stack and process the raster files
ac <- stack(list_file)
ac

ac.brick <- brick(ac)
ac.brick
ac.brick[ac.brick < -100] <- NA
ac.brick

# Compute Pearson correlation
rc2.corr <- layerStats(ac.brick, 'pearson', na.rm = TRUE)
rc2.corr

# Prompt user to select a location to save the CSV file
cat("
===============================================================================
Please choose where to save the correlation matrix file.

You will need to specify both the file name and the file type.

For example, you might enter: correlations.csv
===============================================================================
\n")

# Ask the user for the output file name and location
save_file <- file.choose(new = TRUE)

# Save output
write.csv(rc2.corr, save_file)

# Print completion message
cat("\nCorrelation matrix successfully computed and saved to:", save_file, "\n")
