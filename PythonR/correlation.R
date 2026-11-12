# correlation.R
# Computes a Pearson correlation matrix for the most important environmental
# layers identified after the global maxnet run.
# Packages: raster, sp.
# Depends on get_top_permutation_importance() being sourced from
# permutation_analyzer.R before this function is called.

library(raster)
library(sp)

# Computes pairwise Pearson correlation among top environmental variables
# and writes the matrix to correlation.csv.
run_correlation <- function(maxent_results_csv, threshold, top_count, paths) {

  # Step 1: Identify top variables
  top_vars <- get_top_permutation_importance(maxent_results_csv, threshold,
                                              top_count)

  if (length(top_vars) == 0) {
    stop("No variables passed the importance threshold. Cannot compute correlation.")
  }

  cat("Top variables for correlation:", paste(top_vars, collapse = ", "), "\n")

  # Step 2: Build full paths to .asc files
  asc_paths <- file.path(paths$clipped_layers_path, paste0(top_vars, ".asc"))

  # Step 3: Verify all files exist
  missing_names <- character(0)
  for (f in asc_paths) {
    if (!file.exists(f)) {
      missing_names <- c(missing_names, basename(f))
    }
  }

  if (length(missing_names) > 0) {
    stop("Missing .asc files for variables: ",
         paste(missing_names, collapse = ", "))
  }

  # Step 4: Stack rasters
  stk <- raster::stack(asc_paths)

  # Step 5: Convert to brick
  brk <- raster::brick(stk)

  # Step 6: Handle NoData values (common convention: large negatives like -9999)
  brk[brk < -100] <- NA

  # Step 7: Compute Pearson correlation
  corr_result <- raster::layerStats(brk, "pearson", na.rm = TRUE)

  # Step 8: Write to CSV
  write.csv(corr_result$pearson, paths$correlation_path)

  # Step 9: Confirm
  cat("Correlation matrix saved to:", paths$correlation_path, "\n")

  return(invisible(paths$correlation_path))
}
