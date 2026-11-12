# =============================================================================
# trim_asc_batch.R
#
# Trims every .asc raster in a directory to a fixed set of rectangular regions.
#
#   <old name>.ASC  ->  <old name>_<region>.ASC   (one file per region)
#
# Run with:   source("trim_asc_batch.R")
#       or:   Rscript trim_asc_batch.R
# Package:    install.packages("terra")
# =============================================================================

library(terra)

# Stop GDAL writing a .aux.xml statistics sidecar next to every output file
Sys.setenv(GDAL_PAM_ENABLED = "NO")

# ---- Settings ---------------------------------------------------------------

in_dir    <- "D:/Frogs/ASC/ASC_selected"
out_dir   <- "D:/Frogs/ASC/ASC_selected_trimmed"
na_value  <- -9999
overwrite <- TRUE          # FALSE = skip outputs that already exist
snap      <- "out"         # "out" keeps outputs aligned to the input grid

# name, West (xmin), East (xmax), South (ymin), North (ymax)
regions <- data.frame(
  name  = c("SA",      "NA",       "Africa",  "Asia",    "AusNZ"),
  west  = c( -87.758,  -129.872,   -19.518,    59.758,   110.527),
  east  = c( -32.063,   -59.128,    60.650,   153.849,   180.000),
  south = c( -57.110,    12.178,   -35.328,   -11.424,   -47.445),
  north = c(  12.710,    51.481,    37.989,    45.529,    -9.668),
  stringsAsFactors = FALSE
)

# ---- Checks -----------------------------------------------------------------

stopifnot(dir.exists(in_dir))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
stopifnot(dir.exists(out_dir))

# "NA" is a legitimate region label here, but make sure it is the *string* "NA"
# and not a true missing value (which would produce files called "_NA.ASC" by
# accident if the table were ever read in from a CSV).
regions$name[is.na(regions$name)] <- "NA"

bad <- with(regions, west >= east | south >= north)
if (any(bad))
  stop("Bad bounds for region(s): ", paste(regions$name[bad], collapse = ", "))

files <- list.files(in_dir, pattern = "\\.asc$", ignore.case = TRUE,
                    full.names = TRUE)
if (!length(files)) stop("No .asc files found in: ", in_dir)

cat(sprintf("Found %d .asc file%s and %d regions -> %d outputs\n\n",
            length(files), if (length(files) == 1) "" else "s",
            nrow(regions), length(files) * nrow(regions)))

# ---- Trim -------------------------------------------------------------------

n_ok <- 0L; n_skip <- 0L; n_err <- 0L
log  <- data.frame()

for (f in files) {
  
  base <- tools::file_path_sans_ext(basename(f))
  ext  <- tools::file_ext(f)                       # preserves .asc / .ASC case
  
  r <- tryCatch(rast(f), error = function(e) e)
  if (inherits(r, "error")) {
    cat(sprintf("[ERROR] %s: cannot read (%s)\n", basename(f), conditionMessage(r)))
    n_err <- n_err + nrow(regions)
    next
  }
  
  # Bare .asc files usually carry no CRS; assume lon/lat so the boxes apply directly
  if (!nzchar(crs(r))) crs(r) <- "EPSG:4326"
  
  r_ext <- ext(r)
  
  for (i in seq_len(nrow(regions))) {
    
    nm       <- regions$name[i]
    out_file <- file.path(out_dir, sprintf("%s_%s.%s", base, nm, ext))
    reg_ext  <- ext(regions$west[i], regions$east[i],
                    regions$south[i], regions$north[i])
    
    if (file.exists(out_file) && !overwrite) {
      cat(sprintf("[SKIP]  %s already exists\n", basename(out_file)))
      n_skip <- n_skip + 1L
      next
    }
    
    # Does the region actually touch this raster?
    if (is.null(intersect(r_ext, reg_ext))) {
      cat(sprintf("[SKIP]  %s: region does not overlap raster\n", basename(out_file)))
      n_skip <- n_skip + 1L
      next
    }
    
    res <- tryCatch({
      cr <- crop(r, reg_ext, snap = snap)
      writeRaster(cr, out_file, filetype = "AAIGrid",
                  NAflag = na_value, overwrite = TRUE)
      v <- as.vector(ext(cr))
      cat(sprintf("[OK]    %-45s %4d rows x %4d cols  [%.3f, %.3f, %.3f, %.3f]\n",
                  basename(out_file), nrow(cr), ncol(cr),
                  v[["xmin"]], v[["xmax"]], v[["ymin"]], v[["ymax"]]))
      data.frame(file = basename(f), region = nm, output = basename(out_file),
                 status = "ok", rows = nrow(cr), cols = ncol(cr),
                 xmin = v[["xmin"]], xmax = v[["xmax"]],
                 ymin = v[["ymin"]], ymax = v[["ymax"]],
                 stringsAsFactors = FALSE)
    }, error = function(e) {
      cat(sprintf("[ERROR] %s: %s\n", basename(out_file), conditionMessage(e)))
      data.frame(file = basename(f), region = nm, output = basename(out_file),
                 status = paste("error:", conditionMessage(e)),
                 rows = NA, cols = NA, xmin = NA, xmax = NA, ymin = NA, ymax = NA,
                 stringsAsFactors = FALSE)
    })
    
    if (identical(res$status, "ok")) n_ok <- n_ok + 1L else n_err <- n_err + 1L
    log <- rbind(log, res)
  }
}

# ---- Summary ----------------------------------------------------------------

if (nrow(log))
  write.csv(log, file.path(out_dir, "trim_log.csv"), row.names = FALSE)

cat(sprintf("\nDone. %d written, %d skipped, %d failed.\nOutput: %s\n",
            n_ok, n_skip, n_err, normalizePath(out_dir, winslash = "/")))