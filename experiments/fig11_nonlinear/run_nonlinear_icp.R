
library(nonlinearICP)
library(CondIndTests)

# Usage: Rscript run_nonlinear_icp.R 100 2
# Inputs: data/data_<n>_<trial>.csv and data/true_<n>_<trial>.csv
argv <- commandArgs(trailingOnly=TRUE)
if(length(argv)<2) stop('Usage: Rscript run_nonlinear_icp.R <n> <trial>')
n_ <- as.integer(argv[1]);l <- as.integer(argv[2])
script_path <- sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1])
if(!is.na(script_path)) setwd(dirname(normalizePath(script_path)))
dir.create('nonlinear_results',showWarnings=FALSE)
cat('n:',n_,'trial:',l,'\n')

fourier_basis <- function(X, K=1) {
  X <- as.matrix(X)
  do.call(cbind, lapply(seq_len(ncol(X)), function(j)
    do.call(cbind, lapply(seq_len(K), function(k)
      cbind(sin(k * X[, j]), cos(k * X[, j]))
    ))
  ))
}

RP_fourier <- function(
    Y, environment, X, alpha=.05, verbose=FALSE,
    K=1, nSim=100, ntree=100
) {
  X <- as.matrix(X)
  # nonlinearICP represents the empty set by a column of ones
  if (ncol(X) == 1 && all(X == 1))
    return(ResidualPredictionTest(
      Y, environment, X,
      alpha=alpha, verbose=verbose,
      nSim=nSim, ntree=ntree
    ))
  XBasis <- fourier_basis(X, K)
  ResidualPredictionTest(
    Y, environment, X,
    alpha=alpha, verbose=verbose,
    basis="provided",
    XBasis=XBasis,
    resid_type="OLS",
    nSim=nSim,
    ntree=ntree,
    useX=TRUE
  )
}

# ======================================================================

filename = paste0("./data/data_", n_, "_", l, ".csv")
if(!file.exists(filename)) stop("Missing ",filename,". First run generate_icp_data.py with --n and --trial.")
dat <- read.csv(filename)
X <- as.matrix(dat[, grep("^X", names(dat))])
Y <- dat$Y
E <- factor(dat$E)

filename = paste0("./data/true_", n_, "_", l, ".csv")
if(!file.exists(filename)) stop("Missing ",filename,". First run generate_icp_data.py with --n and --trial.")
S_star <- read.csv(filename)$true_invariant

# ======================================================================


t1 = system.time(
  fit <- nonlinearICP(
    X=X, Y=Y, environment=E,
    condIndTest=InvariantResidualDistributionTest,
    argsCondIndTest=list(fitWithGam=TRUE),
    alpha=.05,
    maxSizeSets=2,
    stopIfEmpty=TRUE,
    retrieveDefiningsSets=FALSE,
    speedUp=FALSE,
    seed=1,
    verbose=FALSE
  )
)

S_hat = fit$retrievedCausalVars
runtime = as.numeric(t1["elapsed"])

rs = c(n_, l, "nonlinearICP", 
       setequal(S_star, S_hat),
       length(intersect(S_star, S_hat)),
       runtime
)

filename = paste0("./nonlinear_results/RS_", n_, "_", l, ".RDS") 
saveRDS(rs, filename)
cat('finished', 'n=', n_, 'l=', l, '\n')

