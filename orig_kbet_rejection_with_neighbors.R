# save rejection status of (original) kBET with neighbors

#truncated normal distribution distribution function
ptnorm <- function(x, mu, sd, a=0, b=1, alpha = 0.05, verbose = FALSE){
  #this is the cumulative density of the truncated normal distribution
  #x ~ N(mu, sd^2), but we condition on a <= x <= b
  if (!is.na(x)){


    if (a > b) {
      warning("Lower and upper bound are interchanged.")
      tmp <- a
      a <- b
      b <- tmp
    }

    if (sd <= 0 || is.na(sd)) {
      if (verbose) {
        warning("Standard deviation must be positive.")
      }
      if (alpha <= 0) {
        stop("False positive rate alpha must be positive.")
      }
      sd <- alpha
    }
    if (x < a || x > b) {
      warning("x out of bounds.")
      cdf <- as.numeric(x > a)
    } else {
      alp <- pnorm((a - mu) / sd)
      bet <- pnorm((b - mu) / sd)
      zet <- pnorm((x - mu) / sd)
      cdf <- (zet - alp) / (bet - alp)
    }
    cdf
  } else {
    return(NA)
  }
}

# residual score function of kBET
residual_score_batch <- function(knn.set, class.freq, batch) {
    # knn.set: indices of nearest neighbors
    # class.freq: global batch proportions
    # batch: batch labels

    # return NA if all values of neighborhood are NA (which may arise from subsampling a knn-graph)
    if (all(is.na(knn.set))) {
        return(NA)
    }
    else{
        # extracts the batch labels of all neighbors (excluding NA)
        # computes local batch frequencies (observed)
        freq.env <- table(batch[knn.set[!is.na(knn.set)]]) / length(!is.na(knn.set))
        
        # create zero vector with length of batches and initialize it with freqs of batches
        full.classes <- rep(0, length(class.freq$class))
        full.classes[class.freq$class %in% names(freq.env)] <- freq.env
        
        # global batch props (expected)
        exp.freqs <- class.freq$freq

        # compute chi-square test statistics
        ## sum((observed - expected)^2 / expected)
        chi_sq_statistic <- sum((full.classes - exp.freqs)^2 / exp.freqs)
        
        return(chi_sq_statistic)
    }
}

# core function of kBET; called for each sample
chi_batch_test <- function(knn.set, class.freq, batch, df) {
    # knn.set: indices of nearest neighbors
    # class.freq: global batch proportions
    # batch: batch labels
    # df: degrees of freedom

    # return NA if all values of neighborhood are NA (which may arise from subsampling a knn-graph)
    if (all(is.na(knn.set))) {
        return(NA)
    }
    else {
        # extracts the batch labels of all neighbors (excluding NA)
        # computes local batch counts (observed)
        freq.env <- table(batch[knn.set[!is.na(knn.set)]])

        # create zero vector with length of batches and initialize it with counts of batches
        full.classes <- rep(0, length(class.freq$class))
        full.classes[class.freq$class %in% names(freq.env)] <- freq.env

        # global batch counts (for this sample)
        exp.freqs <- class.freq$freq * length(knn.set)

        # compute chi-square test statistics
        ## sum((observed - expected)^2 / expected)
        chi.sq.value <- sum((full.classes - exp.freqs)^2 / exp.freqs)

        # calculate p-value
        result <- 1 - pchisq(chi.sq.value, df)

        # I actually would like to know when 'NA' arises.
        if (is.na(result)) {
            return(NA)
        }
        else {
           return(result)
        }
    }
}

rejection_with_neighbors <- function(df, batch, celltype, k0, sc_dir, do.pca = FALSE, knn = NULL, alpha = 0.05, verbose = TRUE, adapt = TRUE) {
    # df: data matrix (cells x features)
    # batch: batch labels for each cell
    # celltype: cell type to process
    # k0: (optimal) # neighbors to test on
    # sc_dir: path to save resulting rejection status and neighborhood

    library(FNN)

    # df <- as.matrix(read.csv(paste0(sc_dir, celltype, "_df.csv"), header=FALSE))
    # batch <- factor(read.csv(paste0(sc_dir, celltype, "_batch.csv"), header = FALSE)[[1]])

    dof <- length(unique(batch)) - 1    # degrees of freedom

    if (is.factor(batch)) {
        batch <- droplevels(batch)
    }
    
    frequencies <- table(batch) / length(batch)

    # get 3 different permutations of the batch label
    batch.shuff <- replicate(3, batch[sample.int(length(batch))])

    class.frequency <- data.frame(
        class = names(frequencies),
        freq = as.numeric(frequencies)
    )

    dataset <- df
    dim.dataset <- dim(dataset)

    # check dimensions
    if (dim.dataset[1] != length(batch) && dim.dataset[2] != length(batch)) {
        stop("Input matrix and batch information do not match. Execution halted.")
    }

    if (dim.dataset[2] == length(batch) && dim.dataset[1] != length(batch)) {
        if (verbose) {
            cat('Input matrix has samples as columns. kBET needs samples as rows. Transposing...\n')
        }
        dataset <- t(dataset)
        dim.dataset <- dim(dataset)
    }

    # check if the dataset is too small
    if (dim.dataset[1] <= 10) {
        if (verbose) {
            cat("Your dataset has less than 10 samples. Abort and return NA.\n")
        }
        return(NA)
    }

    # find knns
    if (is.null(knn)) {
        if (!do.pca) {
            if (verbose) {
                cat('finding knns...')
                tic <- proc.time()
            }
            # use the nearest neighbour index directly for further use in the package
            knn <- get.knn(dataset, k = k0, algorithm = 'cover_tree')$nn.index
        } else {
            dim.comp <- min(dim.pca, dim.dataset[2])

            if (verbose) {
                cat('reducing dimensions with svd first...\n')
            }
            data.pca <- svd(x = dataset, nu = dim.comp, nv = 0)
            if (verbose) {
                cat('finding knns...')
                tic <- proc.time()
            }
            knn <- get.knn(data.pca$u,  k = k0, algorithm = 'cover_tree')
        }
        if (verbose) {
            cat('done. Time:\n')
            print(proc.time() - tic)
        }
    }

    # backward compatibility for knn-graph
    if(is(knn, "list")) {
        knn <- knn$nn.index
        if (verbose) {
            cat('KNN input is a list, extracting nearest neighbour index.\n')
        }
    }

    # try to understand
    # decide to adapt general frequencies
    if (adapt) {
        outsider <- which(!(seq_len(dim.dataset[1]) %in% knn[, seq_len(k0 - 1)]))
        is.imbalanced <- FALSE #initialisation
        p.out <- 1
        
        #avoid unwanted things happen if length(outsider) == 0
        if (length(outsider) > 0) {
            p.out <- chi_batch_test(outsider, class.frequency, batch,  dof)
            if (!is.na(p.out)) {
                is.imbalanced <- p.out < alpha
                if (is.imbalanced) {
                    new.frequencies <- table(batch[-outsider])/length(batch[-outsider])
                    new.class.frequency <- data.frame(class = names(new.frequencies),
                                                    freq = as.numeric(new.frequencies))
                    if (verbose) {
                        outs_percent <- round(length(outsider) / length(batch) * 100, 3)
                        cat(paste(
                            sprintf(
                                'There are %s cells (%s%%) that do not appear in any neighbourhood.',
                                length(outsider), outs_percent
                            ),
                            'The expected frequencies for each category have been adapted.',
                            'Cell indexes are saved to result list.',
                            '', sep = '\n'
                        ))
                    }
                }
                else {
                    if (verbose) {
                        cat(paste0('No outsiders found.'))
                    }
                }
            }
            else {
                if (verbose) {
                    cat(paste0('No outsiders found.'))
                }
            }
        }
    }

    # Get neighbors for all cells
    env <- cbind(knn[, seq_len(k0 - 1)], seq_len(dim.dataset[1]))

    # perform test for each cell in sample
    cf <- if (adapt && is.imbalanced) new.class.frequency else class.frequency  # global batch props

    # apply chi_batch_test to each row of env matrix; 1 is rows
    p.val.test <- apply(env, 1, chi_batch_test, cf, batch, dof)

    is.rejected <- p.val.test < alpha

    # Combine rejection status with first k0 neighbors
    result <- data.frame(
        is.rejected = as.integer(is.rejected),
        knn[, seq_len(k0)]
    )
    
    # save
    write.csv(result, 
            file = paste0(sc_dir, celltype, "_rejection_with_neighbors.csv"),
            row.names = FALSE)

    print("csv file saved successfully")
}