##################################
## generalized-demographic-bias.R
## Generalized demographic and SDoH bias testing via Yeh (2000)
## approximate randomization.
##
## Expanded from the original single-demographic (GENDER) implementation
## to support arbitrary demographics, pairwise interaction terms, and
## a two-sided significance test so that disadvantaged-group deficits
## (negative deltas) are detectable on equal footing with surpluses.
##
## KEY METHODOLOGICAL NOTES
## -------------------------
## 1. Test type  : Yeh approximate randomization (NOT bootstrap).
##                 resampleDemographics() permutes demographic labels
##                 without replacement, preserving exact marginal counts.
## 2. Null hyp.  : Demographic label assignment is independent of model
##                 performance — i.e., any observed delta could have
##                 arisen by chance under random label permutation.
## 3. Two-sided  : RejectFlag uses |permuted| >= |observed|.  Threshold
##                 is alpha (not alpha/2) because the absolute-value
##                 transform already collapses both tails.
## 4. Cutoffs    : calculateCutoffs() returns CutOff_upper and
##                 CutOff_lower (symmetric about 0) for use in plots.
##################################

library(dplyr)
library(tidyr)
library(readr)
library(pROC)
library(mosaic)
library(progress)


## ###############################
## Helper functions
## ###############################

generateLongData <- function(df, modelColumnNames) {
  tmpDF <-
    df %>%
    pivot_longer(cols = all_of(modelColumnNames),
                 names_to  = 'Method',
                 values_to = 'Prediction')
  return(tmpDF)
}

###################
createInteractionGroups <- function(df, demographicVars) {
  interaction_col <- df %>%
    select(all_of(demographicVars)) %>%
    apply(1, function(x) paste(x, collapse = "_"))

  df$INTERACTION_GROUP <- interaction_col
  return(df)
}

###################
getDemographicSubgroups <- function(df, demographicVar) {
  df[[demographicVar]] %>%
    na.omit() %>%
    unique() %>%
    sort()
}

#################
generateAucCI <- function(df, modelName,
                          systemColName    = NULL,
                          demographicFilters = list(),
                          labelCol         = 'Label',
                          drop_ci          = TRUE) {

  if (is.null(systemColName)) {
    systemColName <- modelName
  }

  ## Apply any demographic filters
  filtered_df <- df
  if (length(demographicFilters) > 0) {
    for (demo_var in names(demographicFilters)) {
      demo_values <- demographicFilters[[demo_var]]
      filtered_df <- filtered_df %>%
        filter(.data[[demo_var]] %in% demo_values)
    }
  }

  labels_vec <- unlist(filtered_df[[labelCol]])
  scores_vec <- unlist(filtered_df[[systemColName]])

  naRow <- tibble(Model = modelName, AUC = NA_real_,
                  AUC.lb = NA_real_, AUC.ub = NA_real_)

  if (nrow(filtered_df) < 5) {
    message("generateAucCI: skipping '", modelName,
            "' — subgroup has only ", nrow(filtered_df), " rows.")
    return(if (drop_ci) naRow %>% select(-AUC.lb, -AUC.ub) else naRow)
  }

  if (length(unique(labels_vec)) < 2) {
    message("generateAucCI: skipping '", modelName,
            "' — subgroup contains only one class.")
    return(if (drop_ci) naRow %>% select(-AUC.lb, -AUC.ub) else naRow)
  }

  if (length(unique(scores_vec)) < 2) {
    message("generateAucCI: skipping '", modelName,
            "' — all predictions are identical.")
    return(if (drop_ci) naRow %>% select(-AUC.lb, -AUC.ub) else naRow)
  }

  tmpAuc <- suppressWarnings(
    auc(labels_vec, scores_vec,
        levels    = c(0, 1),
        direction = "<")
  )
  tmpReturn <-
    suppressWarnings(ci(tmpAuc)) %>%
    as_tibble() %>%
    mutate(Model   = modelName) %>%
    mutate(Measure = c('AUC.lb', 'AUC', 'AUC.ub')) %>%
    pivot_wider(names_from  = Measure,
                values_from = value) %>%
    mutate(AUC.lb = as.numeric(AUC.lb),
           AUC    = as.numeric(AUC),
           AUC.ub = as.numeric(AUC.ub))

  if (drop_ci) {
    return(tmpReturn %>% select(-AUC.lb, -AUC.ub))
  } else {
    return(tmpReturn)
  }
}

## ###########
composeDemographicCohorts <- function(df, modelName,
                                      systemColName       = NULL,
                                      demographicVar      = 'GENDER',
                                      demographicSubgroups = NULL,
                                      labelCol            = 'Label') {

  if (is.null(systemColName)) {
    systemColName <- modelName
  }

  if (is.null(demographicSubgroups)) {
    demographicSubgroups <- getDemographicSubgroups(df, demographicVar)
  }

  ## Overall (all subgroups combined)
  tmpComposite <- generateAucCI(
    df, modelName, systemColName,
    demographicFilters = list(),
    labelCol = labelCol
  ) %>%
    mutate(!!demographicVar := paste(demographicSubgroups, collapse = '+'))

  ## Per subgroup
  for (subgroup in demographicSubgroups) {
    filters <- list()
    filters[[demographicVar]] <- subgroup

    tmpComposite <- tmpComposite %>%
      full_join(
        generateAucCI(df, modelName, systemColName,
                      demographicFilters = filters,
                      labelCol = labelCol) %>%
          mutate(!!demographicVar := subgroup),
        by = c("Model", "AUC", demographicVar)
      )
  }

  return(tmpComposite)
}


## #################
generateTable2 <- function(df, modelLabels,
                           predSuffix   = '_pred',
                           labelCol     = 'Label',
                           measureNames = c('Accuracy', 'Specificity',
                                            'Precision', 'Recall', 'F1')) {

  all_cols <- character()
  for (model in modelLabels) {
    if (model %in% names(df)) {
      all_cols <- c(all_cols, model)
    }
    pred_col <- paste0(model, predSuffix)
    if (pred_col %in% names(df)) {
      all_cols <- c(all_cols, pred_col)
    }
  }
  all_cols <- unique(all_cols)

  longDF <- generateLongData(df, all_cols)

  pred_methods <- character()
  for (model in modelLabels) {
    pred_col <- paste0(model, predSuffix)
    if (pred_col %in% all_cols) {
      pred_methods <- c(pred_methods, pred_col)
    } else if (model %in% all_cols) {
      pred_methods <- c(pred_methods, model)
    }
  }

  tmpTable <-
    longDF %>%
    filter(Method %in% pred_methods) %>%
    mutate(Method = factor(Method, levels = pred_methods)) %>%
    mutate(Model = factor(
      sapply(Method, function(m) {
        gsub(paste0(predSuffix, "$"), "", m)
      }),
      levels = modelLabels
    )) %>%
    mutate(Grade = factor(case_when(
      Prediction == 1 & Prediction == .data[[labelCol]] ~ 'TP',
      Prediction == 1 & Prediction != .data[[labelCol]] ~ 'FP',
      Prediction == 0 & Prediction != .data[[labelCol]] ~ 'FN',
      Prediction == 0 & Prediction == .data[[labelCol]] ~ 'TN'),
      levels = c('TP', 'FP', 'FN', 'TN'))) %>%
    group_by(Model, Grade, .drop = FALSE) %>%
    summarise(n = n(), .groups = 'keep') %>%
    pivot_wider(names_from  = c('Grade'),
                values_from = c('n'),
                values_fill = 0)

  if ('Accuracy' %in% measureNames) {
    tmpTable <- tmpTable %>%
      mutate(Accuracy = (TP + TN) / (TP + FP + FN + TN))
  }
  if ('Specificity' %in% measureNames) {
    tmpTable <- tmpTable %>%
      mutate(Specificity = TN / (TN + FP))
  }
  if ('Precision' %in% measureNames | 'F1' %in% measureNames) {
    tmpTable <- tmpTable %>%
      mutate(Precision = TP / (TP + FP))
  }
  if ('Recall' %in% measureNames | 'F1' %in% measureNames) {
    tmpTable <- tmpTable %>%
      mutate(Recall = TP / (TP + FN))
  }
  if ('F1' %in% measureNames) {
    tmpTable <- tmpTable %>%
      mutate(F1 = 2 * Precision * Recall / (Precision + Recall))

    if (!'Precision' %in% measureNames) {
      tmpTable <- tmpTable %>% select(-Precision)
    }
    if (!'Recall' %in% measureNames) {
      tmpTable <- tmpTable %>% select(-Recall)
    }
  }

  tmpTable <- tmpTable %>% select(-TP, -FP, -FN, -TN)
  return(tmpTable)
}

#################
calculateDemographicDeltas <- function(rawDf, modelLabels,
                                       demographics       = list(GENDER = c('F', 'M')),
                                       predSuffix         = '_pred',
                                       labelCol           = 'Label',
                                       measureNames       = c('Accuracy', 'Specificity',
                                                              'Precision', 'Recall', 'F1'),
                                       includeInteractions = FALSE,
                                       min_subgroup_n     = 10) {

  if (includeInteractions && length(demographics) < 2) {
    warning("Interaction analysis requires >= 2 demographic variables. Setting includeInteractions = FALSE.")
    includeInteractions <- FALSE
  }

  all_deltas <- data.frame()

  ## Process each demographic independently
  for (demo_var in names(demographics)) {
    demo_subgroups <- demographics[[demo_var]]

    if (length(demo_subgroups) < 2) next

    ## Calculate AUC by demographic
    tmpDemoAUCs <- data.frame()
    for (model in modelLabels) {
      tmp_cohort <- composeDemographicCohorts(
        rawDf, model,
        systemColName        = model,
        demographicVar       = demo_var,
        demographicSubgroups = demo_subgroups,
        labelCol             = labelCol
      )
      if (nrow(tmpDemoAUCs) == 0) {
        tmpDemoAUCs <- tmp_cohort
      } else {
        tmpDemoAUCs <- tmpDemoAUCs %>%
          full_join(tmp_cohort, by = c("Model", "AUC", demo_var))
      }
    }

    ## Confusion metrics per subgroup
    demo_metrics <- data.frame()
    for (subgroup in demo_subgroups) {
      tmp_metrics <- generateTable2(
        rawDf %>% filter(.data[[demo_var]] == subgroup),
        modelLabels,
        predSuffix   = predSuffix,
        labelCol     = labelCol,
        measureNames = measureNames
      ) %>%
        mutate(!!demo_var := subgroup)

      if (nrow(demo_metrics) == 0) {
        demo_metrics <- tmp_metrics
      } else {
        demo_metrics <- demo_metrics %>%
          full_join(tmp_metrics,
                    by = c("Model", measureNames, demo_var))
      }
    }

    ## Pairwise deltas for all subgroup combinations
    for (i in 1:(length(demo_subgroups) - 1)) {
      for (j in (i + 1):length(demo_subgroups)) {
        subgroup_a <- demo_subgroups[i]
        subgroup_b <- demo_subgroups[j]

        ## Confusion metric deltas
        deltaDf <- demo_metrics %>%
          ungroup() %>%
          pivot_longer(cols      = all_of(measureNames),
                       names_to  = c('Metric'),
                       values_to = c('Value')) %>%
          pivot_wider(names_from  = all_of(demo_var),
                      values_from = c('Value')) %>%
          mutate(Delta         = .data[[subgroup_a]] - .data[[subgroup_b]]) %>%
          mutate(DemographicVar = demo_var) %>%
          mutate(SubgroupA      = subgroup_a) %>%
          mutate(SubgroupB      = subgroup_b) %>%
          select(Model, Metric, Delta, DemographicVar, SubgroupA, SubgroupB)

        ## AUC deltas
        auc_delta <- tmpDemoAUCs %>%
          select(Model, AUC, all_of(demo_var)) %>%
          filter(.data[[demo_var]] %in% c(subgroup_a, subgroup_b)) %>%
          mutate(Metric = 'AUC') %>%
          pivot_wider(names_from  = all_of(demo_var),
                      values_from = c('AUC')) %>%
          mutate(Delta         = .data[[subgroup_a]] - .data[[subgroup_b]]) %>%
          mutate(DemographicVar = demo_var) %>%
          mutate(SubgroupA      = subgroup_a) %>%
          mutate(SubgroupB      = subgroup_b) %>%
          select(Model, Metric, Delta, DemographicVar, SubgroupA, SubgroupB)

        deltaDf <- deltaDf %>%
          full_join(auc_delta,
                    by = c("Model", "Metric", "Delta",
                           "DemographicVar", "SubgroupA", "SubgroupB"))

        all_deltas <- bind_rows(all_deltas, deltaDf)
      }
    }
  }

  ## Interaction terms — only if requested and feasible
  if (includeInteractions && length(demographics) >= 2) {
    interaction_deltas <- calculateInteractionDeltas(
      rawDf, modelLabels, demographics,
      predSuffix     = predSuffix,
      labelCol       = labelCol,
      measureNames   = measureNames,
      min_subgroup_n = min_subgroup_n
    )
    all_deltas <- bind_rows(all_deltas, interaction_deltas)
  }

  return(all_deltas)
}

## ###############################
calculateInteractionDeltas <- function(rawDf, modelLabels,
                                       demographics,
                                       predSuffix     = '_pred',
                                       labelCol       = 'Label',
                                       measureNames   = c('Accuracy', 'Specificity',
                                                          'Precision', 'Recall', 'F1'),
                                       min_subgroup_n = 10) {

  demo_vars <- names(demographics)

  ## Create interaction column
  rawDf_interact <- createInteractionGroups(rawDf, demo_vars)

  ## Retain only interaction groups with sufficient observations.
  ## Without this filter the pairwise loop generates thousands of
  ## NA-delta rows from empty or near-empty cells, inflating output
  ## and swamping FDR correction with uninformative tests.
  group_counts <- rawDf_interact %>%
    group_by(INTERACTION_GROUP) %>%
    summarise(n = n(), .groups = 'drop') %>%
    filter(n >= min_subgroup_n)

  interaction_groups <- group_counts$INTERACTION_GROUP %>% sort()

  if (length(interaction_groups) < 2) {
    warning("calculateInteractionDeltas: fewer than 2 interaction groups ",
            "met min_subgroup_n = ", min_subgroup_n,
            ". Returning empty data frame.")
    return(data.frame())
  }

  ## Metrics per interaction group
  interact_metrics <- data.frame()
  for (group in interaction_groups) {
    tmp_metrics <- generateTable2(
      rawDf_interact %>% filter(INTERACTION_GROUP == group),
      modelLabels,
      predSuffix   = predSuffix,
      labelCol     = labelCol,
      measureNames = measureNames
    ) %>%
      mutate(InteractionGroup = group)

    if (nrow(interact_metrics) == 0) {
      interact_metrics <- tmp_metrics
    } else {
      interact_metrics <- interact_metrics %>%
        full_join(tmp_metrics,
                  by = c("Model", measureNames, "InteractionGroup"))
    }
  }

  ## AUC per interaction group
  interact_aucs <- data.frame()
  for (group in interaction_groups) {
    for (model in modelLabels) {
      tmp_auc <- generateAucCI(
        rawDf_interact %>% filter(INTERACTION_GROUP == group),
        model,
        systemColName = model,
        labelCol      = labelCol,
        drop_ci       = TRUE
      ) %>%
        mutate(InteractionGroup = group)

      if (nrow(interact_aucs) == 0) {
        interact_aucs <- tmp_auc
      } else {
        interact_aucs <- interact_aucs %>%
          full_join(tmp_auc, by = c("Model", "AUC", "InteractionGroup"))
      }
    }
  }

  ## Pairwise deltas across all qualifying interaction groups
  all_interact_deltas <- data.frame()

  for (i in 1:(length(interaction_groups) - 1)) {
    for (j in (i + 1):length(interaction_groups)) {
      group_a <- interaction_groups[i]
      group_b <- interaction_groups[j]

      ## Confusion metric deltas
      deltaDf <- interact_metrics %>%
        ungroup() %>%
        pivot_longer(cols      = all_of(measureNames),
                     names_to  = c('Metric'),
                     values_to = c('Value')) %>%
        pivot_wider(names_from  = InteractionGroup,
                    values_from = c('Value')) %>%
        mutate(Delta         = .data[[group_a]] - .data[[group_b]]) %>%
        mutate(DemographicVar = paste(demo_vars, collapse = " \u00d7 ")) %>%
        mutate(SubgroupA      = group_a) %>%
        mutate(SubgroupB      = group_b) %>%
        select(Model, Metric, Delta, DemographicVar, SubgroupA, SubgroupB)

      ## AUC deltas
      auc_delta <- interact_aucs %>%
        select(Model, AUC, InteractionGroup) %>%
        filter(InteractionGroup %in% c(group_a, group_b)) %>%
        mutate(Metric = 'AUC') %>%
        pivot_wider(names_from  = InteractionGroup,
                    values_from = c('AUC')) %>%
        mutate(Delta         = .data[[group_a]] - .data[[group_b]]) %>%
        mutate(DemographicVar = paste(demo_vars, collapse = " \u00d7 ")) %>%
        mutate(SubgroupA      = group_a) %>%
        mutate(SubgroupB      = group_b) %>%
        select(Model, Metric, Delta, DemographicVar, SubgroupA, SubgroupB)

      deltaDf <- deltaDf %>%
        full_join(auc_delta,
                  by = c("Model", "Metric", "Delta",
                         "DemographicVar", "SubgroupA", "SubgroupB"))

      all_interact_deltas <- bind_rows(all_interact_deltas, deltaDf)
    }
  }

  return(all_interact_deltas)
}


## ##################
resampleDemographics <- function(origDf, demographicVars) {
  ## Permutes each demographic column independently without replacement.
  ## This preserves the exact marginal count of every category value
  ## while breaking any association between demographics and outcomes —
  ## implementing the null hypothesis of independence.
  for (demo_var in demographicVars) {
    origDf[[demo_var]] <- sample(origDf[[demo_var]])
  }
  return(origDf)
}


###################
sampleAndScore <- function(df, modelLabels,
                           predSuffix   = '_pred',
                           labelCol     = 'Label',
                           measureNames = c('Accuracy', 'Specificity',
                                            'Precision', 'Recall', 'F1')) {

  ## AUC for all models
  auc_table <- data.frame()
  for (model in modelLabels) {
    tmp_auc <- generateAucCI(df, model,
                             systemColName = model,
                             labelCol      = labelCol,
                             drop_ci       = TRUE)
    if (nrow(auc_table) == 0) {
      auc_table <- tmp_auc
    } else {
      auc_table <- auc_table %>%
        full_join(tmp_auc, by = join_by(Model, AUC))
    }
  }

  ## Confusion metrics
  confusion_table <- generateTable2(
    df, modelLabels,
    predSuffix   = predSuffix,
    labelCol     = labelCol,
    measureNames = measureNames
  )

  tmpTable <- auc_table %>%
    full_join(confusion_table, by = c('Model'))

  return(tmpTable)
}


## #######################
sampleAndDiff <- function(df, scored_df, modelLabels,
                          measureNames = c('AUC', 'Accuracy', 'Specificity',
                                           'Precision', 'Recall', 'F1')) {

  longDF <-
    scored_df %>%
    pivot_longer(cols      = all_of(measureNames),
                 names_to  = c('Measure'),
                 values_to = c('Value'))

  sampledDeltas <- data.frame(
    Measure = character(),
    ModA    = character(),
    ModB    = character(),
    Delta   = numeric()
  )

  for (iA in seq(1, length(modelLabels) - 1)) {
    for (iB in seq(iA + 1, length(modelLabels))) {
      for (meas in measureNames) {
        tmpDelta <- longDF %>%
          filter(Model == modelLabels[iA]) %>%
          filter(Measure == meas) %>%
          rename(ModA = Model, ValA = Value) %>%
          full_join(longDF %>%
                      filter(Model == modelLabels[iB]) %>%
                      filter(Measure == meas) %>%
                      rename(ModB = Model, ValB = Value),
                    by = c('Measure')) %>%
          mutate(Delta = ValA - ValB) %>%
          select(Measure, ModA, ModB, Delta)

        sampledDeltas <- bind_rows(sampledDeltas, tmpDelta)
      }
    }
  }

  return(sampledDeltas)
}


## ###############################
calculateCutoffs <- function(deltaDf, modelLabels,
                             alpha        = 0.05,
                             measureNames = c('AUC', 'Accuracy', 'Specificity',
                                              'Precision', 'Recall', 'F1')) {
  ## Returns symmetric two-tailed cutoffs (CutOff_upper, CutOff_lower)
  ## derived from the permutation null distribution.
  ##
  ## The professor's original used a single right-tail cutoff
  ## (chopSize = n * alpha/2, sort descending, take head).
  ## That was valid for a directional Female > Male test.
  ## For arbitrary SubgroupA vs SubgroupB pairs we need both tails:
  ## CutOff_upper = quantile(1 - alpha/2)   right boundary
  ## CutOff_lower = quantile(    alpha/2)   left  boundary
  ## Any observed delta outside [lower, upper] is significant.

  has_demographics <- all(c("DemographicVar", "SubgroupA", "SubgroupB") %in%
                            names(deltaDf))

  if (!has_demographics) {
    ## Simple case: Model x Metric only
    tmpCutoffs <- deltaDf %>%
      filter(Model  %in% modelLabels,
             Metric %in% measureNames) %>%
      group_by(Model, Metric) %>%
      summarise(
        CutOff_upper = quantile(Delta, 1 - alpha / 2, na.rm = TRUE),
        CutOff_lower = quantile(Delta,     alpha / 2, na.rm = TRUE),
        .groups = 'drop'
      )

  } else {
    ## Complex case: includes demographics
    tmpCutoffs <- deltaDf %>%
      filter(Model  %in% modelLabels,
             Metric %in% measureNames) %>%
      group_by(Model, Metric, DemographicVar, SubgroupA, SubgroupB) %>%
      summarise(
        CutOff_upper = quantile(Delta, 1 - alpha / 2, na.rm = TRUE),
        CutOff_lower = quantile(Delta,     alpha / 2, na.rm = TRUE),
        .groups = 'drop'
      )
  }

  return(tmpCutoffs)
}


## ###############################
permutationTestDemographicBias <- function(df, modelLabels,
                                           demographics        = list(GENDER = c('F', 'M')),
                                           n_permutations      = 1000,
                                           predSuffix          = '_pred',
                                           labelCol            = 'Label',
                                           measureNames        = c('Accuracy', 'Specificity',
                                                                    'Precision', 'Recall', 'F1'),
                                           alpha               = 0.05,
                                           includeInteractions = FALSE,
                                           min_subgroup_n      = 10,
                                           show_progress       = TRUE) {

  if (includeInteractions && length(demographics) < 2) {
    warning("Interaction analysis requires >= 2 demographic variables. Setting includeInteractions = FALSE.")
    includeInteractions <- FALSE
  }

  ## Observed deltas on real data
  observedDelta <- calculateDemographicDeltas(
    df, modelLabels, demographics,
    predSuffix          = predSuffix,
    labelCol            = labelCol,
    measureNames        = measureNames,
    includeInteractions = includeInteractions,
    min_subgroup_n      = min_subgroup_n
  )

  ## Permutation loop
  if (show_progress) {
    pb <- progress_bar$new(
      format = "  Permuting [:bar] :percent eta: :eta",
      total  = n_permutations, clear = FALSE, width = 60)
  }

  permutedDeltas   <- data.frame()
  demographicVars  <- names(demographics)

  for (i in 1:n_permutations) {
    if (show_progress) pb$tick()

    tmpPermuted <- resampleDemographics(df, demographicVars)
    tmpDelta    <- calculateDemographicDeltas(
      tmpPermuted, modelLabels, demographics,
      predSuffix          = predSuffix,
      labelCol            = labelCol,
      measureNames        = measureNames,
      includeInteractions = includeInteractions,
      min_subgroup_n      = min_subgroup_n
    )
    permutedDeltas <- bind_rows(permutedDeltas, tmpDelta)
  }

  ## Symmetric two-tailed cutoffs for plots
  cutoffs <- calculateCutoffs(
    permutedDeltas, modelLabels,
    alpha        = alpha,
    measureNames = c('AUC', measureNames)
  )

  ## -----------------------------------------------------------------
  ## P-value computation — TWO-SIDED
  ## -----------------------------------------------------------------
  ## We count how often |permuted delta| >= |observed delta|.
  ## Laplace smoothing (+1 numerator, +1 denominator) prevents p = 0.
  ## Significance threshold is alpha (not alpha/2) because the
  ## absolute-value transform already collapses both tails into one.
  ##
  ## Why the change from the original one-sided test:
  ##   The original code tested only PermutedDelta > ObservedDelta,
  ##   which was intentional when Delta was always defined as F - M.
  ##   For arbitrary (SubgroupA, SubgroupB) pairs the sign of the
  ##   observed delta depends on alphabetical ordering, so a one-sided
  ##   test will systematically miss deficits in the alphabetically
  ##   earlier subgroup.  The absolute-value two-sided test treats
  ##   surpluses and deficits symmetrically.
  ## -----------------------------------------------------------------

  if (all(c("DemographicVar", "SubgroupA", "SubgroupB") %in% names(observedDelta))) {

    results <- observedDelta %>%
      left_join(
        permutedDeltas %>%
          rename(PermutedDelta = Delta) %>%
          left_join(
            observedDelta %>%
              rename(ObservedDelta = Delta) %>%
              select(Model, Metric, DemographicVar,
                     SubgroupA, SubgroupB, ObservedDelta),
            by = c("Model", "Metric", "DemographicVar",
                   "SubgroupA", "SubgroupB")
          ) %>%
          ## Two-sided: reject when |permuted| >= |observed|
          mutate(RejectFlag = as.integer(
            abs(PermutedDelta) >= abs(ObservedDelta)
          )) %>%
          group_by(Model, Metric, DemographicVar, SubgroupA, SubgroupB) %>%
          summarise(RejectCounts = sum(RejectFlag, na.rm = TRUE),
                    .groups = 'drop') %>%
          mutate(DistribProb = (RejectCounts + 1) / (n_permutations + 1)),
        by = c("Model", "Metric", "DemographicVar", "SubgroupA", "SubgroupB")
      ) %>%
      ## alpha (not alpha/2) because abs() already collapsed both tails
      mutate(Significant = DistribProb < alpha) %>%
      mutate(Direction = case_when(
        Delta > 0 ~ paste0(SubgroupA, " > ", SubgroupB),
        Delta < 0 ~ paste0(SubgroupB, " > ", SubgroupA),
        TRUE      ~ "No difference"
      ))

  } else {

    results <- observedDelta %>%
      left_join(
        permutedDeltas %>%
          rename(PermutedDelta = Delta) %>%
          left_join(
            observedDelta %>%
              rename(ObservedDelta = Delta) %>%
              select(Model, Metric, ObservedDelta),
            by = c("Model", "Metric")
          ) %>%
          mutate(RejectFlag = as.integer(
            abs(PermutedDelta) >= abs(ObservedDelta)
          )) %>%
          group_by(Model, Metric) %>%
          summarise(RejectCounts = sum(RejectFlag, na.rm = TRUE),
                    .groups = 'drop') %>%
          mutate(DistribProb = (RejectCounts + 1) / (n_permutations + 1)),
        by = c("Model", "Metric")
      ) %>%
      mutate(Significant = DistribProb < alpha)
  }

  return(list(
    observed       = observedDelta,
    cutoffs        = cutoffs,
    results        = results,
    permuted_deltas = permutedDeltas
  ))
}