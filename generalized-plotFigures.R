####################

library("dplyr")
library("tidyr")
library("ggplot2")
library("RColorBrewer")
library("ggtext")
library("forcats")

################
MAX_PANELS_PER_PAGE <- 16   #


################
shortenLabel <- function(x) {
  lookup <- c(
    "BLACK OR AFRICAN AMERICAN"  = "Black/AA",
    "WHITE OR CAUCASIAN"         = "White",
    "HISPANIC OR LATINO"         = "Hispanic",
    "NOT HISPANIC OR LATINO"     = "Non-Hispanic",
    "AMERICAN INDIAN"            = "Am. Indian",
    "ASIAN"                      = "Asian",
    "NATIVE HAWAIIAN"            = "Nat. Hawaiian",
    "OTHER"                      = "Other",
    "UNKNOWN"                    = "Unknown",
    "F"                          = "F",
    "M"                          = "M"
  )
  out <- ifelse(x %in% names(lookup), lookup[x], x)

  ## Handle combined "A+B" strings from build_splits_df
  out <- sapply(out, function(s) {
    if (grepl("\\+", s)) {
      parts <- strsplit(s, "\\+")[[1]]
      short <- ifelse(parts %in% names(lookup), lookup[parts], parts)
      paste(short, collapse = "+")
    } else {
      s
    }
  })

  ## Shorten interaction group labels like "F_BLACK OR AFRICAN AMERICAN"
  out <- gsub("BLACK OR AFRICAN AMERICAN", "Black/AA", out)
  out <- gsub("WHITE OR CAUCASIAN",        "White",    out)
  out <- gsub("HISPANIC OR LATINO",        "Hispanic", out)
  out <- gsub("NOT HISPANIC OR LATINO",    "Non-Hisp", out)
  out <- gsub("NATIVE HAWAIIAN",           "Nat.Hawaiian", out)
  out <- gsub("AMERICAN INDIAN",           "Am.Indian", out)

  unname(out)
}

#' Apply shortenLabel to a ggplot scale_*_discrete replacement.
shortScale <- function(...) {
  ggplot2::scale_color_discrete(labels = shortenLabel, ...)
}

# Natural metric ordering used in heatmap rows
METRIC_ORDER <- c("AUC", "Accuracy", "Specificity", "Precision", "Recall", "F1")

######################
savePaginated <- function(plots, baseFile, nPerPage = MAX_PANELS_PER_PAGE,
                          width = 14, height = 10) {
  ext      <- sub(".*\\.", ".", baseFile)
  stem     <- sub("\\.[^.]+$", "", baseFile)
  chunks   <- split(seq_along(plots),
                    ceiling(seq_along(plots) / nPerPage))
  nPages   <- length(chunks)

  for (pg in seq_along(chunks)) {
    idx     <- chunks[[pg]]
    sublist <- plots[idx]

    combined <- tryCatch({
      requireNamespace("patchwork", quietly = TRUE)
      p <- Reduce(`+`, sublist)
      p + patchwork::plot_layout(ncol = min(4, ceiling(sqrt(length(sublist)))))
    }, error = function(e) {
      tryCatch({
        requireNamespace("cowplot", quietly = TRUE)
        cowplot::plot_grid(plotlist = sublist,
                           ncol = min(4, ceiling(sqrt(length(sublist)))))
      }, error = function(e2) {
        gridExtra::arrangeGrob(grobs = sublist,
                               ncol = min(4, ceiling(sqrt(length(sublist)))))
      })
    })

    suffix <- if (nPages == 1) "" else paste0("_p", pg)
    outFile <- paste0(stem, suffix, ext)
    ggsave(outFile, combined, width = width, height = height, units = "in",
           limitsize = FALSE)
    cat("    saved:", outFile, "\n")
  }
}

#############################
plotDemographicBias <- function(biasResults,
                                figureFile          = NA,
                                metricNames         = c("AUC", "Accuracy",
                                                        "Specificity", "Precision",
                                                        "Recall", "F1"),
                                modelLabels         = NULL,
                                showOnlySignificant = FALSE,
                                flipAxes            = NULL,
                                width               = 10,
                                height              = 6) {

  tmpResults <- biasResults$results

  if (showOnlySignificant) {
    tmpResults <- tmpResults %>% filter(Significant == TRUE)
    if (nrow(tmpResults) == 0) {
      message("plotDemographicBias: no significant results to display.")
      return(invisible(NULL))
    }
  }

  if (!is.null(modelLabels)) {
    tmpResults <- tmpResults %>%
      filter(Model %in% modelLabels) %>%
      mutate(Model = factor(Model, levels = modelLabels))
  }

  tmpResults <- tmpResults %>%
    filter(Metric %in% metricNames) %>%
    mutate(Metric = factor(Metric,
                           levels = intersect(METRIC_ORDER, metricNames)))

  nModels <- n_distinct(tmpResults$Model)

  ## Shorten comparison labels
  tmpResults <- tmpResults %>%
    mutate(
      sgA_short = shortenLabel(SubgroupA),
      sgB_short = shortenLabel(SubgroupB),
      Comparison = paste(sgA_short, "vs", sgB_short)
    )

  nNA <- sum(is.na(tmpResults$Delta))
  if (nNA > 0) {
    message("plotDemographicBias: dropping ", nNA, " NA-Delta row(s).")
    tmpResults <- tmpResults %>% filter(!is.na(Delta))
  }
  if (nrow(tmpResults) == 0) {
    message("plotDemographicBias: no data left after NA removal.")
    return(invisible(NULL))
  }

  demoLabel <- paste(unique(tmpResults$DemographicVar), collapse = " / ")

  if (showOnlySignificant) {
    tmpPlot <-
      ggplot(tmpResults,
             aes(x = Delta, y = Model,
                 color = Comparison)) +
      geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
      geom_segment(aes(x = 0, xend = Delta, yend = Model),
                   linewidth = 0.8, na.rm = TRUE) +
      geom_point(size = 4, na.rm = TRUE) +
      facet_wrap(~Metric, scales = "free_x") +
      scale_color_brewer(palette = "Set1", name = "Comparison") +
      labs(title    = paste("Significant Bias —", demoLabel),
           subtitle = paste0("alpha = 0.05  |  n = ", nrow(tmpResults),
                             " significant finding(s)"),
           x = "\u0394 Performance  (SubgroupA \u2212 SubgroupB)",
           y = "Model") +
      theme_bw(base_size = 12) +
      theme(legend.position  = "bottom",
            legend.key.width = unit(0.6, "cm"),
            plot.title       = element_text(face = "bold"),
            strip.background = element_rect(fill = "grey92"))

    print(tmpPlot)
    if (!is.na(figureFile))
      ggsave(figureFile, width = width, height = max(4, nrow(tmpResults) * 0.5 + 2),
             units = "in", limitsize = FALSE)
    return(invisible(tmpPlot))
  }

  if (is.null(flipAxes)) flipAxes <- (nModels == 1)

  hasPermuted <- !is.null(biasResults$permuted_deltas) &&
    nrow(biasResults$permuted_deltas) > 0

  if (hasPermuted) {
    byVars <- intersect(c("Model", "Metric", "DemographicVar",
                          "SubgroupA", "SubgroupB"),
                        names(biasResults$permuted_deltas))
    permSE <- biasResults$permuted_deltas %>%
      filter(Metric %in% metricNames) %>%
      group_by(across(all_of(byVars))) %>%
      summarise(PermSE = sd(Delta, na.rm = TRUE), .groups = "drop")

    ## Join: match on original (un-shortened) subgroup names
    tmpResults <- tmpResults %>%
      left_join(permSE, by = byVars) %>%
      mutate(
        CI_lo = Delta - 1.96 * PermSE,
        CI_hi = Delta + 1.96 * PermSE
      )
  } else {
    tmpResults <- tmpResults %>%
      mutate(PermSE = NA_real_, CI_lo = NA_real_, CI_hi = NA_real_)
  }

  ## Number of comparison groups — choose color palette
  nComp   <- n_distinct(tmpResults$Comparison)
  pal     <- if (nComp <= 8) brewer.pal(max(3, nComp), "Set1")[seq_len(nComp)] else
    colorRampPalette(brewer.pal(8, "Set1"))(nComp)
  names(pal) <- unique(tmpResults$Comparison)

  if (flipAxes) {
    ## Single model: metrics on x, delta on y, error bars vertical
    tmpPlot <-
      ggplot(tmpResults,
             aes(x = Metric, y = Delta,
                 color = Comparison, group = Comparison)) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
      geom_errorbar(aes(ymin = CI_lo, ymax = CI_hi),
                    width = 0.15, linewidth = 0.6,
                    position = position_dodge(0.4), na.rm = TRUE) +
      geom_point(size = 3.5, na.rm = TRUE,
                 position = position_dodge(0.4)) +
      scale_color_manual(values = pal, name = "Comparison") +
      labs(title    = paste("Performance Bias by", demoLabel),
           subtitle = "Error bars = observed \u00b1 1.96\u00d7SE(permutation)",
           x = "Metric",
           y = "\u0394 Performance  (SubgroupA \u2212 SubgroupB)") +
      theme_bw(base_size = 12) +
      theme(axis.text.x     = element_text(angle = 30, hjust = 1, size = 9),
            legend.position  = "top",
            legend.direction  = "horizontal",
            strip.background = element_rect(fill = "grey92"),
            plot.title       = element_text(face = "bold"),
            plot.subtitle    = element_text(size = 9, color = "grey40"))

  } else {
    ## Multiple models: models on x, metrics faceted, comparisons as color
    tmpPlot <-
      ggplot(tmpResults,
             aes(x = Model, y = Delta,
                 color = Comparison, group = Comparison)) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
      geom_errorbar(aes(ymin = CI_lo, ymax = CI_hi),
                    width = 0.15, linewidth = 0.6,
                    position = position_dodge(0.5), na.rm = TRUE) +
      geom_point(size = 3, na.rm = TRUE,
                 position = position_dodge(width = 0.5)) +
      facet_wrap(~Metric, scales = "free_y", nrow = 2) +
      scale_color_manual(values = pal, name = "Comparison") +
      labs(title    = paste("Performance Bias by", demoLabel),
           subtitle = "Error bars = observed \u00b1 1.96\u00d7SE(permutation)",
           x = "Model",
           y = "\u0394 Performance  (SubgroupA \u2212 SubgroupB)") +
      theme_bw(base_size = 12) +
      theme(axis.text.x     = element_text(angle = 40, hjust = 1, size = 9),
            legend.position  = "top",
            legend.direction  = "horizontal",
            legend.key.width = unit(0.5, "cm"),
            strip.background = element_rect(fill = "grey92"),
            plot.title       = element_text(face = "bold"),
            plot.subtitle    = element_text(size = 9, color = "grey40"))
  }

  print(tmpPlot)
  if (!is.na(figureFile))
    ggsave(figureFile, width = width, height = height, units = "in",
           limitsize = FALSE)
  invisible(tmpPlot)
}

##########################
plotBiasHeatmap <- function(biasResults,
                            figureFile     = NA,
                            metricNames    = c("AUC", "Accuracy",
                                               "Specificity",
                                               "Precision",
                                               "Recall", "F1"),
                            modelLabels    = NULL,
                            demographicVar = NULL,
                            width          = 9,
                            height         = 6) {

  tmpResults <- biasResults$results

  if (!is.null(modelLabels)) {
    tmpResults <- tmpResults %>%
      filter(Model %in% modelLabels) %>%
      mutate(Model = factor(Model, levels = modelLabels))
  }
  if (!is.null(demographicVar)) {
    tmpResults <- tmpResults %>% filter(DemographicVar == demographicVar)
  }

  tmpResults <- tmpResults %>%
    filter(Metric %in% metricNames) %>%
    mutate(
      Metric = factor(Metric,
                      levels = rev(intersect(METRIC_ORDER, metricNames))),
      sgA_short = shortenLabel(SubgroupA),
      sgB_short = shortenLabel(SubgroupB),
      Comparison = paste(sgA_short, "vs", sgB_short)
    )

  nComp      <- n_distinct(tmpResults$Comparison)
  demoLabel  <- paste(unique(tmpResults$DemographicVar), collapse = " / ")
  limMax     <- max(abs(tmpResults$Delta), na.rm = TRUE)

  tmpPlot <-
    ggplot(tmpResults,
           aes(x = Model, y = Metric, fill = Delta)) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_text(aes(label = ifelse(Significant, "\u2605", "")),
              size = 5, color = "black") +
    scale_fill_gradient2(low      = "#3B5998",
                         mid      = "white",
                         high     = "#CC0000",
                         midpoint = 0,
                         limits   = c(-limMax, limMax),
                         name     = "\u0394") +
    labs(title    = paste("Bias Heatmap —", demoLabel),
         subtitle = "\u2605 = significant at \u03b1 = 0.05",
         x = "Model", y = NULL) +
    theme_bw(base_size = 12) +
    theme(axis.text.x     = element_text(angle = 40, hjust = 1, size = 9),
          strip.background = element_rect(fill = "grey92"),
          plot.title       = element_text(face = "bold"),
          legend.key.height = unit(1.2, "cm"))

  if (nComp > 1) {
    tmpPlot <- tmpPlot + facet_wrap(~Comparison)
    width   <- width + (nComp - 1) * 3
  }

  print(tmpPlot)
  if (!is.na(figureFile))
    ggsave(figureFile, width = width, height = height, units = "in",
           limitsize = FALSE)
  invisible(tmpPlot)
}

#########################
plotPermutationDistributions <- function(biasResults,
                                          figureFile     = NA,
                                          metricNames    = c("AUC", "F1"),
                                          modelLabels    = NULL,
                                          demographicVar = NULL,
                                          subgroupA      = NULL,
                                          subgroupB      = NULL,
                                          useFacetGrid   = TRUE,
                                          width          = 12,
                                          height         = 8) {

  tmpPermuted <- biasResults$permuted_deltas
  tmpObserved <- biasResults$observed
  tmpCutoffs  <- biasResults$cutoffs

  for (obj in c("tmpPermuted", "tmpObserved", "tmpCutoffs")) {
    d <- get(obj)
    if (!is.null(modelLabels))    d <- d %>% filter(Model  %in% modelLabels)
    if (!is.null(metricNames))    d <- d %>% filter(Metric %in% metricNames)
    if (!is.null(demographicVar)) d <- d %>% filter(DemographicVar == demographicVar)
    if (!is.null(subgroupA))      d <- d %>% filter(SubgroupA == subgroupA)
    if (!is.null(subgroupB))      d <- d %>% filter(SubgroupB == subgroupB)
    assign(obj, d)
  }

  ## Shorten labels
  for (obj in c("tmpPermuted", "tmpObserved", "tmpCutoffs")) {
    d <- get(obj)
    if (all(c("SubgroupA","SubgroupB") %in% names(d)))
      d <- d %>%
        mutate(SubgroupA = shortenLabel(SubgroupA),
               SubgroupB = shortenLabel(SubgroupB))
    assign(obj, d)
  }

  addComp <- function(d) d %>%
    mutate(Comparison = paste(SubgroupA, "vs", SubgroupB))

  tmpPermuted <- addComp(tmpPermuted)
  tmpObserved <- addComp(tmpObserved)
  tmpCutoffs  <- addComp(tmpCutoffs)

  nComparisons <- n_distinct(tmpPermuted$Comparison)
  nModels      <- n_distinct(tmpPermuted$Model)
  nMetrics     <- n_distinct(tmpPermuted$Metric)

  ## x-axis label
  if (nComparisons == 1) {
    sg_a    <- unique(tmpPermuted$SubgroupA)[1]
    sg_b    <- unique(tmpPermuted$SubgroupB)[1]
    x_label <- paste0("\u0394 Performance  (", sg_a, " \u2212 ", sg_b, ")")
  } else {
    x_label <- "\u0394 Performance  (SubgroupA \u2212 SubgroupB)"
  }

  ## Build base plot constructor for a subset of the data
  makePlot <- function(perm_sub, obs_sub, cut_sub, stripSz = 7) {
    ggplot(perm_sub, aes(x = Delta)) +
      geom_histogram(fill = "#AEC6CF", color = "#1B4F72",
                     alpha = 0.8, bins = 30) +
      geom_vline(data = obs_sub, aes(xintercept = Delta),
                 color = "black", linewidth = 1) +
      geom_vline(data = cut_sub, aes(xintercept = CutOff_upper),
                 color = "#CC0000", linetype = "dashed", linewidth = 0.9) +
      geom_vline(data = cut_sub, aes(xintercept = CutOff_lower),
                 color = "#CC0000", linetype = "dashed", linewidth = 0.9) +
      xlab(x_label) + ylab(NULL) +
      scale_y_continuous(breaks = NULL) +
      theme_bw(base_size = 11) +
      theme(panel.border    = element_blank(),
            axis.line.x     = element_line(color = "black"),
            strip.background = element_rect(fill = "grey92"),
            strip.text       = element_text(size = stripSz))
  }

  ## ── single-comparison: grid layout ───────────────────────────────────────
  if (useFacetGrid && nComparisons == 1) {
    nPanels <- nModels * nMetrics
    stripSz <- if (nPanels > 20) 6 else 7

    p <- makePlot(tmpPermuted, tmpObserved, tmpCutoffs, stripSz) +
      facet_grid(Model ~ Metric, scales = "free")
    print(p)
    if (!is.na(figureFile))
      ggsave(figureFile, width = width, height = max(6, nModels * 1.2),
             units = "in", limitsize = FALSE)
    return(invisible(p))
  }

  ## ── multi-comparison: paginate ───────────────────────────────────────────
  ## Split by Comparison × Model so each page stays readable
  combos <- tmpPermuted %>%
    select(Comparison, Model) %>%
    distinct() %>%
    arrange(Comparison, Model)

  nPanelsPerCombo <- nMetrics  ## one row per metric per combo
  totalPanels     <- nrow(combos) * nPanelsPerCombo

  if (totalPanels <= MAX_PANELS_PER_PAGE || is.na(figureFile)) {
    ## Fits on one page
    stripSz <- if (totalPanels > 24) 5 else 7
    p <- makePlot(tmpPermuted, tmpObserved, tmpCutoffs, stripSz) +
      facet_wrap(~Comparison + Model + Metric, scales = "free",
                 ncol = nMetrics * min(4, nComparisons))
    print(p)
    if (!is.na(figureFile))
      ggsave(figureFile, width = width, height = max(6, nrow(combos) * 1.4),
             units = "in", limitsize = FALSE)
    return(invisible(p))
  }

  ## Build one mini-plot per (Comparison × Model)  then paginate
  plotList <- lapply(seq_len(nrow(combos)), function(i) {
    cmp <- combos$Comparison[i]
    mdl <- combos$Model[i]
    p_s <- tmpPermuted %>% filter(Comparison == cmp, Model == mdl)
    o_s <- tmpObserved  %>% filter(Comparison == cmp, Model == mdl)
    c_s <- tmpCutoffs   %>% filter(Comparison == cmp, Model == mdl)
    makePlot(p_s, o_s, c_s, stripSz = 7) +
      facet_wrap(~Metric, nrow = 1, scales = "free") +
      labs(title = paste0(mdl, "  |  ", cmp)) +
      theme(plot.title = element_text(size = 9, face = "bold"))
  })

  savePaginated(plotList, figureFile,
                nPerPage = floor(MAX_PANELS_PER_PAGE / nMetrics),
                width = width, height = height)
  invisible(plotList)
}


#####################
## plotCutoffDistribution  (single-comparison diagnostic; unchanged logic)
#####################
plotCutoffDistribution <- function(biasResults,
                                    figureFile     = NA,
                                    modelName,
                                    metricName,
                                    demographicVar = NULL,
                                    subgroupA      = NULL,
                                    subgroupB      = NULL,
                                    width          = 6,
                                    height         = 4) {

  tmpPermuted <- biasResults$permuted_deltas %>%
    filter(Model == modelName, Metric == metricName)
  tmpObserved <- biasResults$observed %>%
    filter(Model == modelName, Metric == metricName)
  tmpCutoffs  <- biasResults$cutoffs %>%
    filter(Model == modelName, Metric == metricName)

  for (obj in c("tmpPermuted", "tmpObserved", "tmpCutoffs")) {
    d <- get(obj)
    if (!is.null(demographicVar)) d <- d %>% filter(DemographicVar == demographicVar)
    if (!is.null(subgroupA))      d <- d %>% filter(SubgroupA == subgroupA)
    if (!is.null(subgroupB))      d <- d %>% filter(SubgroupB == subgroupB)
    assign(obj, d)
  }

  title_parts <- c(modelName, metricName)
  if (!is.null(demographicVar)) title_parts <- c(title_parts, demographicVar)
  if (!is.null(subgroupA) && !is.null(subgroupB))
    title_parts <- c(title_parts,
                     paste(shortenLabel(subgroupA), "vs", shortenLabel(subgroupB)))
  plot_title <- paste(title_parts, collapse = " \u2014 ")

  tmpPlot <-
    ggplot(tmpPermuted, aes(x = Delta)) +
    geom_histogram(fill = "#AEC6CF", color = "#1B4F72", alpha = 0.8, bins = 30) +
    geom_vline(data = tmpObserved, aes(xintercept = Delta),
               color = "black", linewidth = 1.2) +
    geom_vline(data = tmpCutoffs, aes(xintercept = CutOff_upper),
               color = "#CC0000", linewidth = 1, linetype = "dashed") +
    geom_vline(data = tmpCutoffs, aes(xintercept = CutOff_lower),
               color = "#CC0000", linewidth = 1, linetype = "dashed") +
    labs(x     = "\u0394 Performance (SubgroupA \u2212 SubgroupB)",
         y     = "Frequency",
         title = plot_title) +
    theme_bw(base_size = 12) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))

  print(tmpPlot)
  if (!is.na(figureFile))
    ggsave(figureFile, width = width, height = height, units = "in")
  invisible(tmpPlot)
}



#####################
plotDemographicSplits <- function(splitsDF,
                                   figureFile        = NA,
                                   metricNames       = c("F1"),
                                   demographicColumn = "Gender",
                                   modelSubset       = NULL,
                                   connectLines      = FALSE,
                                   useOffset         = TRUE,
                                   coordFlip         = TRUE,
                                   colorPalette      = "Dark2",
                                   width             = 7,
                                   height            = 4) {

  tmpData <- splitsDF
  if (!is.null(modelSubset))
    tmpData <- tmpData %>% filter(Model %in% modelSubset)

  ## Shorten demographic labels
  if (demographicColumn %in% names(tmpData)) {
    tmpData[[demographicColumn]] <- sapply(
      tmpData[[demographicColumn]],
      function(s) if (grepl("\\+", s)) "Overall" else shortenLabel(s)
    )
  }

  tmpDataLong <- tmpData %>%
    pivot_longer(cols      = all_of(metricNames),
                 names_to  = "Metric",
                 values_to = "Value") %>%
    mutate(Metric = factor(Metric, levels = metricNames))

  nNA <- sum(is.na(tmpDataLong$Value))
  if (nNA > 0) {
    message("plotDemographicSplits: dropping ", nNA, " NA-Value row(s).")
    tmpDataLong <- tmpDataLong %>% filter(!is.na(Value))
  }

  if (useOffset) {
    demo_groups <- tmpDataLong %>%
      filter(.data[[demographicColumn]] != "Overall") %>%
      pull(.data[[demographicColumn]]) %>%
      unique()

    tmpDataLong <- tmpDataLong %>%
      mutate(DemoOffset = case_when(
        .data[[demographicColumn]] == "Overall" ~ 0,
        TRUE ~ match(.data[[demographicColumn]], demo_groups) * 0.18 - 0.18
      ))

    tmpPlot <-
      ggplot(tmpDataLong,
             aes(x = as.numeric(factor(Model)) + DemoOffset,
                 y = Value))
  } else {
    tmpPlot <-
      ggplot(tmpDataLong, aes(x = Model, y = Value))
  }

  n_models_in_data <- n_distinct(tmpDataLong$Model)
  nGroups <- n_distinct(tmpDataLong[[demographicColumn]])
  nColors <- max(3, min(nGroups, 8))

  if (connectLines && n_models_in_data >= 2) {
    tmpPlot <- tmpPlot +
      geom_line(aes(group = .data[[demographicColumn]],
                    color = .data[[demographicColumn]]),
                na.rm = TRUE, linewidth = 0.7)
  }

  tmpPlot <- tmpPlot +
    geom_point(aes(group = .data[[demographicColumn]],
                   color = .data[[demographicColumn]]),
               size = 3.5, stroke = 2.5, shape = 21, fill = "white",
               na.rm = TRUE) +
    scale_color_manual('Group',
                       values = brewer.pal(nColors, colorPalette)) +
    labs(x = "Model", y = "Performance") +
    theme_bw(base_size = 12) +
    ## HP1 (reviewer): legend moved top — prevents wide race/ethnicity labels
    ## from stealing horizontal space that should belong to the performance axis.
    theme(legend.position  = "top",
          legend.direction  = "horizontal",
          legend.key.width  = unit(0.5, "cm"),
          legend.text       = element_text(size = 9),
          plot.margin       = margin(t = 2, r = 4, l = 2, b = 2))

  if (length(metricNames) > 1) {
    tmpPlot <- tmpPlot + facet_wrap(~Metric, scales = "free_x") +
      theme(strip.background = element_rect(fill = "grey92"))
  } else {
    tmpPlot <- tmpPlot + labs(y = metricNames[1])
  }

  if (coordFlip) tmpPlot <- tmpPlot + coord_flip()

  if (useOffset) {
    modelNames <- levels(factor(tmpDataLong$Model))
    nModels    <- length(modelNames)

    tmpPlot <- tmpPlot +
      scale_x_continuous(
        minor_breaks = if (nModels > 1) seq(1.5, nModels - 0.5) else numeric(0),
        breaks       = seq_len(nModels),
        labels       = modelNames
      )

    gridLines <- if (coordFlip) {
      theme(panel.grid.major.x = element_blank(),
            panel.grid.minor.x = element_line(color = "grey70", linewidth = 0.4,
                                               linetype = 2))
    } else {
      theme(panel.grid.major.y = element_blank(),
            panel.grid.minor.y = element_line(color = "grey70", linewidth = 0.4,
                                               linetype = 2))
    }
    tmpPlot <- tmpPlot + gridLines
  }

  print(tmpPlot)
  if (!is.na(figureFile))
    ggsave(figureFile, width = width, height = height, units = "in",
           limitsize = FALSE)
  invisible(tmpPlot)
}


#####################
plotInteractionBias <- function(biasResults,
                                 figureFile    = NA,
                                 metricNames   = c("AUC", "F1"),
                                 modelLabels   = NULL,
                                 interactionVar = NULL,
                                 width          = 12,
                                 height         = 7) {

  tmpResults <- biasResults$results %>%
    filter(grepl("\u00d7", DemographicVar))

  if (nrow(tmpResults) == 0)
    stop("No interaction terms found. Set includeInteractions = TRUE.")

  if (!is.null(interactionVar))
    tmpResults <- tmpResults %>% filter(DemographicVar == interactionVar)

  if (!is.null(modelLabels)) {
    tmpResults <- tmpResults %>%
      filter(Model %in% modelLabels) %>%
      mutate(Model = factor(Model, levels = modelLabels))
  }

  tmpResults <- tmpResults %>%
    filter(Metric %in% metricNames) %>%
    mutate(Metric = factor(Metric,
                           levels = intersect(METRIC_ORDER, metricNames)))

  ## Shorten subgroup labels
  tmpResults <- tmpResults %>%
    mutate(
      SubgroupA_s = shortenLabel(SubgroupA),
      SubgroupB_s = shortenLabel(SubgroupB),
      Comparison  = paste(SubgroupA_s, "vs", SubgroupB_s)
    )

  nModels <- n_distinct(tmpResults$Model)
  modelColors <- setNames(
    brewer.pal(max(3, min(nModels, 8)), "Set1")[seq_len(nModels)],
    levels(tmpResults$Model)
  )

  hasPermuted <- !is.null(biasResults$permuted_deltas) &&
    nrow(biasResults$permuted_deltas) > 0

  if (hasPermuted) {
    byVarsInt <- intersect(c("Model", "Metric", "DemographicVar",
                             "SubgroupA", "SubgroupB"),
                           names(biasResults$permuted_deltas))
    permSE_int <- biasResults$permuted_deltas %>%
      filter(grepl("\u00d7", DemographicVar),
             Metric %in% metricNames) %>%
      group_by(across(all_of(byVarsInt))) %>%
      summarise(PermSE = sd(Delta, na.rm = TRUE), .groups = "drop")

    tmpResults <- tmpResults %>%
      left_join(permSE_int, by = byVarsInt) %>%
      mutate(
        CI_lo = Delta - 1.96 * PermSE,
        CI_hi = Delta + 1.96 * PermSE
      )
  } else {
    tmpResults <- tmpResults %>%
      mutate(PermSE = NA_real_, CI_lo = NA_real_, CI_hi = NA_real_)
  }

  yTextSz <- max(6, 11 - n_distinct(tmpResults$Comparison) * 0.5)

  tmpPlot <-
    ggplot(tmpResults,
           aes(x = Delta, y = Comparison,
               color = Model, shape = Significant)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
    ## HP5: horizontal error bars — bars crossing zero = not significant
    geom_errorbarh(aes(xmin = CI_lo, xmax = CI_hi),
                   height = 0.25, linewidth = 0.5,
                   position = position_dodge(width = 0.6), na.rm = TRUE) +
    geom_point(size = 3, stroke = 1.2,
               position = position_dodge(width = 0.6)) +
    scale_color_manual(values = modelColors, name = "Model") +
    scale_shape_manual(values = c("TRUE" = 16, "FALSE" = 1),
                       labels = c("TRUE" = "Significant",
                                  "FALSE" = "Not sig."),
                       name   = NULL) +
    facet_grid(DemographicVar ~ Metric,
               scales = "free_y", space = "free_y") +
    labs(title    = "Interaction Bias",
         subtitle = "Error bars = observed \u00b1 1.96\u00d7SE(permutation)",
         x = "\u0394 Performance  (SubgroupA \u2212 SubgroupB)",
         y = "Subgroup Comparison") +
    theme_bw(base_size = 11) +
    theme(axis.text.y      = element_text(size = yTextSz),
          strip.background = element_rect(fill = "grey92"),
          strip.text       = element_text(size = 9),
          legend.position  = "bottom",
          legend.box       = "horizontal",
          legend.key.width = unit(0.4, "cm"),
          plot.title       = element_text(face = "bold"),
          plot.subtitle    = element_text(size = 9, color = "grey40"))

  print(tmpPlot)

  if (!is.na(figureFile)) {
    nComp <- n_distinct(tmpResults$Comparison)
    nMet  <- n_distinct(tmpResults$Metric)
    w <- max(8, nMet * 3.5)
    h <- max(5, ceiling(nComp / 2.5) + 3)
    ggsave(figureFile, width = w, height = h, units = "in", limitsize = FALSE)
  }

  invisible(tmpPlot)
}