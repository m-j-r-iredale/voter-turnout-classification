# Voter Turnout Classification — Experimental Appendix
# Applied Social Data Science final project
#
# This appendix keeps exploratory and model-selection work separate from the
# final deployment model.
#
# Included here:
# - missing-data inspection
# - broad feature-removal logic
# - imputation-indicator dataset
# - high-dimensional Random Forest selection models
# - 100-tree vs 500-tree comparisons
# - weighted-model experiment
# - VSURF experiment
# - intermediate 114-predictor budget model
#
# The final deployment workflow is in R/deployment_model.R.

library(caret)
library(randomForest)
library(VSURF)

data_path <- "data/CCES22_Common_OUTPUT_vv_topost_Clean.csv"

if (!file.exists(data_path)) {
  stop(
    paste0(
      "Input data not found at ", data_path, ". ",
      "See data/README.md for source-data guidance."
    )
  )
}

data <- read.csv(data_path)

# 1. Exploratory missing-data checks
missing_percentages <- colMeans(is.na(data)) * 100

hist(
  missing_percentages,
  breaks = seq(0, 100, by = 10),
  xlab = "Percentage of Missing Data",
  ylab = "Frequency",
  main = "Histogram of Missing Data Percentage",
  xlim = c(0, 100),
  ylim = c(0, 500),
  las = 1,
  cex.axis = 0.8,
  cex.lab = 0.8
)

print(sort(missing_percentages[missing_percentages > 50], decreasing = TRUE))
print(sort(missing_percentages[missing_percentages > 25], decreasing = TRUE))

# 2. Broad feature screening used in the original project
na_percentages <- colMeans(is.na(data)) * 100
na_percentages <- na_percentages[!names(na_percentages) %in% "CC22_309b"]

high_missing_columns <- names(na_percentages[na_percentages >= 50])
data <- data[, !names(data) %in% high_missing_columns]

drop_by_pattern <- function(df, patterns) {
  for (pattern in patterns) {
    matched <- grep(pattern, names(df), value = TRUE, ignore.case = TRUE)
    df <- df[, !names(df) %in% matched, drop = FALSE]
  }
  df
}

data <- drop_by_pattern(
  data,
  c(
    "_timing", "_post", "_t", "_flag",
    "^CC22_365", "^CC22_366", "^CC22_367", "^CC22_309",
    "^CC22_401", "^CC22_402", "^CC22_403", "^CC22_404",
    "^CC22_405", "^CC22_406", "^CC22_411", "^CC22_412",
    "^CC22_413", "^CC22_414", "^CC22_415", "^CC22_416",
    "^CC22_340", "^TS_voterstatus", "^TS_g2022", "^TS_p2022",
    "^TS_state", "^cdid", "multrace", "comptype", "ccesmodule",
    "hispanic", "CC22_310", "CC22_311", "post"
  )
)

explicit_drops <- c(
  "presvote", "regzip", "inputzip", "votereg_f", "inputstate",
  "CCEStake", "add_confirm", "X", "caseid", "commonpostweight",
  "vvweight", "vvweight_post", "tookpost",
  "CC22_350a", "CC22_350b", "CC22_350c", "CC22_350d",
  "CC22_350e", "CC22_350f", "CC22_350g", "CC22_350h",
  "CC22_430", "CC21_362", "CC22_443_1",
  "CC22_309b_1", "CC22_309b_4",
  "countyfips", "lookupzip", "CC22_pid7",
  "CC22_431a", "CC22_431b", "CC22_432a", "CC22_432b",
  "CC22_320d", "CC22_320e", "CC22_320f", "CC22_320g", "CC22_320h"
)

data <- data[, !names(data) %in% explicit_drops, drop = FALSE]

numeric_columns <- sapply(data, is.numeric)
data <- data[, numeric_columns, drop = FALSE]

data <- data[complete.cases(data$CC22_363), ]

data <- drop_by_pattern(
  data,
  c(
    "^CC22_300a", "^CC22_340h", "^CC22_340i", "^CC22_340j",
    "^CC22_340k", "^CC22_340m", "^CC22_340n"
  )
)

# 3. Two imputation strategies
replace_na_with_mode <- function(x) {
  observed <- x[!is.na(x)]
  mode_value <- as.numeric(names(sort(table(observed), decreasing = TRUE)[1]))
  x[is.na(x)] <- mode_value
  x
}

non_iv_data <- data

for (continuous_col in c("CC22_421r", "CC22_422r")) {
  if (continuous_col %in% names(non_iv_data)) {
    non_iv_data[[continuous_col]][is.na(non_iv_data[[continuous_col]])] <-
      mean(non_iv_data[[continuous_col]], na.rm = TRUE)
  }
}

remaining_na <- names(non_iv_data)[colSums(is.na(non_iv_data)) > 0]
non_iv_data[remaining_na] <- lapply(
  non_iv_data[remaining_na],
  replace_na_with_mode
)

iv_data <- data
na_columns <- names(iv_data)[colSums(is.na(iv_data)) > 0]

for (col in na_columns) {
  iv_data[[paste0("IV_", col)]] <- ifelse(is.na(iv_data[[col]]), 1, 0)
}

for (continuous_col in c("CC22_421r", "CC22_422r")) {
  if (continuous_col %in% names(iv_data)) {
    iv_data[[continuous_col]][is.na(iv_data[[continuous_col]])] <-
      mean(iv_data[[continuous_col]], na.rm = TRUE)
  }
}

remaining_na <- names(iv_data)[colSums(is.na(iv_data)) > 0]
iv_data[remaining_na] <- lapply(
  iv_data[remaining_na],
  replace_na_with_mode
)

# 4. Recode response variable
recode_turnout <- function(df) {
  df <- df[!(df$CC22_363 %in% c(2, 4, 6)), ]
  df$CC22_363[df$CC22_363 == 5] <- 0
  df$CC22_363[df$CC22_363 == 3] <- 1
  df
}

non_iv_data <- recode_turnout(non_iv_data)
iv_data <- recode_turnout(iv_data)

# 5. High-dimensional selection model experiments
run_selection_model <- function(
  df,
  mtry_grid = c(12, 16, 20),
  final_mtry = 20,
  ntree = 100,
  replace = TRUE
) {
  y <- factor(df$CC22_363)
  x <- df[, !names(df) %in% "CC22_363", drop = FALSE]
  model_data <- cbind(y = y, x)

  train_size <- floor(nrow(model_data) * 0.80)
  train_index <- sample(seq_len(nrow(model_data)), size = train_size)

  training <- model_data[train_index, ]
  testing <- model_data[-train_index, ]

  cv_model <- train(
    y ~ .,
    data = training,
    method = "rf",
    tuneGrid = expand.grid(mtry = mtry_grid),
    trControl = trainControl(method = "cv", number = 5),
    metric = "Accuracy"
  )

  rf_model <- randomForest(
    y ~ .,
    data = training,
    mtry = final_mtry,
    ntree = ntree,
    replace = replace,
    importance = TRUE
  )

  predictions <- predict(rf_model, newdata = testing)
  confusion <- confusionMatrix(predictions, testing$y)

  list(
    cv_model = cv_model,
    rf_model = rf_model,
    confusion = confusion,
    training = training,
    testing = testing
  )
}

selection_non_iv_100 <- run_selection_model(non_iv_data, ntree = 100)
selection_non_iv_500 <- run_selection_model(non_iv_data, ntree = 500)
selection_iv_100 <- run_selection_model(iv_data, ntree = 100)
selection_iv_500 <- run_selection_model(iv_data, ntree = 500)

# 6. Weighted-model experiment
# The report records that commonweight was tested and then dropped because it
# marginally worsened performance. The original code removed commonweight
# during feature preparation, so this block is retained as documentary
# pseudocode unless a version of the working dataset containing commonweight is
# deliberately constructed.
#
# weights <- training$commonweight
# weighted_model <- train(
#   y ~ .,
#   data = training,
#   method = "rf",
#   weights = weights,
#   tuneGrid = expand.grid(mtry = c(12, 16, 20)),
#   trControl = trainControl(method = "cv", number = 5),
#   metric = "Accuracy"
# )

# 7. VSURF experiment
# The original project tried VSURF but abandoned it because of computational
# load. This call is intentionally not run by default.
#
# vsurf_result <- VSURF(
#   x = selection_non_iv_100$training[, -1],
#   y = selection_non_iv_100$training$y,
#   mtry = 10
# )

# 8. Intermediate budget model
budget_1_predictors <- c(
  "birthyr", "educ", "race", "votereg", "pid3", "pid7", "region",
  "CC22_300_1", "CC22_300_2", "CC22_300_3", "CC22_300_4",
  "CC22_300d_3", "CC22_300d_6", "CC22_302", "CC22_303",
  "CC22_304", "CC22_305_11", "CC22_305_12", "CC22_307",
  "CC22_306", "CC22_320a", "CC22_320b", "CC22_320c", "immstat",
  "CC22_321_2", "CC22_321_3", "CC22_321_4", "CC22_327a",
  "CC22_327c", "CC22_330a", "CC22_330b", "CC22_330c", "CC22_330d",
  "CC22_330e", "CC22_330f", "CC22_331a", "CC22_331b", "CC22_331c",
  "CC22_331d", "CC22_332a", "CC22_332b", "CC22_332c", "CC22_332d",
  "CC22_332e", "CC22_332f", "CC22_333", "CC22_333a", "CC22_333b",
  "CC22_333c", "CC22_333d", "CC22_333e", "CC22_334a", "CC22_334c",
  "CC22_334d", "CC22_334e", "CC22_334f", "CC22_334g", "CC22_355a",
  "CC22_355b", "CC22_355c", "CC22_355d", "CC22_355e", "CC22_361",
  "urbancity", "ideo5", "employ", "investor", "pew_bornagain",
  "pew_religimp", "pew_churatd", "pew_prayer", "religpew", "marstat",
  "union", "union_coverage", "unionhh", "dualcit", "ownhome", "newsint",
  "faminc_new", "milstat_4", "milstat_5", "child18", "healthins_1",
  "healthins_2", "healthins_6", "phone", "internethome", "internetwork",
  "industry", "sexuality", "CC22_420_2", "CC22_420_3", "CC22_420_5",
  "CC22_420_6", "CC22_420_7", "CC22_421r", "CC22_422r", "CC22_423",
  "CC22_424", "CC22_430a_8", "CC22_433a", "CC22_440a", "CC22_440b",
  "CC22_440c", "CC22_440d", "CC22_441a", "CC22_441b", "CC22_443_2",
  "CC22_443_3", "CC22_443_4", "CC22_443_5", "gunown", "numchildren"
)

budget_1_predictors <- intersect(budget_1_predictors, names(non_iv_data))

budget_1_data <- non_iv_data[
  ,
  c("CC22_363", budget_1_predictors),
  drop = FALSE
]

budget_1_result <- run_selection_model(
  budget_1_data,
  mtry_grid = c(10, 12, 15),
  final_mtry = 15,
  ntree = 100
)

# The final nine-variable model is intentionally excluded from this appendix.
# See R/deployment_model.R.
