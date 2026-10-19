# Voter Turnout Classification — Deployment Model
# Applied Social Data Science final project
#
# Purpose:
# Reproduce the final "budget" model using only the nine predictors selected
# during the original modelling process.
#
# Expected input:
# data/CCES22_Common_OUTPUT_vv_topost_Clean.csv
#
# Required packages:
#   caret
#   randomForest
#
# Notes:
# - The response variable is CC22_363.
# - Responses 2, 4 and 6 are excluded, matching the original project.
# - Response 3 ("already voted") is recoded to 1 and response 5 ("no") to 0.
# - Missing predictor values are imputed using the mode, matching the original
#   final-model workflow.
# - The original project did not set a random seed, so exact train/test results
#   can vary between runs. A seed is set here for reproducibility.

library(caret)
library(randomForest)

set.seed(2022)

data_path <- "data/CCES22_Common_OUTPUT_vv_topost_Clean.csv"
target <- "CC22_363"

predictors <- c(
  "birthyr", "votereg", "pid3", "pid7",
  "CC22_320a", "CC22_320b", "CC22_320c",
  "ideo5", "newsint"
)

replace_na_with_mode <- function(x) {
  observed <- x[!is.na(x)]

  if (length(observed) == 0) {
    stop("Cannot impute a column containing only missing values.")
  }

  mode_value <- as.numeric(names(sort(table(observed), decreasing = TRUE)[1]))
  x[is.na(x)] <- mode_value
  x
}

calculate_metrics <- function(conf_matrix) {
  cm <- conf_matrix$table

  tn <- cm["1", "1"]
  fp <- cm["0", "1"]
  fn <- cm["1", "0"]
  tp <- cm["0", "0"]

  precision <- tp / (tp + fp)
  recall <- tp / (tp + fn)
  specificity <- tn / (tn + fp)
  false_positive_rate <- fp / (fp + tn)
  miss_rate <- fn / (fn + tp)
  f1 <- 2 * precision * recall / (precision + recall)

  data.frame(
    accuracy = sum(diag(cm)) / sum(cm),
    precision = precision,
    recall = recall,
    specificity = specificity,
    false_positive_rate = false_positive_rate,
    miss_rate = miss_rate,
    f1_score = f1
  )
}

if (!file.exists(data_path)) {
  stop(
    paste0(
      "Input data not found at ", data_path, ". ",
      "See data/README.md for source-data guidance."
    )
  )
}

data <- read.csv(data_path)

required_columns <- c(target, predictors)
missing_columns <- setdiff(required_columns, names(data))

if (length(missing_columns) > 0) {
  stop(
    paste(
      "The input file is missing required columns:",
      paste(missing_columns, collapse = ", ")
    )
  )
}

model_data <- data[, required_columns]

model_data <- model_data[!(model_data[[target]] %in% c(2, 4, 6)), ]
model_data <- model_data[!is.na(model_data[[target]]), ]

model_data[[target]][model_data[[target]] == 3] <- 1
model_data[[target]][model_data[[target]] == 5] <- 0

model_data[predictors] <- lapply(
  model_data[predictors],
  replace_na_with_mode
)

model_data$y <- factor(model_data[[target]], levels = c(0, 1))
model_data[[target]] <- NULL

train_size <- floor(nrow(model_data) * 0.80)
train_index <- sample(seq_len(nrow(model_data)), size = train_size)

training <- model_data[train_index, ]
testing <- model_data[-train_index, ]

tuning_grid <- expand.grid(
  mtry = c(3, 4, 5)
)

cv_model <- train(
  y ~ .,
  data = training,
  method = "rf",
  tuneGrid = tuning_grid,
  trControl = trainControl(method = "cv", number = 5),
  metric = "Accuracy"
)

print(cv_model$bestTune)

final_model <- randomForest(
  y ~ .,
  data = training,
  mtry = 3,
  ntree = 100,
  replace = TRUE,
  importance = TRUE
)

print(final_model)
print(importance(final_model))

predictions <- predict(final_model, newdata = testing)

conf_matrix <- confusionMatrix(
  predictions,
  testing$y,
  positive = "0"
)

print(conf_matrix)

metrics <- calculate_metrics(conf_matrix)
print(metrics)

dir.create("outputs", showWarnings = FALSE)

png(
  filename = "outputs/final_model_variable_importance.png",
  width = 1600,
  height = 1000,
  res = 160
)

varImpPlot(
  final_model,
  main = "Final Budget Model — Variable Importance",
  type = 2,
  scale = FALSE
)

dev.off()

write.csv(
  metrics,
  "outputs/final_model_metrics.csv",
  row.names = FALSE
)
