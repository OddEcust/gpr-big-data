calculate_RMSE <- function(actual_vector, prediction_vector){
  calculated_RMSE <- 
    sqrt(sum((actual_vector - prediction_vector)^2)/length(actual_vector))
  return(calculated_RMSE)
}