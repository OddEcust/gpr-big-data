calculate_MAE <- function(actual_vector, prediction_vector){
  calculated_MAE<- 
    sum(abs(actual_vector - prediction_vector))/length(actual_vector)
  return(calculated_MAE)
}