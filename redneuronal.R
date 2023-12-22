# Tratamiento de datos y gráficos
install.packages("recipes")
# ==============================================================================
# Tratamiento de datos y gráficos
# ==============================================================================library(tidymodels)
library(tidyverse)
library(skimr)
library(DataExplorer)
library(ggpubr)
library(mosaicData)
library(rsample)
library(tidymodels)

#install.packages("tidymodels")
#install.packages("tidyverse")

#explicaciones
#install.packages("caret")
#install.packages("DALEX")
#install.packages("lime")

# Modelado
# ==============================================================================
# sudo apt install default-jre
library(h2o)

view(SaratogaHouses)
#str(SaratogaHouses)
#His(SaratogaHouses$price)

data("SaratogaHouses", package = "mosaicData")
datos <- SaratogaHouses

# Se renombran las columnas para que sean más descriptivas
colnames(datos) <- c("precio", "metros_totales", "antiguedad", "precio_terreno",
                     "metros_habitables", "universitarios",
                     "dormitorios", "chimenea", "banyos", "habitaciones",
                     "calefaccion", "consumo_calefacion", "desague",
                     "vistas_lago","nueva_construccion", "aire_acondicionado")

##
#
#
# Especifica la ruta donde deseas guardar el archivo CSV
ruta_csv <- "C:/Users/Acer/Downloads/Machine_Learning_Online-master/Machine_Learning_Online-master/mi_data.csv"

# Exporta el dataframe a un archivo CSV
write.csv(datos, file = ruta_csv, row.names = FALSE)

# Mensaje de confirmación
cat("Datos exportados exitosamente a", ruta_csv, "\n")




# Exporta el dataframe a un archivo CSV
write.csv(mi_data, file = ruta_csv, row.names = FALSE)

# Mensaje de confirmación
cat("Datos exportados exitosamente a", ruta_csv, "\n")

# Tabla resumen
# ==============================================================================
skim(datos)



?SaratogaHouses


plot_missing(
  data    = datos, 
  title   = "Porcentaje de valores ausentes",
  ggtheme = theme_bw(),
  theme_config = list(legend.position = "none")
)


# Distribución variable respuesta
# ==============================================================================
ggplot(data = datos, aes(x = precio)) +
  geom_density(fill = "steelblue", alpha = 0.8) +
  geom_rug(alpha = 0.1) +
  scale_x_continuous(labels = scales::comma) +
  labs(title = "Distribución original") +
  theme_bw() 


# Tabla de estadísticos principales 
summary(datos$precio)


# Gráfico de distribución para cada variable numérica
# ==============================================================================
plot_density(
  data    = datos %>% select(-precio),
  ncol    = 3,
  title   = "Distribución variables continuas",
  ggtheme = theme_bw(),
  theme_config = list(
    plot.title = element_text(size = 14, face = "bold"),
    strip.text = element_text(colour = "black", size = 12, face = 2)
  )
)



# Valores observados de chimenea
# ==============================================================================
table(datos$chimenea)


# Se convierte la variable chimenea a factor.
datos <- datos %>%
  mutate(chimenea = as.factor(chimenea))



# Gráfico para cada variable cualitativa
# ==============================================================================
plot_bar(
  datos,
  ncol    = 3,
  title   = "Número de observaciones por grupo",
  ggtheme = theme_bw(),
  theme_config = list(
    plot.title = element_text(size = 14, face = "bold"),
    strip.text = element_text(colour = "black", size = 8, face = 2),
    legend.position = "none"
  )
)

table(datos$chimenea)



datos <- datos %>%
  mutate(
    chimenea = recode_factor(
      chimenea,
      `2` = "2_mas",
      `3` = "2_mas",
      `4` = "2_mas"
    )
  )

table(datos$chimenea)




# Reparto de datos en train y test
# ==============================================================================
set.seed(123)

split_inicial <- initial_split(
  data   = datos,
  prop   = 0.8,
  strata = precio
)

datos_train <- training(split_inicial)
datos_test  <- testing(split_inicial)

summary(datos_train$precio)



summary(datos_test$precio)



# Se almacenan en un objeto `recipe` todos los pasos de preprocesado y, finalmente,
# se aplican a los datos.
transformer <- recipe(
  formula = precio ~ .,
  data =  datos_train
) %>%
  step_naomit(all_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes())

transformer



# Se entrena el objeto recipe
transformer_fit <- prep(transformer)
transformer_fit

# Se aplican las transformaciones al conjunto de entrenamiento y de test
datos_train_prep <- bake(transformer_fit, new_data = datos_train)
datos_test_prep  <- bake(transformer_fit, new_data = datos_test)

glimpse(datos_train_prep)



# Inicialización del cluster
# ==============================================================================
h2o.init(
  nthreads = -1,
  max_mem_size = "4g"
)


# Se eliminan los datos del cluster por si ya había sido iniciado.
###
h2o.removeAll()
h2o.no_progress()



datos_train  <- as.h2o(datos_train_prep, key = "datos_train")
datos_test   <- as.h2o(datos_test_prep, key = "datos_test")


# Espacio de búsqueda de cada hiperparámetro
# ==============================================================================
hiperparametros <- list(
  epochs = c(50, 100, 500),
  hidden = list(5, 10, 25, 50, c(10, 10))
)


# Búsqueda por validación cruzada
# ==============================================================================
variable_respuesta <- 'precio'
predictores <- setdiff(colnames(datos_train), variable_respuesta)

grid <- h2o.grid(
  algorithm    = "deeplearning",
  activation   = "Rectifier",
  x            = predictores,
  y            = variable_respuesta,
  training_frame  = datos_train,
  nfolds       = 3, #validacion cruzada
  standardize  = FALSE,
  hyper_params = hiperparametros,
  search_criteria = list(strategy = "Cartesian"),
  seed         = 123,
  grid_id      = "grid"
)



# Resultados del grid
# ==============================================================================
resultados_grid <- h2o.getGrid(
  sort_by = 'rmse',
  grid_id = "grid",
  decreasing = FALSE
)
data.frame(resultados_grid@summary_table)



# Mejor modelo encontrado
# ==============================================================================
modelo_final <- h2o.getModel(resultados_grid@model_ids[[1]])
modelo_final


predicciones <- h2o.predict(
  object  = modelo_final,
  newdata = datos_test
)

predicciones <- predicciones %>%
  as_tibble() %>%
  mutate(valor_real = as.vector(datos_test$precio))

predicciones %>% head(5)


rmse(predicciones, truth = valor_real, estimate = predict, na_rm = TRUE)


modelo_final@allparameters

#####rendimiento####

# Suponiendo que 'predicciones' es tu dataframe con las predicciones y valores reales
library(dplyr)
library(Metrics)
#install.packages("Metrics")
# Calcular el MAE (Error Absoluto Medio)
mae <- mean(abs(predicciones$predict - predicciones$valor_real))
cat("MAE:", mae, "\n")

# Calcular el MSE (Error Cuadrático Medio)
mse <- mean((predicciones$predict - predicciones$valor_real)^2)
cat("MSE:", mse, "\n")

# Calcular el R^2 (Coeficiente de Determinación)
rss <- sum((predicciones$predict - predicciones$valor_real)^2)
tss <- sum((predicciones$valor_real - mean(predicciones$valor_real))^2)
r2 <- 1 - rss / tss
cat("R^2:", r2, "\n")

#explicabilidad

# Cargar la biblioteca 
library(lime)
library(data.table)
# Convertir los datos de prueba a data.table
datos_test_dt <- as.data.table(as.data.frame(datos_test_prep))

# Función para predecir con contribuciones
predict_contributions <- function(model, data) {
  contribs <- h2o.predict_contributions(model, data)
  as.data.frame(contribs)
}

# Crear el modelo para lime
model_lime <- lime(datos_train_prep, model = modelo_final, bin_continuous = FALSE)

# Seleccionar una observación de prueba para explicar
observacion_a_explicar <- datos_test_dt[1, ]

# Obtener la explicación de lime
explicacion <- explain(observacion_a_explicar, model_lime, n_features = 19)

# Mostrar la explicación
print(explicacion)
glimpse(explicacion)
View(explicacion)
# Visualizar la explicación con lime

lime::plot_features(explicacion)
###
lime::plot_explanations(explicacion)


# Obtener las predicciones del modelo
# Obtén la importancia de variables
#if (!requireNamespace("devtools", quietly = TRUE)) {
#  install.packages("devtools")
#}
library(devtools)
#install_version("caret", version = "6.0-92", repos = "http://cran.us.r-project.org")
library(caret)
importancia_variables <- h2o.varimp(modelo_final)

# Gráfico de barras de importancia de variables
library(ggplot2)
ggplot(importancia_variables, aes(x = reorder(variable, relative_importance), y = relative_importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Importancia de Variables", x = "Variable", y = "Importancia Relativa") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#
# Propiedades 
modelo_final@allparameters  # Obtén todos los parámetros del modelo

modelo_final@model_id  # Obtén la identificación única del modelo

modelo_final@model  # Obtén un resumen del modelo, incluyendo estadísticas de entrenamiento y validación

modelo_final@algorithm

modelo_final@have_mojo

modelo_final@params














#
sesion_info <- devtools::session_info()
dplyr::select(
  tibble::as_tibble(sesion_info$packages),
  c(package, loadedversion, source)
)
h2o.shutdown(prompt = FALSE)