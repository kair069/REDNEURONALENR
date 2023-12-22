library(shiny)
library(neuralnet)
library(e1071)

library(shiny)
library(neuralnet)
library(e1071)

ui <- shinyUI(
  fluidPage(
    # Application title
    titlePanel("Machine Learning Application"),
    hr(),
    # Show a plot of the generated distribution
    tabsetPanel(
      tabPanel("Machine Learning Application",
               sidebarPanel(
                 h3("Choose Model Type"),
                 selectInput("modeltype", label = 'Select Model from the options',
                             choices = list("Artificial Neural Network" = 1, "Support Vector Machine" = 2)),
                 hr(),
                 h3("Model Parameters"),
                 fileInput('file1', h5('Choose Training Data File (CSV)'),
                           accept = c('text/csv',
                                      'text/comma-separated-values,text/plain',
                                      '.csv')),
                 tags$hr(),
                 splitLayout(
                   div(checkboxInput('header', 'Header', TRUE),
                       radioButtons('sep', 'Separator',
                                    c(Comma = ',',
                                      Semicolon = ';',
                                      Tab = '\t'),
                                    ',')),
                   radioButtons('quote', 'Quote',
                                c(None = '',
                                  'Double Quote' = '"',
                                  'Single Quote' = "'"),
                                '"')
                 ),
                 hr(),
                 uiOutput(outputId = "features"),
                 textOutput(outputId = "test_checkbox"),
                 div(textOutput(outputId = "feature_error"), style = "color:red;font-weight:200"),
                 uiOutput(outputId = "ann_parameters"),
                 width = 5
               ),
               mainPanel(
                 # Panel only displayed after the file is read
                 conditionalPanel(condition = "output.ann_parameters !== null",
                                  div(
                                    h4("Preview Of Read Data: "),
                                    helpText("Showing 10 Columns only"),
                                    # This is the preview of the read file
                                    tableOutput(outputId = "contents"),
                                    hr(),
                                    h4("Model Plot"),
                                    plotOutput("ann_plot"),
                                    hr(),
                                    h4("Model Overview"),
                                    uiOutput("ann_printstats"),
                                    hr(),
                                    h4("Model Output"),
                                    tableOutput(outputId = "ann_result")
                                  )
                 ),
                 width = 7
               )
      )
    )
  )
)


train_data <<- NULL

library(shiny)
library(neuralnet)
library(e1071)

ui <- shinyUI(
  fluidPage(
    # Application title
    titlePanel("Machine Learning Application"),
    hr(),
    # Show a plot of the generated distribution
    tabsetPanel(
      tabPanel("Machine Learning Application",
               sidebarPanel(
                 h3("Choose Model Type"),
                 selectInput("modeltype", label = 'Select Model from the options',
                             choices = list("Artificial Neural Network" = 1, "Support Vector Machine" = 2)),
                 hr(),
                 h3("Model Parameters"),
                 fileInput('file1', h5('Choose Training Data File (CSV)'),
                           accept = c('text/csv',
                                      'text/comma-separated-values,text/plain',
                                      '.csv')),
                 tags$hr(),
                 splitLayout(
                   div(checkboxInput('header', 'Header', TRUE),
                       radioButtons('sep', 'Separator',
                                    c(Comma = ',',
                                      Semicolon = ';',
                                      Tab = '\t'),
                                    ',')),
                   radioButtons('quote', 'Quote',
                                c(None = '',
                                  'Double Quote' = '"',
                                  'Single Quote' = "'"),
                                '"')
                 ),
                 hr(),
                 uiOutput(outputId = "features"),
                 textOutput(outputId = "test_checkbox"),
                 div(textOutput(outputId = "feature_error"), style = "color:red;font-weight:200"),
                 uiOutput(outputId = "ann_parameters"),
                 width = 5
               ),
               mainPanel(
                 # Panel only displayed after the file is read
                 conditionalPanel(condition = "output.ann_parameters !== null",
                                  div(
                                    h4("Preview Of Read Data: "),
                                    helpText("Showing 10 Columns only"),
                                    # This is the preview of the read file
                                    tableOutput(outputId = "contents"),
                                    hr(),
                                    h4("Model Plot"),
                                    plotOutput("ann_plot"),
                                    hr(),
                                    h4("Model Overview"),
                                    uiOutput("ann_printstats"),
                                    hr(),
                                    h4("Model Output"),
                                    tableOutput(outputId = "ann_result")
                                  )
                 ),
                 width = 7
               )
      )
    )
  )
)

train_data <<- NULL

shinyServer(function(input, output, session) {
  
  # ... (rest of the code)
  
  # ----------------------------------------------
  # This renders parameters for the ANN model 
  # ----------------------------------------------
  
  output$ann_parameters <- renderUI({
    
    if(is.null(input$prediction_radio)) {return(NULL)}
    
    else if(input$modeltype == 1) {  # Return ann parameters
      return(div(br(), hr(), h4("Neural Network Parameters"),
                 numericInput("hiddenlayers", label= h5("Hidden Layers"), value=3),
                 numericInput("threshold", label= h5("Stop Threshold"), value= 0.01),
                 helpText("Numeric value specifying the threshold for the partial derivatives of the error function as stopping criteria."),
                 sliderInput("stepmax" , label = h5("Maximum Steps (10^x)"), min= 3, max = 7 ,value=5),
                 helpText("The maximum steps for the training of the neural network."),
                 checkboxInput("linear_out" , label="Scale Output", value=F ), 
                 helpText("Should the logistic function be applied to ANN output"),
                 hr(), h4("Testing Parameters"), 
                 checkboxInput("sample_from_train", label="Sample From Training Data" , value=FALSE),
                 uiOutput("testing_parameters"),
                 actionButton("runIt", label= "Run ANN")
      ))
    } else if(input$modeltype == 2) {  # Otherwise return SVM Parameters
      return(div(br(), hr(), h4("Support Vector Machine Parameters"),
                 radioButtons("kernel", label= h5("Kernel Type"), 
                              choices=list("Linear"="linear", "Polynomial"= "polynomial", "Radial Basis"="radial", "Sigmoid"= "sigmoid")), 
                 numericInput("costparam", label= h5("Cost Parameter"), value= 1),
                 helpText("Numeric value specifying the cost of constraint violation"),
                 numericInput("threshold", label= h5("Stop Threshold"), value= 0.001),
                 helpText("Numeric value specifying the tolerance of termination"),
                 checkboxInput("scale_variables" , label=h5("Scale Variables"), value=F ), 
                 helpText("Should the variables be scaled"),
                 checkboxInput("scale_output" , label="Scale Output", value=F ), 
                 helpText("Should the logistic function be applied to the output"),
                 hr(), h4("Testing Parameters"), 
                 checkboxInput("sample_from_train", label="Sample From Training Data" , value=FALSE),
                 uiOutput("testing_parameters"),
                 actionButton("runIt", label= "Run SVM")
      ))
    }
  })
  
  # ... (rest of the code)
  
  # ----------------------------------------------
  # This load the testing parameters. E.g. sampling from data or specifying a file
  # ----------------------------------------------
  
  output$testing_parameters <- renderUI({
    if(input$sample_from_train == TRUE){
      return (sliderInput("percent_train" , label= "Percentage Training Data" , min =0.0, max =1.0, value=0.5) )
    } else if((!input$sample_from_train) == TRUE){
      return(
        fileInput('testfile', h5('Test Data File (CSV)'),
                  accept=c('text/csv', 
                           'text/comma-separated-values,text/plain', 
                           '.csv'))
      )
    }
    
    return (NULL);
  })
  
  # ... (rest of the code)
  
  # ----------------------------------------------
  # This function renders the output table with predicted values
  # ----------------------------------------------
  
  output$ann_result <- renderTable({
    if(is.null(input$runIt) ){return(NULL)}
    
    if( (input$runIt == 0)  ){
      return(NULL)
    }
    if(input$runIt >0 ){
      withProgress(message="Processing!", value=0.1, {
        # ... (rest of the code)
      })
    }
  })
  
  # ... (rest of the code)
  
})

shinyApp(ui = ui, server = server)