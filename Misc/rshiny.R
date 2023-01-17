library(gapminder)
library(shiny)
library(tidyverse)

ui <- fluidPage(
  h1("Gapminder"),
  sidebarLayout(
    sidebarPanel(
      sliderInput(inputId = "life", label = "Life expectancy",
                  min = 0, max = 120,
                  value = c(30, 50)),
      selectInput("continent", "Continent",
                  choices = c("All", levels(gapminder$continent))),
      downloadButton(outputId = "download_data", label = "Download")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Plot", plotOutput("plot")),
        tabPanel("Table", DT::dataTableOutput("table"))
      )
    )
  )
)
server <- function(input, output) {
  filtered_data <- reactive({
    # Filter the data (copied from previous exercise)
    data <- gapminder %>%
            filter(lifeExp >= input$life[1] & lifeExp <= input$life[2])
    if (input$continent != "All") {
      data <- subset(
        data,
        continent == input$continent
      )
    }
    data
  })
  # Create the table render function
  output$table <- DT::renderDataTable({
    data <- filtered_data()
    data
  })
  
  # Create the plot render function  
  output$plot <- renderPlot({
    data <- filtered_data()
    ggplot(data, aes(gdpPercap, lifeExp)) +
      geom_point() +
      scale_x_log10()
  })
  # Create download function
  output$download_data <- downloadHandler(
    filename = "gapminder_data.csv",
    content = function(file) {
      # Use the filtered_data variable to create the data for
      # the downloaded file
      data <- filtered_data()
      write.csv(data, file, row.names = FALSE)
    }
  )
}

shinyApp(ui, server)