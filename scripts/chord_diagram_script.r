# Libraries
library(tidyverse)
library(viridis)
library(patchwork)
library(hrbrthemes)
library(circlize)
library(chorddiag)  #devtools::install_github("mattflor/chorddiag")

# Load dataset
data <- read.csv('data/results/instances_output/summary_outputs/rcp26_flowmatrix2020to2050.csv', row.names = 1, header= TRUE)

data_long <- data %>%
  rownames_to_column %>%
  gather(key = 'key', value = 'value', -rowname)

data_long <- subset(data_long, data_long[c("rowname")] != data_long[c("key")])


# parameters
circos.clear()
circos.par(start.degree = 90, gap.degree = 4, track.margin = c(-0.1, 0.1), points.overflow.warning = FALSE)
par(mar = rep(0, 4))

# color palette
set.seed(2)
mycolor <- viridis(11, alpha = 1, begin = 0, end = 1, option = "D")
mycolor <- mycolor[sample(1:11)]

# Base plot
circos.par(circle.margin = 0.28)

chordDiagram(
  x = data_long, 
  grid.col = mycolor,
  transparency = 0.25,
  directional = 1,
  direction.type = c("arrows", "diffHeight"), 
  diffHeight  = -0.04,
  annotationTrack = "grid", 
  annotationTrackHeight = c(0.05, 0.1),
  link.arr.type = "big.arrow", 
  link.sort = TRUE, 
  link.largest.ontop = TRUE)

# Add text and axis
circos.trackPlotRegion(
  track.index = 1, 
  bg.border = NA, 
  panel.fun = function(x, y) {
    
    xlim = get.cell.meta.data("xlim")
    sector.index = get.cell.meta.data("sector.index")
    
    # Add names to the sector. 
    circos.text(
      x = mean(xlim), 
      y = 7.5, 
      labels = sector.index, 
      facing = "clockwise",
      niceFacing = TRUE,
      font = 2,
      cex = 0.8
    )
    
    # Add graduation on axis
    circos.axis(
      h = "top",
      minor.ticks = 1,
      major.tick.length = 0.3,
      labels.facing = "clockwise",
      labels.cex = 0.7,
      labels.niceFacing = TRUE)
  }
)