library(VineCopula)
library(fitdistrplus)
library(ks)
library(MASS)

setwd('./RESULTS_SAMPLE/Snaps_LoopTrue')
trainFile='TrainData_50.csv'
testFile='TestData_50.csv'
trainProbsFile='TrainProbs_50.csv'
testProbsFile='TestProbs_50.csv'

trainData <- read.csv(trainFile, sep=' ', header=FALSE)
testData <- read.csv(testFile, sep=' ', header=FALSE)
trainProbs <- read.csv(trainProbsFile, sep=' ', header=FALSE)
testProbs <- read.csv(testProbsFile, sep=' ', header=FALSE)

trainData <- as.matrix(trainData)
testData <- as.matrix(testData)
trainProbs <- as.matrix(trainProbs)
testProbs <- as.matrix(testProbs)

cvine <- RVineStructureSelect(trainData, type=1)
ll <- RVineLogLik(trainData, cvine)$loglik

pdf <- RVinePDF(trainData, cvine)
PDF <- pdf * trainProbs[,1] * trainProbs[,2]

nrow = dim(trainData)[2]
ncol = dim(trainData)[2]
pdf("Train_PP.pdf", height=100, width=100)
par(mfrow=c(nrow, ncol), cex=0.5, mai=c(0.1, 0.1, 0.1, 0.1))
par("mar")
par(mar=c(1,1,1,1))
for(row in 1:ncol)
  for(col in 1:ncol)
    if(row<=col){
      plot.new()
      text(0.5, 0.5, cor(trainData[,row], trainData[,col]), cex=16)} else{
        fhat <- kde(x = trainData[,c(row,col)])
        #plot(mod_data[,row], mod_data[,col],cex=0.1)
        plot(fhat, display="filled.contour2", cont=seq(10,90, by=10))}
dev.off()

sim_data <- RVineSim(10000, cvine)
# Save sim_data to file
write.table(sim_data, file="Simulated Data from Fitted Copula", sep=" ", col.names = F, row.names = F)

pdf("SIM_PP.pdf", height=100, width=100)
par(mfrow=c(nrow, ncol), cex=0.5, mai=c(0.1, 0.1, 0.1, 0.1))
par("mar")
par(mar=c(1,1,1,1))
for(row in 1:ncol)
  for(col in 1:ncol)
    if(row<=col){
      plot.new()
      text(0.5, 0.5, cor(sim_data[,row], sim_data[,col]), cex=16)} else{
        fhat <- kde(x = sim_data[,c(row,col)])
        #plot(mod_data[,row], mod_data[,col],cex=0.1)
        plot(fhat, display="filled.contour2", cont=seq(10,90, by=10))}
dev.off()

pdf <- RVinePDF(testData, cvine) * testProbs[,1] * testProbs[,2]
logpdf <- log(pdf)
write.table(logpdf, file="LogPDF.txt", sep=" ", col.names=F, row.names=T)
