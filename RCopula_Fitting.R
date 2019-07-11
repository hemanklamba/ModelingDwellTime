# Loads the package - copula
library(VineCopula)
library(fitdistrplus)
library(ks)
library(MASS)

setwd('/Users/hemanklamba/Documents/Experiments/Snap_DwellTime/')
################### Loop=FALSE Setting #########################
ovFile = 'Original Parameter File'
trainFile='Train Data Parameters'
testFile='Test Data Parameters'
trainProbsFile='File containing probability for observing each train instance according to the ind. models'
testProbsFile='File containing probability for observing each test instance according to the ind. models'
#synthFile = './AnalysisFiles/LoopFalse_Copula/2021Sep/SynthData.csv'
#synthProbsFile = './AnalysisFiles/LoopFalse_Copula/2021Sep/SynthProb.csv'

trainData <- read.csv(trainFile, sep=' ', header=FALSE)
testData <- read.csv(testFile, sep=' ', header=FALSE)
ovData <- read.csv(ovFile, sep=' ', header=FALSE)
trainProbs <- read.csv(trainProbsFile, sep=' ', header=FALSE)
testProbs <- read.csv(testProbsFile, sep=' ', header=FALSE)
#synthData <- read.csv(synthFile, sep=' ', header=FALSE)
#synthProbs <- read.csv(synthProbsFile, sep=' ', header=FALSE)

trainData <- as.matrix(trainData)
testData <- as.matrix(testData)
ovData <- as.matrix(ovData)
reGenData <- as.matrix(reGenData)
trainProbs <- as.matrix(trainProbs)
testProbs <- as.matrix(testProbs)
#synthData <- as.matrix(synthData)
#synthProbs <- as.matrix(synthProbs)

nrow = dim(trainData)[2]
ncol = dim(trainData)[2]

pdf("Original File Pair Plot over Parameters", height=100, width=100)
par(mfrow=c(nrow, ncol), cex=0.5, mai=c(0.1, 0.1, 0.1, 0.1))
par("mar")
par(mar=c(1,1,1,1))
for(row in 1:ncol)
  for(col in 1:ncol)
    if(row<=col){
      plot.new()
      text(0.5, 0.5, cor(ovData[,row], ovData[,col]), cex=16)} else{
        fhat <- kde(x = ovData[,c(row,col)])
        #plot(mod_data[,row], mod_data[,col],cex=0.1)
        plot(fhat, display="filled.contour2", cont=seq(10,90, by=10))}
dev.off()

pdf("Pair Plot over Parameters for training instances", height=100, width=100)
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

# Now Fitting CVine Copula
cvine <- RVineStructureSelect(trainData, type=1)
ll <- RVineLogLik(trainData, cvine)$loglik

sim_data <- RVineSim(10000, cvine)
# Save sim_data to file
write.table(sim_data, file="Simulated Data from Fitted Copula", sep=" ", col.names = F, row.names = F)

reGenFile='./AnalysisFiles/LoopFalse_Copula/2021Sep/ReGenData_Simulated.txt'
reGenData <- read.csv(reGenFile, sep=' ', header=FALSE)

pdf("Simulated Data Pair Plot over Parameters", height=100, width=100)
par(mfrow=c(nrow, ncol), cex=0.5, mai=c(0.1, 0.1, 0.1, 0.1))
par("mar")
par(mar=c(1,1,1,1))
for(row in 1:ncol)
  for(col in 1:ncol)
    if(row<=col){
      plot.new()
      text(0.5, 0.5, cor(reGenData[,row], reGenData[,col]), cex=16)} else{
        fhat <- kde(x = reGenData[,c(row,col)])
        #plot(mod_data[,row], mod_data[,col],cex=0.1)
        plot(fhat, display="filled.contour2", cont=seq(10,90, by=10))}
dev.off()

pdf("Pair PDF Plots over Sampled Instances after Copula Fitting", height=100, width=100)
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

pdf <- RVinePDF(testData, cvine) * testProbs[,1] * testProbs[,2] * testProbs[,3]
logpdf <- log(pdf)
