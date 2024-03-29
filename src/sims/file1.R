###########################
# MANIPULATIONS BY AUTHOR
# 1. NEED TO DOWNLOAD OWN VERSION OF THE DATA AND CHANGE VARIABLE NAMES TO FIT
# 2. 
#
##########################


#########################
#Install packages (uncomment if you have not installed them already)

#install.packages("foreign")
#install.packages("IsingFit")
#install.packages("qgraph")
#install.packages("igraph")
#install.packages("bootnet")
#install.packages("IsingSampler")
#install.packages("compute.es")
#install.packages("NetworkComparisonTest")
#install.packages("mgm")
#install.packages("Matrix")

#########################
#Load packages

library(foreign)
library(IsingFit)
library(qgraph)
library(igraph)
library(bootnet)
library(IsingSampler)
#library(compute.es)
#library(NetworkComparisonTest)
#library(mgm)
#library(Matrix)

setwd("/Users/mo6xj/Projects/ibl-can/src/sims")
#########################
#Load data 
ANES2012 <- read.dta('/Users/mo6xj/Projects/ibl-can/data-in/35157-0001-Data.dta')#loads the data to the object ANES2012

#########################
#Recode variables 
#Items regarding Obama
ObamaCog <- data.frame(Mor = as.numeric(ANES2012$CTRAIT_DPCMORAL),#this creates a data frame containing the items tapping beliefs
                       Led = as.numeric(ANES2012$CTRAIT_DPCLEAD),
                       Car = as.numeric(ANES2012$CTRAIT_DPCCARE),
                       Kno = as.numeric(ANES2012$CTRAIT_DPCKNOW),
                       Int = as.numeric(ANES2012$CTRAIT_DPCINT),
                       Hns = as.numeric(ANES2012$CTRAIT_DPCHONST))

ObamaCog[ObamaCog < 3] <- NA#values below 3 represent missing values
ObamaCog <- binarize(ObamaCog, 5, removeNArows = FALSE)#this binarizes the data (this is done because the model we use for simulating networks assumes binary data); (not) endorsing the beliefs is encoded as 1 (0) 

ObamaAff <- data.frame(Ang = as.numeric(ANES2012$CANDAFF_ANGDPC),#this creates a data frame containing the items tapping feelings
                       Hop = as.numeric(ANES2012$CANDAFF_HPDPC), 
                       Afr = as.numeric(ANES2012$CANDAFF_AFRDPC), 
                       Prd = as.numeric(ANES2012$CANDAFF_PRDDPC))

ObamaAff[ObamaAff < 3] <- NA#values below 3 represent missing values
ObamaAff <- binarize(ObamaAff, 4, removeNArows = FALSE)#(not) endorsing the feelings is encoded as 1 (0)

Obama <- data.frame(ObamaCog,ObamaAff)#this creates a data frame containing all items tapping evaluative reactions
Obama <- na.omit(Obama)#this deletes missing values casewise

######FOR IBL MODEL IN PUT
write.csv(Obama,file="/Users/mo6xj/Projects/ibl-can/data-in/Obama.csv",quote=FALSE,sep=',',row.names=FALSE)

#########################
#Network estimation

ObamaFit <- IsingFit(Obama)
#ObamaGraph <- qgraph(ObamaFit $ weiadj, layout = 'spring', cut = .8)
#ObamaiGraph <- graph_from_adjacency_matrix(abs(ObamaFit $ weiadj), 'undirected', weighted = TRUE, add.colnames = FALSE)

SimInput <- ObamaFit

set.seed(1)

#SampleBeta10 <- IsingSampler(n=100, graph=SimInput$weiadj, 
#                          thresholds=SimInput$thresholds,beta=10,
#                          responses = c(0L,1L))


#write.csv(SampleBeta10,file="/Users/mo6xj/Projects/ibl-can/data_out/can_beta_10.csv",quote=FALSE,sep=',',row.names=FALSE)

#SampleBeta1 <- IsingSampler(n=100, graph=SimInput$weiadj, 
#                             thresholds=SimInput$thresholds,beta=1,
#                             responses = c(0L,1L))

#write.csv(SampleBeta1,file="/Users/mo6xj/Projects/ibl-can/data_out/can_beta_1.csv",quote=FALSE,sep=',',row.names=FALSE)

#SampleBeta005 <- IsingSampler(n=100, graph=SimInput$weiadj, 
#                             thresholds=SimInput$thresholds,beta=0.05,
#                            responses = c(0L,1L))

#write.csv(SampleBeta005,file="/Users/mo6xj/Projects/ibl-can/data_out/can_beta_005.csv",quote=FALSE,sep=',',row.names=FALSE)

#LOOP IT SO CAN LOOK AT NUM ATTRACTORS BY BETA

beta.list <- c(20,10,9,8,7,6,5,4,3,2,1,.5,.25,.125,.075,.03,.01)
for (i in beta.list){
  print(i)
  assign(paste("SampleBeta_",i,sep=""),IsingSampler(n=100, graph=SimInput$weiadj, 
                               thresholds=SimInput$thresholds,beta=i,
                               responses = c(0L,1L)))
  write.csv(get(paste("SampleBeta_",i,sep="")),file=paste("/Users/mo6xj/Projects/ibl-can/data-out/can_beta_",i,".csv",sep=""),quote=FALSE,sep=',',row.names=FALSE)
}


#SIMS FOR PERTURBATION 
SampleHns <- IsingSampler(100, SimInput$weiadj,
                          c(SimInput$thresholds[1:5],1,SimInput$thresholds[7:10]),
                          beta=2,responses = c(0L,1L))
write.csv(SampleHns,file="/Users/mo6xj/Projects/ibl-can/data-out/can_cue_Hns.csv",quote=FALSE,sep=',',row.names=FALSE)

SampleAng <- IsingSampler (100, SimInput$weiadj, 
                           c(SimInput$thresholds[1:6],1,SimInput$thresholds[8:10]),
                           beta=2,responses = c(0L,1L))
write.csv(SampleAng,file="/Users/mo6xj/Projects/ibl-can/data-out/can_cue_Ang.csv",quote=FALSE,sep=',',row.names=FALSE)

SampleMor <- IsingSampler(100, SimInput$weiadj, 
                          c(1,SimInput$thresholds[2:10]),
                          beta=2,responses = c(0L,1L))
write.csv(SampleMor,file="/Users/mo6xj/Projects/ibl-can/data-out/can_cue_Mor.csv",quote=FALSE,sep=',',row.names=FALSE)

SampleLed <- IsingSampler(100, SimInput$weiadj, 
                          c(SimInput$thresholds[1],1,SimInput$thresholds[3:10]),
                          beta=2,responses = c(0L,1L))
write.csv(SampleLed,file="/Users/mo6xj/Projects/ibl-can/data-out/can_cue_Led.csv",quote=FALSE,sep=',',row.names=FALSE)

SampleCar <- IsingSampler(100, SimInput$weiadj, 
                          c(SimInput$thresholds[1:2],1,SimInput$thresholds[4:10]),
                          beta=2,responses = c(0L,1L))
write.csv(SampleCar,file="/Users/mo6xj/Projects/ibl-can/data-out/can_cue_Car.csv",quote=FALSE,sep=',',row.names=FALSE)

SampleKno <- IsingSampler(100, SimInput$weiadj, 
                          c(SimInput$thresholds[1:3],1,SimInput$thresholds[5:10]),
                        beta=2,responses = c(0L,1L))
write.csv(SampleKno,file="/Users/mo6xj/Projects/ibl-can/data-out/can_cue_Kno.csv",quote=FALSE,sep=',',row.names=FALSE)

SampleInt <- IsingSampler(100, SimInput$weiadj, 
                          c(SimInput$thresholds[1:4],1,SimInput$thresholds[6:10]),
                        beta=2,responses = c(0L,1L))
write.csv(SampleInt,file="/Users/mo6xj/Projects/ibl-can/data-out/can_cue_Int.csv",quote=FALSE,sep=',',row.names=FALSE)

SampleHop <- IsingSampler(100, SimInput$weiadj, 
                          c(SimInput$thresholds[1:7],1,SimInput$thresholds[9:10]),
                        beta=2,responses = c(0L,1L))
write.csv(SampleHop,file="/Users/mo6xj/Projects/ibl-can/data-out/can_cue_Hop.csv",quote=FALSE,sep=',',row.names=FALSE)

SampleAfr <- IsingSampler(100, SimInput$weiadj, 
                          c(SimInput$thresholds[1:8],1,SimInput$thresholds[10]),
                        beta=2,responses = c(0L,1L))
write.csv(SampleAfr,file="/Users/mo6xj/Projects/ibl-can/data-out/can_cue_Afr.csv",quote=FALSE,sep=',',row.names=FALSE)

SamplePrd <- IsingSampler(100, SimInput$weiadj, 
                          c(SimInput$thresholds[1:9],1),
                        beta=2,responses = c(0L,1L))
write.csv(SamplePrd,file="/Users/mo6xj/Projects/ibl-can/data-out/can_cue_Prd.csv",quote=FALSE,sep=',',row.names=FALSE)

#EOF


