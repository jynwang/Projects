setwd("D:/MSFE/Practicum/data");
#####################################################
#draw plots to get empirical understanding of spreads, which are stored in the pdf named: spreads of each day.pdf
pdf("spreads of each day.pdf")
filenames = c("20130401","20130402","20130403","20130404","20130405")
ndays = 5
for(i in 1:ndays)
{
    filename = filenames[i]
    mydata <- read.csv(file=paste("D:/MSFE/Practicum/data/",filename,".csv",sep=""), header = TRUE, sep = ",", quote = "\"", dec = ".", fill = TRUE)
    names(mydata)
    attach(mydata)
    #convert the time to decimal hour
    mydata[,1] = mydata[,1]%/%10000000+(mydata[,1]%%10000000)/6000000
    mydata[,2] = mydata[,2]%/%10000000+(mydata[,2]%%10000000)/6000000
    plot(mydata[,1],mydata[,17],xlab = "begin_interval", ylab = "effective_spread",main = paste("Efficient Spread during",filename,sep=" "))
   
    #plot three histograms: hist(data,breakpoints,x-axis_range_shown_in_plot )
    hist(Time_Weigthed_Spread,c(seq(min(Time_Weigthed_Spread,na.rm=TRUE)-0.001,max(Time_Weigthed_Spread,na.rm=TRUE)+0.001,0.001)), xlim=c(-0.02,0.02),main = paste("Histogram of Time Weighted Spread of ",filename,sep=" "))
    hist(Volume_Weighted_Spread,c(seq(min(Volume_Weighted_Spread,na.rm=TRUE)-0.001,max(Volume_Weighted_Spread,na.rm=TRUE)+0.001,0.001)), xlim=c(-0.01,0.03),main = paste("Histogram of Volume Weighted Spread of ",filename,sep=" "))
    hist(Effective_Spread,c(seq(min(Effective_Spread,na.rm=TRUE)-0.001,max(Effective_Spread,na.rm=TRUE)+0.001,0.001)), xlim=c(0,0.035),main = paste("Histogram of Efficient Spread of ",filename,sep=" "))
 
}
dev.off()

#from the plots we can see that for each kind of spreads, they has similar shape of distribution on different days
#the time weigthed spread are right-skewed, the volumn weighted spread looks similar to Normal distribution, and 
#the effecient spread are left skewed. I don't why the three kinds shows such different patterns

######################################################
#we test whether volumn weighted spreads is a normal distributon
shapiro.test(Volume_Weighted_Spread); #错误于shapiro.test(Volume_Weighted_Spread) : 样本大小必需在3和5000之间
qqnorm(Volume_Weighted_Spread, main = paste("QQ-Plot of Volume Weighted Spread of ",filename,sep=" "));
qqline(Volume_Weighted_Spread, col = 2)
#the qq-plot shows that the daily Volume_Weighted_Spread of the whole market is not a normal distribution

######################################################
#for the t-test, first we try the volumn weighted spreads since they're most similar to the normal distribution
filename = filenames[1]
mydata <- read.csv(file=paste("D:/MSFE/Practicum/data/",filename,".csv",sep=""), header = TRUE, sep = ",", quote = "\"", dec = ".", fill = TRUE)
a = mydata[,16]
filename = filenames[2]
mydata <- read.csv(file=paste("D:/MSFE/Practicum/data/",filename,".csv",sep=""), header = TRUE, sep = ",", quote = "\"", dec = ".", fill = TRUE)
b = mydata[,16]
t = t.test(a, b,alternative = "two.sided", paired = FALSE, var.equal = FALSE)
t

#whihch shows that their means are not equal！But there are numerous outside variables accounting for this, how do we
#account for the difference?

#######################################################
#draw 5-days volume weighted spread of APPL, also do Shapiro Test for each day to see wheather it's normal distribution
setwd("D:/MSFE/Practicum/data");
filenames = c("20130401","20130402","20130403","20130404","20130405")
ndays = 5
testresult <- array("list",c(ndays,4))
pdf("volumn weighted spreads of APPL of each day.pdf")
for(i in 1:ndays)
{
    filename = filenames[i]
    mydata <- read.csv(file=paste("D:/MSFE/Practicum/data/",filename,".csv",sep=""), header = TRUE, sep = ",", quote = "\"", dec = ".", fill = TRUE)
    newdata<- data.frame(mydata[ which(mydata$Symbol_t =='AAPL'),16])
    testresult[i,] = shapiro.test(newdata[,1])
    hist(newdata[,1],c(seq(min(newdata,na.rm=TRUE)-0.002,max(newdata,na.rm=TRUE)+0.002,0.002)), xlim=c(-0.05,0.3),main = paste("Hist of Volume Weighted Spread of APPL on",filenames[i],sep=" ")) 
}
dev.off()
#and 

#######################################################
#t-test for two day's volume weighted spread of APPL
filename = filenames[1]
mydata <- read.csv(file=paste("D:/MSFE/Practicum/data/",filename,".csv",sep=""), header = TRUE, sep = ",", quote = "\"", dec = ".", fill = TRUE)
newdata<- mydata[ which(mydata$Symbol_t =='AAPL'),]
a = data.frame(newdata[,16])#must use data.frame to form a column
filename = filenames[2]
mydata <- read.csv(file=paste("D:/MSFE/Practicum/data/",filename,".csv",sep=""), header = TRUE, sep = ",", quote = "\"", dec = ".", fill = TRUE)
newdata<- mydata[ which(mydata$Symbol_t =='AAPL'),]
b = data.frame(newdata[,16])
pdf("volumn weighted spread for APPL.pdf")
hist(a[,1],c(seq(min(a,na.rm=TRUE)-0.002,max(a,na.rm=TRUE)+0.002,0.002)), xlim=c(-0.05,0.3),xlab = "Volume Weighted Spread",main = paste("Hist of Volume Weighted Spread of APPL on",filenames[1],sep=" "))
hist(b[,1],c(seq(min(b,na.rm=TRUE)-0.002,max(b,na.rm=TRUE)+0.002,0.002)), xlim=c(-0.01,0.3),xlab = "Volume Weighted Spread",main = paste("Hist of Volume Weighted Spread of APPL on",filenames[2],sep=" "))
filename = filenames[3]
mydata <- read.csv(file=paste("D:/MSFE/Practicum/data/",filename,".csv",sep=""), header = TRUE, sep = ",", quote = "\"", dec = ".", fill = TRUE)
newdata<- mydata[ which(mydata$Symbol_t =='AAPL'),]
dev.off()
#a<-scale(a, center = TRUE, scale = TRUE)
#b<-scale(b, center = TRUE, scale = TRUE)
t = t.test(a, b,alternative = "two.sided", paired = FALSE, var.equal = FALSE)
t
#still different mean

########################################################
#test one day's APPLE


########################################################
#draw histogram of return
i=1;
hist(Return_,c(seq(min(Return_,na.rm=TRUE)-0.001,max(Return_,na.rm=TRUE)+0.001,0.0001)), xlim=c(-0.01,0.01),main = paste("Histogram of Return of ",filenames[i],sep=" "))

