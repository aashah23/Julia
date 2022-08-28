#importing required packages
using SeisMain
using SeisPlot
using SeisProcessing
using Random, Distributions
using PyPlot
using SigFigs

#defining intial time (s) of section
ot=0.0;
#defining every dt (s)
dt=0.002;
#defining number of time samples 
nt=300;


#defining position now (m)
ox1=0.0;
#define space sampling (m)
dx1=10;
#define number of samples
nx1=50;


#the intercepts (tau)
tmax= (nt-1)*dt;
taumin= 0.1*tmax;
taumax = 0.9*tmax; 


#making a random (4x1) matrix for multiple tau values (c for convolutional)
r= rand(Uniform(0,1),4);
#expressing tau
tau = taumin .+ (taumax-taumin).*r;


#defining xmax
xmax = (nx1-1)*dx1;


#setting the pmin and pmax
pmax= (0.9*tmax.-tau)/xmax;
pmin = (-0.9*tau)/xmax;


#defining p 
p = pmin .+ (pmax.-pmin).*r;


#defining the amplitudes between (-1,1)
amp=rand(Uniform(-1,1),4);


#setting central frequency
fc=20.0;


#closing all previous figures
close("all")


#labelling the axes
labelx= "x (m)";
labely= "Time (s)";


#setting the number of training examples
train_cnexamples=200;

#define number of channels
nc=1;

#creating empty arrays for data
train_clabels=zeros(nt,nx1,nc,train_cnexamples); #for clean data
train_cdne=zeros(nt,nx1,nc,train_cnexamples); #for noisy data


#creating random noise for training data between (0.5,3)
train_crnoise= rand(Uniform(0.5,3),train_cnexamples);


#for loop
for i in 1:train_cnexamples
    train_cnevents= rand(1:train_cnexamples) #set number of random events
    r=rand(Uniform(0,1),train_cnevents); 
    tau = taumin .+ (taumax-taumin).*r;
    pmax= (0.9*tmax.-tau)/xmax;
    pmin = (-0.9*tau)/xmax;
    p = pmin .+ (pmax.-pmin).*r;
    sign = rand([1,-1],train_cnevents);
    np = sign .* p;

    p1=p;
    p2=p3=p4=zero(p1);
    amp=rand(Uniform(-1,1),train_cnevents);
    sign = rand([1,-1],train_cnevents);
    namp = sign .* amp;

    efc = rand(20:0.1:30);

    # modelling clean data
    train_clabels[:,:,:,i] = SeisLinearEvents(ot=ot,dt=dt,nt=nt,ox1=ox1,dx1=dx1,nx1=nx1,tau=tau,p1=np,p2=p2,p3=p3,p4=p4,amp=namp,f0=efc);

    # model noisy data (add noise to clean data dne = dce + noise)
    # snr = some_equation_that_adi_will_figure_=
    
    #adding noise to clean data and call it dne
    train_cdne[:,:,:,i] = SeisAddNoise(train_clabels[:,:,:,i],train_crnoise[i])
end


#closing all previous figures
close("all")


#setting name of plot
figname = "SeisLinConvEvents_plot";

#figure details 
w=20; #width
h=20; #height

#axis labels for figure
labelx = "Distance (m)";
labely = "Time (s)";
thetitle = "with noise";

#defining the figure
figure(figname,figsize=(w,h));


# MAKING PLOTS FOR CLEAN DATA
#j,k,l,m are random sample numbers for plotting random samples 
# (241) means for 2 rows and 4 columns, plot at index 1
j=rand(1:train_cnexamples);
subplot(241); 
SeisPlotTX(train_clabels[:,:,nc,j], fignum=figname, xlabel = labelx, ylabel=labely);

k = rand(1:train_cnexamples);
subplot(242);
SeisPlotTX(train_clabels[:,:,nc,k], fignum=figname, xlabel = labelx, ylabel=labely);

l = rand(1:train_cnexamples);
subplot(243);
SeisPlotTX(train_clabels[:,:,nc,l], fignum=figname, xlabel = labelx, ylabel=labely); 

m = rand(1:train_cnexamples);
subplot(244);
SeisPlotTX(train_clabels[:,:,nc,m],fignum=figname, xlabel = labelx, ylabel=labely);



# MAKING PLOTS FOR NOISY DATA|
subplot(245);
cSNR1 = round(train_crnoise[j],digits=3); #rounding random noise for random sample to 3 dp
SeisPlotTX(train_clabels[:,:,nc,j], fignum=figname, xlabel = labelx, ylabel=labely, title = "SNR="*"$(cSNR1)");

cSNR2 = round(train_crnoise[k],digits=3);
subplot(246);
SeisPlotTX(train_cdne[:,:,nc,k], fignum=figname, xlabel = labelx, ylabel=labely, title = "SNR="*"$(cSNR2)");

cSNR3 = round(train_crnoise[l],digits=3);
subplot(247);
SeisPlotTX(train_cdne[:,:,nc,l], fignum=figname, xlabel = labelx, ylabel=labely, title= "SNR = "*"$(cSNR3)");

cSNR4 = round(train_crnoise[m],digits=3);
subplot(248);
SeisPlotTX(train_cdne[:,:,nc,m], fignum=figname, xlabel = labelx, ylabel=labely, title= "SNR = "*"$(cSNR4)");


#tight_layout(); #auto-adjust height and width
gcf() #get the current figure


#EXPORTING DATA
#defining file for training data to be written on
file_to_write = "train_convnoisy.bin"

#creating the file for noisy data 
fid = open(file_to_write,"w");

#writing noisy data generated onto current file
write(fid,train_cdne);
#closing current file
close(fid)

#reading the noisy data
file_to_read = "train_convnoisy.bin"
#opening file to read
fid = open(file_to_read,"r");
train_Nc = nt*nx1*nc*train_cnexamples;

data1 = zeros(Float64,train_Nc);
read!(fid,data1);
close(fid)

#chekcing if read and write done correctly
Bc = reshape(data1,nt,nx1,nc,train_cnexamples);

#difference should be 0 for this to be true
diff = sum(abs.(train_cdne[:]-Bc[:]))

close("all")

file_to_write_2 = "train_convlabels.bin";
fid = open(file_to_write_2,"w");
write(fid,train_clabels);
close(fid)

file_to_read_2 = "train_convlabels.bin";
fid = open(file_to_read_2,"r");
trainl_Nc = nt*nx1*nc*train_cnexamples;

data2 = zeros(Float64,trainl_Nc);
read!(fid,data2);
close(fid)
B2c = reshape(data2,nt,nx1,nc,train_cnexamples);

#difference should be 0 for this to be true
diff = sum(abs.(train_clabels[:]-B2c[:]))