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
#define space sampling
dx1=10;
#define number of samples
nx1=50;


#the intercepts (tau)
tmax= (nt-1)*dt;
taumin= 0.1*tmax;
taumax = 0.9*tmax; 


#making a random (4x1) matrix for multiple tau values
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
train_nexamples=200;


#creating empty arrays for data
train_labels=zeros(nt,nx1,train_nexamples); #for clean data
train_dne=zeros(nt,nx1,train_nexamples); #for noisy data


#creating random noise for training data between (0.5,3)
train_rnoise= rand(Uniform(0.5,3),train_nexamples);


#for loop
for i in 1:train_nexamples
    train_nevents= rand(1:train_nexamples) #set number of random events
    r=rand(Uniform(0,1),train_nevents); 
    tau = taumin .+ (taumax-taumin).*r;
    pmax= (0.9*tmax.-tau)/xmax;
    pmin = (-0.9*tau)/xmax;
    p = pmin .+ (pmax.-pmin).*r;
    sign = rand([1,-1],train_nevents);
    np = sign .* p;

    p1=p;
    p2=p3=p4=zero(p1);
    amp=rand(Uniform(-1,1),train_nevents);
    sign = rand([1,-1],train_nevents);
    namp = sign .* amp;

    efc = rand(20:0.1:30);

    # modelling clean data
    train_labels[:,:,i] = SeisLinearEvents(ot=ot,dt=dt,nt=nt,ox1=ox1,dx1=dx1,nx1=nx1,tau=tau,p1=np,p2=p2,p3=p3,p4=p4,amp=namp,f0=efc);

    # model noisy data (add noise to clean data dne = dce + noise)
    # snr = some_equation_that_adi_will_figure_=
    
    #adding noise to clean data and call it dne
    train_dne[:,:,i] = SeisAddNoise(train_labels[:,:,i],train_rnoise[i])
end


#closing all previous figures
close("all")


#setting name of plot
figname = "SeisLinEvents_plot";

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
j=rand(1:train_nexamples);
subplot(241); 
SeisPlotTX(train_labels[:,:,j], fignum=figname, xlabel = labelx, ylabel=labely);

k = rand(1:train_nexamples);
subplot(242);
SeisPlotTX(train_labels[:,:,k], fignum=figname, xlabel = labelx, ylabel=labely);

l = rand(1:train_nexamples);
subplot(243);
SeisPlotTX(train_labels[:,:,l], fignum=figname, xlabel = labelx, ylabel=labely); 

m = rand(1:train_nexamples);
subplot(244);
SeisPlotTX(train_labels[:,:,m],fignum=figname, xlabel = labelx, ylabel=labely);



# MAKING PLOTS FOR NOISY DATA
subplot(245);
SNR1 = round(train_rnoise[j],digits=3); #rounding random noise for random sample to 3 dp
SeisPlotTX(train_labels[:,:,j], fignum=figname, xlabel = labelx, ylabel=labely, title = "SNR="*"$(SNR1)");

SNR2 = round(train_rnoise[k],digits=3);
subplot(246);
SeisPlotTX(train_dne[:,:,k], fignum=figname, xlabel = labelx, ylabel=labely, title = "SNR="*"$(SNR2)");

SNR3 = round(train_rnoise[l],digits=3);
subplot(247);
SeisPlotTX(train_dne[:,:,l], fignum=figname, xlabel = labelx, ylabel=labely, title= "SNR = "*"$(SNR3)");

SNR4 = round(train_rnoise[m],digits=3);
subplot(248);
SeisPlotTX(train_dne[:,:,m], fignum=figname, xlabel = labelx, ylabel=labely, title= "SNR = "*"$(SNR4)");


#tight_layout(); #auto-adjust height and width
gcf() #get the current figure


#EXPORTING DATA
#defining file for training data to be written on
file_to_write = "train_noisy.bin"

#creating the file for noisy data 
fid = open(file_to_write,"w");

#writing noisy data generated onto current file
write(fid,train_dne);
#closing current file
close(fid)

#reading the noisy data
file_to_read = "train_noisy.bin"
#opening file to read
fid = open(file_to_read,"r");

train_N = nt*nx1*train_nexamples;
data = zeros(Float64,train_N);
read!(fid,data);
close(fid)

#chekcing if read and write done correctly
B = reshape(data,nt,nx1,train_nexamples);

#difference should be 0 for this to be true
diff = sum(abs.(train_dne[:]-B[:]))

close("all")

file_to_write_2 = "train_labels.bin";
fid = open(file_to_write_2,"w");
write(fid,train_labels);
close(fid)
file_to_read_2 = "train_labels.bin";
fid = open(file_to_read_2,"r");
trainl_N = nt*nx1*train_nexamples;
data = zeros(Float64,trainl_N);
read!(fid,data);
close(fid)
B2 = reshape(data,nt,nx1,train_nexamples);

#difference should be 0 for this to be true
diff = sum(abs.(train_labels[:]-B2[:]))