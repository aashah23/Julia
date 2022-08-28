#defining the new number of examples for testing dataset
test_nexamples = 20;

#defining the testing dataset
#making array for labels
test_labels = zeros(nt,nx1,test_nexamples);
#making an empty array to fill in
test_dne = zeros(nt,nx1,test_nexamples);


#defining noise between 0.5 and 3 for testing data
test_rnoise= rand(Uniform(0.5,3),test_nexamples);

#for loop for the testing data
for i in 1:test_nexamples
    test_nevents= rand(1:test_nexamples)
    r=rand(Uniform(0,1),test_nevents);
    tau = taumin .+ (taumax-taumin).*r;
    pmax= (0.9*tmax.-tau)/xmax;
    pmin = (-0.9*tau)/xmax;
    p = pmin .+ (pmax.-pmin).*r;
    sign = rand([1,-1],test_nevents);
    np = sign .* p;

    p1=p;
    p2=p3=p4=zero(p1);
    amp=rand(Uniform(-1,1),test_nevents);
    sign = rand([1,-1],test_nevents);
    namp = sign .* amp;

    efc = rand(20:0.1:30);

    # model clean data
    test_labels[:,:,i] = SeisLinearEvents(ot=ot,dt=dt,nt=nt,ox1=ox1,dx1=dx1,nx1=nx1,tau=tau,p1=np,p2=p2,p3=p3,p4=p4,amp=namp,f0=efc);

    # model noisy data (add noise to clean data dne = dce + noise)
    # snr = some_equation_that_adi_will_figure_=
    
    #adding noise to dce
    test_dne[:,:,i] = SeisAddNoise(test_labels[:,:,i],test_rnoise[i])
end

close("all")

figname = "adi_plot2";

#figure details 
w=20; 
h=20;
labelx = "Distance (m)";
labely = "Time (s)";
thetitle = "with noise";


figure(figname,figsize=(w,h));

j=rand(1:test_nexamples);
subplot(241);
SeisPlotTX(test_labels[:,:,j], fignum=figname, xlabel = labelx, ylabel=labely);

k = rand(1:test_nexamples);
subplot(242);
SeisPlotTX(test_labels[:,:,k], fignum=figname, xlabel = labelx, ylabel=labely);

l = rand(1:test_nexamples);
subplot(243);
SeisPlotTX(test_labels[:,:,l], fignum=figname, xlabel = labelx, ylabel=labely); 

m = rand(1:test_nexamples);
subplot(244);
SeisPlotTX(test_labels[:,:,m],fignum=figname, xlabel = labelx, ylabel=labely);


subplot(245);
SNR1 = round(test_rnoise[j],digits=3);
SeisPlotTX(test_labels[:,:,j], fignum=figname, xlabel = labelx, ylabel=labely, title = "SNR="*"$(SNR1)");

SNR2 = round(test_rnoise[k],digits=3);
subplot(246);
SeisPlotTX(test_dne[:,:,k], fignum=figname, xlabel = labelx, ylabel=labely, title = "SNR="*"$(SNR2)");

SNR3 = round(test_rnoise[l],digits=3);
subplot(247);
SeisPlotTX(test_dne[:,:,l], fignum=figname, xlabel = labelx, ylabel=labely, title= "SNR = "*"$(SNR3)");

SNR4 = round(test_rnoise[m],digits=3);
subplot(248);
SeisPlotTX(test_dne[:,:,m], fignum=figname, xlabel = labelx, ylabel=labely, title= "SNR = "*"$(SNR4)");


#tight_layout()
gcf() 


#now to export the data into a binary file
#defining file to be written
file_to_write = "test_noisy.bin"

fid = open(file_to_write,"w");

write(fid,test_dne);
close(fid)

#reading the data
file_to_read = "test_noisy.bin"
fid = open(file_to_read,"r");

test_N = nt*nx1*test_nexamples;
data = zeros(Float64,test_N);
read!(fid,data);
close(fid)

#chekcing if read and write done correctly
B3 = reshape(data,nt,nx1,test_nexamples);

#difference should be 0 for this to be true
diff = sum(abs.(test_dne[:]-B3[:]))

close("all")

file_to_write_2 = "test_labels.bin";
fid = open(file_to_write_2,"w");
write(fid,test_labels);
close(fid)
file_to_read_2 = "test_labels.bin";
fid = open(file_to_read_2,"r");
testl_N = nt*nx1*test_nexamples;
data = zeros(Float64,testl_N);
read!(fid,data);
close(fid)
B4 = reshape(data,nt,nx1,test_nexamples);

#difference should be 0 for this to be true
diff = sum(abs.(test_labels[:]-B4[:]))