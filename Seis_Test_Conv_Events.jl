#defining the new number of examples for testing dataset
test_cnexamples = 40;

#defining the testing dataset
#making array for labels
test_clabels = zeros(nt,nx1,nc,test_cnexamples);
#making an empty array to fill in
test_cdne = zeros(nt,nx1,nc,test_cnexamples);


#defining noise between 0.5 and 3 for testing data
test_crnoise= rand(Uniform(0.5,3),test_cnexamples);

#for loop for the testing data
for i in 1:test_cnexamples
    test_cnevents= rand(1:test_cnexamples)
    r=rand(Uniform(0,1),test_cnevents);
    tau = taumin .+ (taumax-taumin).*r;
    pmax= (0.9*tmax.-tau)/xmax;
    pmin = (-0.9*tau)/xmax;
    p = pmin .+ (pmax.-pmin).*r;
    sign = rand([1,-1],test_cnevents);
    np = sign .* p;

    p1=p;
    p2=p3=p4=zero(p1);
    amp=rand(Uniform(-1,1),test_cnevents);
    sign = rand([1,-1],test_cnevents);
    namp = sign .* amp;

    efc = rand(20:0.1:30);

    # model clean data
    test_clabels[:,:,:,i] = SeisLinearEvents(ot=ot,dt=dt,nt=nt,ox1=ox1,dx1=dx1,nx1=nx1,tau=tau,p1=np,p2=p2,p3=p3,p4=p4,amp=namp,f0=efc);

    # model noisy data (add noise to clean data dne = dce + noise)
    # snr = some_equation_that_adi_will_figure_=
    
    #adding noise to dce
    test_cdne[:,:,:,i] = SeisAddNoise(test_clabels[:,:,:,i],test_crnoise[i])
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

cj=rand(1:test_cnexamples);
subplot(241);
SeisPlotTX(test_clabels[:,:,nc,cj], fignum=figname, xlabel = labelx, ylabel=labely);

ck = rand(1:test_cnexamples);
subplot(242);
SeisPlotTX(test_clabels[:,:,nc,ck], fignum=figname, xlabel = labelx, ylabel=labely);

cl = rand(1:test_cnexamples);
subplot(243);
SeisPlotTX(test_clabels[:,:,nc,cl], fignum=figname, xlabel = labelx, ylabel=labely); 

cm = rand(1:test_cnexamples);
subplot(244);
SeisPlotTX(test_clabels[:,:,nc,cm],fignum=figname, xlabel = labelx, ylabel=labely);


subplot(245);
cSNR1 = round(test_crnoise[cj],digits=3);
SeisPlotTX(test_clabels[:,:,nc,cj], fignum=figname, xlabel = labelx, ylabel=labely, title = "SNR="*"$(cSNR1)");

cSNR2 = round(test_crnoise[ck],digits=3);
subplot(246);
SeisPlotTX(test_cdne[:,:,nc,ck], fignum=figname, xlabel = labelx, ylabel=labely, title = "SNR="*"$(cSNR2)");

cSNR3 = round(test_crnoise[cl],digits=3);
subplot(247);
SeisPlotTX(test_cdne[:,:,nc,cl], fignum=figname, xlabel = labelx, ylabel=labely, title= "SNR = "*"$(cSNR3)");

cSNR4 = round(test_crnoise[cm],digits=3);
subplot(248);
SeisPlotTX(test_cdne[:,:,nc,cm], fignum=figname, xlabel = labelx, ylabel=labely, title= "SNR = "*"$(cSNR4)");


#tight_layout()
gcf() 


#now to export the data into a binary file
#defining file to be written
file_to_write = "test_convnoisy.bin"

fid = open(file_to_write,"w");

write(fid,test_cdne);
close(fid)

#reading the data
file_to_read = "test_convnoisy.bin"
fid = open(file_to_read,"r");

test_Nc = nt*nx1*nc*test_cnexamples;
data = zeros(Float64,test_Nc);
read!(fid,data);
close(fid)

#chekcing if read and write done correctly
B3c = reshape(data,nt,nx1,nc,test_cnexamples);

#difference should be 0 for this to be true
diffc = sum(abs.(test_cdne[:]-B3c[:]))

close("all")

file_to_write_2 = "test_convlabels.bin";
fid = open(file_to_write_2,"w");
write(fid,test_clabels);
close(fid)
file_to_read_2 = "test_convlabels.bin";
fid = open(file_to_read_2,"r");

testl_Nc = nt*nx1*nc*test_cnexamples;
data = zeros(Float64,testl_Nc);
read!(fid,data);
close(fid)
B4c = reshape(data,nt,nx1,nc,test_cnexamples);

#difference should be 0 for this to be true
diffc = sum(abs.(test_clabels[:]-B4c[:]))