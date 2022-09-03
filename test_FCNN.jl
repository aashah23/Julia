using Flux, Flux.Optimise, Flux.Data

#creating empty containers for reading the clean and noisy testing data
cont_full_test_noisy = zeros(nt,nx1,test_nexamples); 
newdne2 = read!("test_noisy.bin",cont_full_test_noisy); 

cont_full_test_labels= zeros(nt,nx1,test_nexamples); 
newdce2 = read!("test_labels.bin",cont_full_test_labels); 

#assigning clean data to array newdce2
datalabel2 = newdce2;

#defining function to get testing batches for FCNN
function get_test_full_seismic_batches(;nw=50, nh=300, nc=1, batch_size=20, shuffle=true,snr=0.05)

    slice2 = newdne2[:,:,1:test_nexamples];
    #reshaping the data 
    N5 = test_nexamples; #simplify test_nexamples parameter 

    #creating empty array for FCNN noisy testing data
    SeisTest_full = zeros(Float64,nw*nh*nc,N5);
    #for loop to reshape noisy testing data array into a 2D format for plotting
    for i in 1:N5
        SeisTest_full[:,i] = reshape(slice2[:,:,i],nw*nh*nc);
    end

    #adding noise
    noise_test = Float32.(snr .* randn(size(SeisTest_full)))
    SeisTest_full .+= noise_test

    #making labels same dim as SeisTrain_full (2D)
    datalabel2 = newdce2;
    newdatalabel2 = reshape(datalabel2,nh*nw,test_nexamples)

    #loading testing noise and labels into data loader
    test_loader = DataLoader((data=SeisTest_full,label= newdatalabel2),batchsize=batch_size, shuffle=shuffle)

    #return the loaders
    return SeisTest_full, test_loader
end

#calling function to return the loaders
SeisTest_full, test_loader = get_test_full_seismic_batches();



#NOW, TO TEST THE FCNN BY RUNNING TESTING DATA THROUGH IT

# define the testing data input
te_xin = copy(SeisTest_full);
# pass input to the autoencoder (AE) 
te_xout = FCAE(te_xin);
# safe-guard the size
@assert size(te_xin) == size(te_xout)

# plot examples at particular section (example number) between 1:(no of test examples)
section = 20;
te_xin_sec = reshape(te_xin[:,section],(nt,nx)); 
te_xout_sec = reshape(te_xout[:,section],(nt,nx));
te_clean_sec = reshape(newdce2[:,:,section], (nt,nx));

close("all")

te_new_xin = reshape(te_xin_sec,nt,nx);
te_new_xout = reshape(te_xout_sec,nt,nx);
SeisPlotTX([te_clean_sec te_new_xin te_new_xout],cmap="gray",xlabel=labelx,ylabel=labely,title = "clean              noisy          denoised")
#plotting clean, noisy and denoised data side-by-side
figname7 = "Test_Clean_Noisy_Denoised";
gcf()
#savefig("Test_Clean_Noisy_Denoised")
close("all")


#getting the amount of noise
#te_approx_noise = tr_new_xin - tr_new_xout;
te_approx_noise = te_xin_sec - te_xout_sec;

#next formula should give us ~nothing (tells us how close our model is to real event)
te_approx_nothing = te_clean_sec - te_xout_sec;

#plotting them all 
figname8 = "Test_5_in_1";
SeisPlotTX([te_clean_sec te_xin_sec te_xout_sec te_approx_noise te_approx_nothing],xlabel=labelx,ylabel=labely,cmap="gray",title = "   clean   noisy  denoised ~noise ~nothing")
gcf()
#savefig("Test_5_in_1")
