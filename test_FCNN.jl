# create container for reading the data
cont_full_test_noisy = zeros(nt,nx1,test_nexamples); 
newdne2 = read!("test_noisy.bin",cont_full_test_noisy); 

cont_full_test_labels= zeros(nt,nx1,test_nexamples); 
newdce2 = read!("test_labels.bin",cont_full_test_labels); 

datalabel2 = newdce2;



function get_test_full_seismic_batches(;nw=50, nh=300, nc=1, batch_size=20, shuffle=true,snr=0.05)

    numb2=test_nexamples;
    slice2 = newdne2[:,:,1:numb2];
    #reshaping the data 
    
    N5 = size(slice2)[3];

    SeisTest_full = zeros(Float64,nw*nh*nc,N5);
    
    for i in 1:N5
        SeisTest_full[:,i] = reshape(slice2[:,:,i],nw*nh*nc);
    end

    #adding noise
    noise_test = Float32.(snr .* randn(size(SeisTest_full)))
    SeisTest_full .+= noise_test

    #making labels same dim as SeisTrain_full
    datalabel2 = newdce2;
    newdatalabel2 = reshape(datalabel2,nh*nw,numb2)

    #loading training and labels into data loader
    test_loader = DataLoader((data=SeisTest_full,label= newdatalabel2),batchsize=batch_size, shuffle=shuffle)

    #return the loaders
    return SeisTest_full, test_loader
end


SeisTest_full, test_loader = get_test_full_seismic_batches();


# define the test input
tr_xin = copy(SeisTest_full);
# pass input to the autoencoder (AE) 
tr_xout = FCAE(tr_xin);
# safe-guard the size
@assert size(tr_xin) == size(tr_xout)

# plot examples at i = 10,50,90
nx = 50;
tr_xin20 = reshape(tr_xin[:,20],(nt,nx)); 
tr_xout20 = reshape(tr_xout[:,20],(nt,nx));
tr_clean20 = reshape(newdce2[:,:,20], (nt,nx));

close("all")
SeisPlotTX([tr_clean20 tr_xin20 tr_xout20],cmap="gray")
gcf()


#xin50 = reshape(xin[:,50],nt,nx); xout50 = reshape(xout[:,50],nt,nx);

#xin90 = reshape(xin[:,90],nt,nx); xout90 = reshape(xout[:,90],nt,nx);

tr_new_xin = reshape(tr_xin20,nt,nx);
tr_new_xout = reshape(tr_xout20,nt,nx);

#getting the amount of noise
tr_approx_noise = tr_new_xin - tr_new_xout;
close("all")


tr_approx_nothing = tr_clean20 - tr_new_xout;

SeisPlotTX([tr_clean20 tr_xin20 tr_xout20 tr_approx_noise tr_approx_nothing],cmap="gray")
#PS: middle row is called Latent space = output of encoder. First row is input going into the encoder
# third row is 


gcf()