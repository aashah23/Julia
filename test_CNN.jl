# create container (con) for reading the data
cont_conv_test_noisy = zeros(nt,nx1,nc,test_cnexamples);
newdne2c = read!("test_convnoisy.bin",cont_conv_test_noisy);

cont_conv_test_labels = zeros(nt,nx1,nc,test_cnexamples);
newdce2c = read!("test_convlabels.bin",cont_conv_test_labels);



function get_test_conv_seismic_batches(;nw=50, nh=300,nc=1,batch_size=20,shuffle=true, snr=0.5)

    numb2c = test_cnexamples
    slice2c = newdne2c[:,:,:,1:numb2c]
    N5c = size(slice2c)[4];
    datalabel2c = newdce2c;
    SeisTest_conv_clean = newdce2c;

    SeisTest_conv = zeros(Float32,nw,nh,nc,N5c);
    SeisTest_conv = copy(slice2c)


    #adding noise
    test_convnoise = Float32.(snr .* randn(size(SeisTest_conv)));
    SeisTest_conv .+= test_convnoise


    datalabel2c = newdce2c;
    newdatalabel2c = reshape(datalabel2c,nh,nw,nc,test_cnexamples)

    #loading training and labels into data loader
    test_convloader = DataLoader((data=SeisTest_conv,label=newdatalabel2c),batchsize=batch_size, shuffle=shuffle)
    
    #return the loaders
    return SeisTest_conv, test_convloader
end


SeisTest_conv, test_convloader = get_test_conv_seismic_batches();


# define the test input
tr_xinconv = copy(SeisTest_conv);
# pass input to the autoencoder (AE) 
tr_xoutconv = CAE(tr_xinconv);
# safe-guard the size
@assert size(tr_xinconv) == size(tr_xoutconv)

# plot examples at i = 10,50,90
nx = 50;
tr_xinconv20 = reshape(tr_xinconv[:,20],(nt,nx)); 
tr_xoutconv20 = reshape(tr_xoutconv[:,20],(nt,nx));
tr_clean20 = reshape(newdce2[:,:,20], (nt,nx));

close("all")
SeisPlotTX([tr_clean20 tr_xinconv20 tr_xoutconv20],cmap="gray")
gcf()


#xin50 = reshape(xin[:,50],nt,nx); xout50 = reshape(xout[:,50],nt,nx);

#xin90 = reshape(xin[:,90],nt,nx); xout90 = reshape(xout[:,90],nt,nx);

tr_new_xinconv = reshape(tr_xinconv20,nt,nx);
tr_new_xoutconv = reshape(tr_xoutconv20,nt,nx);

#getting the amount of noise
tr_approx_noise_conv = tr_new_xinconv - tr_new_xoutconv;
close("all")


tr_approx_nothing_conv = tr_clean20 - tr_new_xoutconv;

SeisPlotTX([tr_clean20 tr_xinconv20 tr_xoutconv20 tr_approx_noise_conv tr_approx_nothing_conv],cmap="gray")
#PS: middle row is called Latent space = output of encoder. First row is input going into the encoder
# third row is 


gcf()