using Flux, Flux.Optimise, Flux.Data

# create container (con) for reading the data
cont_conv_test_noisy = zeros(nt,nx1,nc,test_cnexamples);
newdne2c = read!("test_convnoisy.bin",cont_conv_test_noisy);

cont_conv_test_labels = zeros(nt,nx1,nc,test_cnexamples);
newdce2c = read!("test_convlabels.bin",cont_conv_test_labels);



function get_test_conv_seismic_batches(;nw=50, nh=300,nc=1,batch_size=20,shuffle=true, snr=0.5)
    #reshaping the data 
    #test_cnexamples;
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




#NOW, TO TEST THE CNN BY RUNNING TESTING DATA THROUGH IT

# define the testing data input
te_xinconv = copy(SeisTest_conv);
# pass input to the autoencoder (AE) 
te_xoutconv = CAE(te_xinconv);
# safe-guard the size
@assert size(te_xinconv) == size(te_xoutconv)

# plot examples at particular section (example number) between 1:(no of test examples)
nx = 50;
sectionc = 40;

new_cleanc = newdce1c[:,:,1,sectionc];
newtr_xinc = tr_xinc[:,:,1,sectionc];
newtr_xoutc = tr_xoutc[:,:,1,sectionc];

close("all")

#plotting clean, noisy and denoised data side-by-side
SeisPlotTX([new_cleanc newtr_xinc newtr_xoutc],cmap="gray",xlabel=labelx,ylabel=labely,title = "clean              noisy          denoised")
gcf()


#getting the amount of noise
#te_approx_noise = tr_new_xin - tr_new_xout;
approx_noise = newtr_xinc - newtr_xoutc;

close("all")

#next formula should give us ~nothing (tells us how close our model is to real event)
approx_nothing = new_cleanc - newtr_xoutc;

SeisPlotTX([new_cleanc newtr_xinc newtr_xoutc approx_noise approx_nothing],xlabel=labelx,ylabel=labely,cmap="gray",title = "   clean   noisy  denoised ~noise ~nothing")
gcf()
#savefig("Test_Conv_5_in_1")