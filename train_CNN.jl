# create container (con) for reading the data
cont_conv_train_noisy = zeros(nt,nx1,nc,train_cnexamples);
newdne1c = read!("train_convnoisy.bin",cont_conv_train_noisy);

cont_conv_train_labels = zeros(nt,nx1,nc,train_cnexamples);
newdce1c = read!("train_convlabels.bin",cont_conv_train_labels);

datalabel1c = newdce1c;


#defining function to get training batches for CNN
function get_train_conv_seismic_batches(;nw=50, nh=300,nc=1,batch_size=20,shuffle=true, snr=0.5)
    #reshaping the data 
    numb1c = train_cnexamples
    slicec = newdne1c[:,:,:,1:numb1c]
    Nc = size(slicec)[4];
    
    SeisTrain_conv = zeros(Float32,nw,nh,nc,Nc);

    SeisTrain_conv = copy(slicec)

    SeisTrain_full_clean = newdce1c;


    #adding noise
    train_convnoise = Float32.(snr .* randn(size(SeisTrain_conv)));
    SeisTrain_conv .+= train_convnoise

    #making labels same dim as SeisTrain_conv
    datalabel1c = newdce1c;
    newdatalabel1c = reshape(datalabel1c,nh,nw,nc,numb1c)

    #loading training and labels into data loader
    train_convloader = DataLoader((data=SeisTrain_conv,label=newdatalabel1c,),batchsize=batch_size, shuffle=shuffle)
    
    #return the loaders
    return SeisTrain_conv, train_convloader
end




#defining convolutional autoencoder function
function show_convautoencoder(input, lat_size,out_size,ConvEncoder,ConvDecoder;indexes=[3,4,5],nrow=3,ncol=3, figname="convautoencoder_figure", fig_size=(8,8))

    a1 = 1.0;
    a2 = 0.0001;
    #the convolutional latent space
    lat_cspace = ConvEncoder(input);
    lat_creshape = reshape(lat_cspace, lat_size...,:)

    #reconstruction
    outc = ConvDecoder(lat_cspace)
    out_creshape = reshape(outc, out_size...,:) 

    #reshaping input
    in_creshape = reshape(input, out_size..., :)

    #number of Plots
    #nfigs = nrow*ncol

    #start figure
    figure(figname, figsize=fig_size)

    for j in 1:ncol
        subplot(nrow,ncol,j);
        imshow(in_creshape[:,:,indexes[j]],aspect="auto", cmap = "RdBu", vmin=-a1,vmax=a1)

        subplot(nrow,ncol,j+3);
        imshow(lat_creshape[:,:,indexes[j]], cmap = "RdBu")
        

        subplot(nrow,ncol,j+6)
        imshow(out_creshape[:,:,indexes[j]],aspect="auto", cmap = "RdBu", vmin=-a2,vmax=a2)
    
    end
end


SeisTrain_conv, train_convloader = get_train_conv_seismic_batches();



#MAKING THE CNN - input is (300x50x1x150)
#defining the convolutional encoder
ConvEncoder = Chain(
    Conv((6,6), 1=>4, stride = 2,leakyrelu), #(148, 23, 8, 150)
    Conv((6,5), 4=>8, stride = 2, leakyrelu), #(72, 10, 16, 150)
    Conv((4,4), 8=>8, stride = 2, leakyrelu), #(35, 4, 16, 150)
    x-> reshape(x,35*4*8,:), #(2240, 150)

    #defining the fully-connected encoder
    Dense(1120,5, relu), #/2

#output size is now (7, 150), which is the input size to decoder
)


ConvDecoder = Chain(
#defining the fully-connected decoder
    Dense(5,1120, leakyrelu),


    #defining the convolutional decoder
    x-> reshape(x,35,4,16,:), #(35, 4, 16, 150)
    ConvTranspose((4,4),8=>8, stride = 2, relu), #(72, 10, 16, 150)
    ConvTranspose((6,5),8=>4, stride = 2, relu), #(148, 23, 8, 150)
    ConvTranspose((6,6), 4=>1, stride = 2,identity), #(300, 50, 1, 150)
)

#combining the encoder and decoder
CAE = Chain(ConvEncoder,ConvDecoder);


close("all)")


#plotting something 
show_convautoencoder(SeisTrain_conv,
                (5,5),(300,50),
                ConvEncoder,ConvDecoder,
                indexes=[10,11,12])

#tight_layout()
gcf()


#defining cost function
cost(x,y) = Flux.mse(CAE(x),y); #x is the input, CAE(x) is the output, y is labels

#setting the optimizier
opt = Adam(0.001, (0.9, 0.8));

#epochs
epochs = 40;

#creating empty arrays for losses and mean losses
trainconvlosses=[];
testconvlosses=[];

mean_trainconvlosses=[];
mean_testconvlosses = [];


for epoch in 1:epochs
    println("Epoch: ",epoch)
    println("For training")
    #loop over batches
    #TRAINING
    train_convloss = 0;
    for (i,(newdne1c,newdce1c)) in enumerate(train_convloader)

        #println(newdne1c |> size)
        #println(newdce1c |> size)

        #get gradients
        grads = gradient(params(CAE)) do 
        tr_convl = cost(newdne1c,newdce1c)
        @show tr_convl
        train_convloss = copy(tr_convl);   
    end
       push!(trainconvlosses,train_convloss)
        #gradient step
        update!(opt,params(CAE),grads)
    end

    #TESTING
    println("For testing")
    test_convloss = 0;
    for (i,(newdne2c,newdce2c)) in enumerate(test_convloader)
        te_convl = cost(newdne2c,newdce2c)
        @show te_convl
        test_convloss = copy(te_convl);
        push!(testconvlosses,test_convloss)
    end
    

    
end 

close("all")


#plotting the training loss
PyPlot.plot(trainconvlosses)#, xlabel = "Number of epochs", ylabel = "Error Loss");
PyPlot.plot(testconvlosses)#,  xlabel = "Number of epochs", ylabel = "Error Loss");
gcf()


#plotting the testing loss


close("all")


#making the mean training loss
mean_trainconvlosses = [];
for i = 1:epochs
        first = 1 + (i-1)*5
        last = 5 + (i-1)*5
        push!(mean_trainconvlosses,mean(trainconvlosses[first:last]) )
end

#plotting mean training loss
PyPlot.plot(mean_trainconvlosses)
gcf()


#making the mean testing loss
mean_testconvlosses = [];
for i = 1:epochs
        first1 = 1 + (i-1)*2
        last1 = 2 + (i-1)*2
        push!(mean_testconvlosses,mean(testconvlosses[first1:last1]) )
end

close("all")
#plotting the mean testing loss
PyPlot.plot(mean_testconvlosses)
gcf()


show_convautoencoder(SeisTrain_conv,
                (4,4),(300,50),
                encoder,decoder;
                indexes=[10,11,12], figname="same_actv_before_training")
            


                
                
               
                
                gcf()




# define the input
tr_xinc = copy(SeisTrain_conv);
# pass input to the autoencoder 

tr_xoutc = CAE(tr_xinc);
# safe-guard the size
@assert size(tr_xinc) == size(tr_xoutc)

# plot examples at i = 10,50,90
nx = 50;
sectionc = 50;

new_cleanc = newdce1c[:,:,1,sectionc]


newtr_xinc = tr_xinc[:,:,1,sectionc]


newtr_xoutc = tr_xoutc[:,:,1,sectionc]




close("all")
SeisPlotTX([new_cleanc newtr_xinc newtr_xoutc],cmap="gray",xlabel=labelx,ylabel=labely,title = "clean              noisy          denoised")
gcf()


#xin50 = reshape(xin[:,50],nt,nx); xout50 = reshape(xout[:,50],nt,nx);

#xin90 = reshape(xin[:,90],nt,nx); xout90 = reshape(xout[:,90],nt,nx);

#new_xin = reshape(xin20,nt,nx);
#new_xout = reshape(xout20,nt,nx);

#getting the amount of noise
approx_noise = newtr_xinc - newtr_xoutc;
close("all")


approx_nothing = new_cleanc - newtr_xoutc;

SeisPlotTX([new_cleanc newtr_xinc newtr_xoutc approx_noise approx_nothing],cmap="gray")
#PS: middle row is called Latent space = output of encoder. First row is input going into the encoder
# third row is 


gcf()