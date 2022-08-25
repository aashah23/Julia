#defining function to get training batches for CNN
function get_train_conv_seismic_batches(;nw=50, nh=300,nc=1,batch_size=20,shuffle=true, snr=0.5)
    #reshaping the data 
    numb1=train_nexamples;
    slice = newdne1[:,:,1:numb1];
    N = size(slice)[3];
    
    SeisTrain_conv = zeros(Float32,nw,nh,nc,N);
    SeisTrain_conv = copy(slice)
    SeisTrain_full_clean = newdce1;


    #adding noise
    train_convnoise = Float32.(snr .* randn(size(SeisTrain_conv)));
    SeisTrain_conv .+= train_convnoise

    #making labels same dim as SeisTrain_conv
    datalabel1 = newdce1;
    newdatalabel1 = reshape(datalabel1,nh*nw,numb1)

    #loading training and labels into data loader
    train_convloader = DataLoader((data=SeisTrain_conv,label=newdatalabel1,),batchsize=batch_size, shuffle=shuffle)
    
    #return the loaders
    return SeisTrain_conv, train_convloader
end




#defining convolutional autoencoder function
function show_convautoencoder(input, lat_size,out_size,ConvEncoder,ConvDecoder;indexes=[3,4,5],nrow=3,ncol=3, figname="convautoencoder_figure", fig_size=(8,8))

    a1 = 1.0;
    a2 = 0.001;
    #the latent space
    lat_space = ConvEncoder(input);
    lat_reshape = reshape(lat_space, lat_size...,:)

    #reconstruction
    out = ConvDecoder(lat_space)
    out_reshape = reshape(out, out_size...,:) 

    #reshaping input
    in_reshape = reshape(input, out_size..., :)

    #number of Plots
    #nfigs = nrow*ncol

    #start figure
    figure(figname, figsize=fig_size)

    for j in 1:ncol
        subplot(nrow,ncol,j);
        imshow(in_reshape[:,:,indexes[j]],aspect="auto", cmap = "RdBu", vmin=-a1,vmax=a1)

        subplot(nrow,ncol,j+3);
        imshow(lat_reshape[:,:,indexes[j]], cmap = "RdBu")
        

        subplot(nrow,ncol,j+6)
        imshow(out_reshape[:,:,indexes[j]],aspect="auto", cmap = "RdBu", vmin=-a2,vmax=a2)
    
    end
end


SeisTrain_conv, train_convloader = get_train_conv_seismic_batches();


#x=rand(Float32,300,50,1,20);

#MAKING THE CNN - input is (300x50x1x20)
#defining the convolutional encoder
ConvEncoder = Chain(

    Conv((6,6), 1=>8, stride = 2,relu), #(148, 23, 8, 20)
    Conv((6,5), 8=>16, stride = 2, relu), #(72, 10, 16, 20)
    Conv((4,4), 16=>16, stride = 2, relu), #(35, 4, 16, 20)
    x-> reshape(x,35*4*16,:), #(2240, 20)

    #defining the fully-connected encoder
    Dense(2240,1120), #/2
    Dense(1120,560), #/2
    Dense(560,280), #/2
    Dense(280,140), #/2
    Dense(140,70), #/2
    Dense(70,35), #/2
    Dense(35,7)  #/5
#output size is now (7, 20), which is the input size to decoder
)


ConvDecoder = Chain(
#defining the fully-connected decoder
    Dense(7,35),
    Dense(35,70),
    Dense(70,140),
    Dense(140,280),
    Dense(280,560),
    Dense(560,1120),
    Dense(1120,2240),

    #defining the convolutional decoder
    x-> reshape(x,35,4,16,:),
    ConvTranspose((4,4),16=>16, stride = 2, relu),
    ConvTranspose((6,5),16=>8, stride = 2, relu),
    ConvTranspose((6,6), 8=>1, stride = 2,relu)

)

#combining the encoder and decoder
CAE = Chain(ConvEncoder,ConvDecoder);


close("all)")


#plotting something 
show_convautoencoder(SeisTrain_conv,
                (4,4),(300,50),
                encoder,decoder,
                indexes=[10,11,12])

#tight_layout()
gcf()










# Load triaing labels and training images 
container1= zeros(301,nx1,train_nexamples);
newdne1 = read!("train_dne.bin",container1);

container2= zeros(301,nx1,train_nexamples);
newdce1 = read!("train_labels.bin",container2);







#defining cost function
cost(x,y) = Flux.mse(CAE(x),y); #x is the input, CAE(x) is the output, y is labels
#cost(x,y) = Flux.mse(ConvAE(x),y);

opt = Adam(0.001, (0.9, 0.8));

#epochs
epochs = 50;

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
    for (i,(newdne1,newdce1)) in enumerate(train_convloader)
        #get gradients
        grads = gradient(params(ConvAE)) do 
        tr_convl = cost(newdne1,newdce1)
        @show tr_convl
        train_convloss = copy(tr_convl);   
       # tmp += l;
        #losses = [loss];
    end
       push!(trainconvlosses,train_convloss)
      # @show tmp
      # push!(mean_losses,tmp)
       # println(losses[i])
        #gradient step
        update!(opt,params(ConvAE),grads)
    end

    #TESTING
    println("For testing")
    test_convloss = 0;
    for (i,(newdne2,newdce2)) in enumerate(test_convloader)
        te_l = cost(newdne2,newdce2)
        @show te_convl
        test_loss = copy(te_convl);
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
        first1 = 1 + (i-1)*3
        last1 = 3 + (i-1)*3
        push!(mean_testconvlosses,mean(testconvlosses[first1:last1]) )
end

close("all")
#plotting the mean testing loss
PyPlot.plot(mean_testconvlosses)
gcf()