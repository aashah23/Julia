#adding the required packages
using Flux, Flux.Optimise, Flux.Data
using Flux: mse
using Flux: @epochs
using Flux: params
using Plots, LinearAlgebra
using MLDatasets:MNIST
using Images.ImageCore
#using PyPlot
using ImageTransformations
using Plots

# create container for reading the data
container1= zeros(300,nx1,train_nexamples);
newdne1 = read!("train_noisy.bin",container1);

container2= zeros(300,nx1,train_nexamples);
newdce1 = read!("train_labels.bin",container2);

datalabel1 = newdce1;

#defining function to get a Float32 array
get_array(x) = Float32.(channelview(x)); 

#defining function to get training batches for FCNN
function get_train_full_seismic_batches(;nw=50, nh=300, nc=1, batch_size=20, shuffle=true,snr=0.05)

    numb1=train_nexamples;
    slice = newdne1[:,:,1:numb1];
    #reshaping the data 
    
    N = size(slice)[3];

    SeisTrain_full = zeros(Float64,nw*nh*nc,N);
    
    for i in 1:N
        SeisTrain_full[:,i] = reshape(slice[:,:,i],nw*nh*nc);
    end

    #adding noise
    train_noise = Float32.(snr .* randn(size(SeisTrain_full)))
    SeisTrain_full .+= train_noise

    #making labels same dim as SeisTrain_full
    datalabel1 = newdce1;
    newdatalabel1 = reshape(datalabel1,nh*nw,numb1)

    #loading training and labels into data loader
    train_loader = DataLoader((data=SeisTrain_full,label= newdatalabel1),batchsize=batch_size, shuffle=shuffle)

    #return the loaders
    return SeisTrain_full, train_loader
end


#function to show FC autoencoder
function show_fcautoencoder(input, lat_size,out_size,encoder,decoder;indexes=[3,4,5],nrow=3,ncol=3, figname="fcautoencoder_figure", fig_size=(8,8))

    a1 = 1.0;
    a2 = 1;
    #the latent space
    lat_space = encoder(input);
    lat_reshape = reshape(lat_space, lat_size...,:)

    #reconstruction
    out = decoder(lat_space)
    out_reshape = reshape(out, out_size...,:) 

    #reshaping input
    in_reshape = reshape(input, out_size..., :)

    #number of Plots
    nfigs = nrow*ncol

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


 
##now to train the data ##
SeisTrain_full, train_loader = get_train_full_seismic_batches();

#MAKING THE NEURAL NETWORK
#defining the encoding layer
encoder = Chain(
    Dense(300*50,512,relu),
    Dense(512,64,relu),
    Dense(64,32,relu),
    Dense(32,16,relu),
    Dense(16,8,relu))


#defining decoding layer
decoder = Chain(
    Dense(8,16,relu),
    Dense(16,32,relu),
    Dense(32,64,relu),
    Dense(64,512,relu),
    Dense(512,300*50,tanh))

#defining the FCAE (fully connected autoencoder)
FCAE = Chain(encoder,decoder);

close("all)")

#plotting something 
show_fcautoencoder(SeisTrain_full,
                (4,4),(300,50),
                encoder,decoder,
                indexes=[10,11,12])

#tight_layout()
gcf()


#defining the cost function
#cost(x) = Flux.mse(AE(x),x); #x is the input, and AE(x) is the output
cost(x,y) = Flux.mse(FCAE(x),y);



#step-size
#ss = 0.001;

opt = Adam(0.001, (0.9, 0.8))

#epochs
#trainmode!(m,true)
epochs = 100;

#creating empty arrays for losses and mean losses
trainlosses=[];
testlosses=[];

mean_trlosses=[];
mean_tslosses = [];


for epoch in 1:epochs
    println("Epoch: ",epoch)
    println("For training")
    #loop over batches
    #TRAINING
    train_loss = 0;
    for (i,(newdne1,newdce1)) in enumerate(train_loader)
        #get gradients
        grads = gradient(params(FCAE)) do 
        tr_l = cost(newdne1,newdce1)
        @show tr_l
        train_loss = copy(tr_l);   
       # tmp += l;
        #losses = [loss];
    end
       push!(trainlosses,train_loss)
      # @show tmp
      # push!(mean_losses,tmp)
       # println(losses[i])
        #gradient step
        update!(opt,params(FCAE),grads)
    end

    #TESTING
    println("For testing")
    test_loss = 0;
    for (i,(newdne2,newdce2)) in enumerate(test_loader)
        te_l = cost(newdne2,newdce2)
        @show te_l
        test_loss = copy(te_l);
        push!(testlosses,test_loss)
    end
    

    
end 

close("all")


#plotting the training loss
PyPlot.plot(trainlosses)#, xlabel = "Number of epochs", ylabel = "Error Loss");
PyPlot.plot(testlosses)#,  xlabel = "Number of epochs", ylabel = "Error Loss");
gcf()


#plotting the testing loss


close("all")


#making the mean training loss
mean_trainlosses = [];
for i = 1:epochs
        first = 1 + (i-1)*5
        last = 5 + (i-1)*5
        push!(mean_trainlosses,mean(trainlosses[first:last]) )
end

#plotting mean training loss
PyPlot.plot(mean_trainlosses)

#making the mean testing loss
mean_testlosses = [];
for i = 1:epochs
        first1 = 1 + (i-1)*2 #3 stands for the batch size
        last1 = 2 + (i-1)*2
        push!(mean_testlosses,mean(testlosses[first1:last1]) )
end

#plotting the mean testing loss
PyPlot.plot(mean_testlosses)
gcf()

close("all")


show_fcautoencoder(SeisTrain_full,
                (4,4),(300,50),
                encoder,decoder;
                indexes=[10,11,12], figname="same_actv_before_training")
            


                
                
               
                
                gcf()

# define the input
xin = copy(SeisTrain_full);
# pass input to the autoencoder 
xout = FCAE(xin);
# safe-guard the size
@assert size(xin) == size(xout)

# plot examples at i = 10,50,90
nx = 50;
section = 50;
xin10 = reshape(xin[:,section],(nt,nx)); 
xout10 = reshape(xout[:,section],(nt,nx));
clean10 = reshape(newdce1[:,:,section], (nt,nx));

close("all")
SeisPlotTX([clean10 xin10 xout10],cmap="gray",xlabel=labelx,ylabel=labely,title = "clean              noisy          denoised")
gcf()

#getting the amount of noise
approx_noise = xin10 - xout10;
close("all")


approx_nothing = clean10 - xout10;

SeisPlotTX([clean10 xin10 xout10 approx_noise approx_nothing],cmap="gray")
#PS: middle row is called Latent space = output of encoder. First row is input going into the encoder
# third row is 


gcf()

#pyplot(epochs,l, c=:black, lw=2);
#yaxis!("Loss", :log);
#xaxis!("Training epoch")


#finding max value for train loss
maxtrl = max(trainlosses...)

maxtel = max(testlosses...)

normtrainloss = trainlosses./maxtrl

normtestloss = testlosses./maxtel

close("all")
PyPlot.plot(normtrainloss)
PyPlot.plot(normtestloss)
gcf()






close("all")


#making the mean training loss
mean_normtrainloss = [];
for i = 1:epochs
        first = 1 + (i-1)*5
        last = 5 + (i-1)*5
        push!(mean_normtrainloss,mean(normtrainloss[first:last]) )
end

#plotting mean training loss
PyPlot.plot(mean_normtrainloss)
#gcf()


#making the mean testing loss
mean_normtestloss = [];
for i = 1:epochs
        first1 = 1 + (i-1)*2
        last1 = 2 + (i-1)*2
        push!(mean_normtestloss,mean(normtestloss[first1:last1]) )
end

#close("all")
#plotting the mean testing loss
PyPlot.plot(mean_normtestloss)
gcf()
