#adding the required packages
using Flux, Flux.Optimise, Flux.Data
using Flux: mse
using Flux: @epochs
using Flux: params
using Plots, LinearAlgebra
using Images.ImageCore
using ImageTransformations
using Plots

#creating empty containers for reading the clean and noisy training data
cont_full_train_noisy = zeros(300,nx1,train_nexamples);
newdne1 = read!("train_noisy.bin",cont_full_train_noisy);

cont_full_train_labels = zeros(300,nx1,train_nexamples);
newdce1 = read!("train_labels.bin",cont_full_train_labels);

#assigning clean data to array newdce1
datalabel1 = newdce1;

#defining function to get a Float32 array
get_array(x) = Float32.(channelview(x)); 

#defining function to get training batches for FCNN
function get_train_full_seismic_batches(;nw=50, nh=300, nc=1, batch_size=20, shuffle=true,snr=0.05)

    slice = newdne1[:,:,1:train_nexamples];
    N = train_nexamples; #simplify train_nexamples parameter 
    #creating empty array for FCNN noisy training data
    SeisTrain_full = zeros(Float64,nw*nh*nc,N);
    #making for loop to reshape noisy training data array into a 2D format for plotting
    for i in 1:N
        SeisTrain_full[:,i] = reshape(slice[:,:,i],nw*nh*nc);
    end

    #adding noise
    train_noise = Float32.(snr .* randn(size(SeisTrain_full)))
    SeisTrain_full .+= train_noise

    #making labels same dim as SeisTrain_full (2D)
    datalabel1 = newdce1;
    newdatalabel1 = reshape(datalabel1,nh*nw,train_nexamples)

    #loading training noise and labels into data loader
    train_loader = DataLoader((data=SeisTrain_full,label= newdatalabel1),batchsize=batch_size, shuffle=shuffle)

    #return the loaders
    return SeisTrain_full, train_loader
end


#function to show FCNN autoencoder
function show_fcautoencoder(input, lat_size,out_size,encoder,decoder;indexes=[10,11,12],nrow=3,ncol=3, figname="fcautoencoder_figure", fig_size=(8,8))
    #setting the colour scales for the plot (they are different)
    a1 = 1.0;
    a2 = 1.0; #0.001 before training, 1.0 after training

    #defining the latent space (middle row)
    lat_space = encoder(input); #takes input from encoder
    lat_reshape = reshape(lat_space, lat_size...,:) #reshapes to custom latent space size

    #reconstruction
    out = decoder(lat_space) #runs latent space through decoder
    out_reshape = reshape(out, out_size...,:) #reshapes to produce a particular size

    #reshaping input
    in_reshape = reshape(input, out_size..., :) #reshapes size from decoder of input  

    #defining the figure
    figure(figname2, figsize=fig_size)

    #plotting the different figures together
    for j in 1:ncol
        #plotting the noisy input data
        subplot(nrow,ncol,j);
        imshow(in_reshape[:,:,indexes[j]],aspect="auto", cmap = "RdBu", vmin=-a1,vmax=a1)

        #plotting the latent space taken through encoder
        subplot(nrow,ncol,j+3);
        imshow(lat_reshape[:,:,indexes[j]], cmap = "RdBu")
        
        #plots the linear events before and after training (before and after epochs run) 
        subplot(nrow,ncol,j+6)
        imshow(out_reshape[:,:,indexes[j]],aspect="auto", cmap = "RdBu", vmin=-a2,vmax=a2)
    
    end
end

#setting name of training plot(s)
figname1 = "FCNN_before_training";
figname2 = "FCNN_after_training";
 
##TRAINING THE DATA ##
SeisTrain_full, train_loader = get_train_full_seismic_batches(); #calling fucntion to get the batches

#MAKING THE NEURAL NETWORK
#defining the encoding layer
encoder = Chain(
    Dense(300*50,64,relu),
    Dense(64,32,relu),
    Dense(32,8,relu))
#    Dense(32,16,relu),
  #  Dense(16,8,relu))


#defining the decoding layer
decoder = Chain(
    Dense(8,32,relu),
 #   Dense(16,32,relu),
    #Dense(32,32,relu),
    Dense(32,64,relu),
    Dense(64,300*50,identity))

#defining the FCAE (fully connected autoencoder)
FCAE = Chain(encoder,decoder);

close("all)")

#plotting something at index[i,j,k]
show_fcautoencoder(SeisTrain_full,
                (4,4),(300,50),
                encoder,decoder,
                indexes=[10,11,12])

tight_layout()
gcf()

#saving the figure we just plotted (switch between these two)
savefig("FCNN_before_training")
#savefig("FCNN_after_training")

#defining the cost function
#x is the input, AE(x) is the output, y are the labels
cost(x,y) = Flux.mse(FCAE(x),y);

#defijing the optimizer
opt = Adam(0.001, (0.9, 0.8))

#setting number of epochs
epochs = 100;

#creating empty arrays for losses and mean losses
trainlosses=[];
testlosses=[];

mean_trlosses=[];
mean_tslosses = [];

#for loop for raw data
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
        @show tr_l #show training loss
        train_loss = copy(tr_l); #copy training loss values 
    end
       push!(trainlosses,train_loss) #push values into array
        #gradient step
        update!(opt,params(FCAE),grads)
    end

    #TESTING
    println("For testing")
    test_loss = 0;
    for (i,(newdne2,newdce2)) in enumerate(test_loader)
        te_l = cost(newdne2,newdce2)
        @show te_l #show testing loss
        test_loss = copy(te_l); #copy testing loss values 
        push!(testlosses,test_loss) #push values into array
    end
    

    
end 

close("all")

#plotting training and testing loss for FCNN
figname3 = "FCNN_Loss_Vs_Batches";
PyPlot.plot(trainlosses, label="Train Loss")
PyPlot.plot(testlosses, label="Test Loss")
PyPlot.xlabel("Number of batches")
PyPlot.ylabel("Loss")
PyPlot.title("FCNN Losses vs Number of Batches")
PyPlot.legend(loc=1)
gcf()
#savefig("FCNN_Loss_Vs_Batches")

close("all")

#making the mean training loss using arithmetic series
mean_trainlosses = [];
for i = 1:epochs
        first = 1 + (i-1)*5
        last = 5 + (i-1)*5
        push!(mean_trainlosses,mean(trainlosses[first:last]) )
end

#making the mean testing loss using arithmetic series
mean_testlosses = [];
for i = 1:epochs
        first1 = 1 + (i-1)*2
        last1 = 2 + (i-1)*2
        push!(mean_testlosses,mean(testlosses[first1:last1]) )
end


#plotting the mean training and testing FCNN losses
figname4 = "Mean_FCNN_Loss_Vs_Batches";
PyPlot.plot(mean_trainlosses, label="Mean Train Loss");
PyPlot.plot(mean_testlosses, label="Mean Test Loss");
PyPlot.xlabel("Number of epochs");
PyPlot.ylabel("Mean Loss");
PyPlot.legend(loc=1);
PyPlot.title("Mean FCNN Losses vs Number of Epochs");
gcf()
#savefig("Mean_FCNN_Loss_Vs_Epochs")
close("all")
# NORMALIZING THE MEAN LOSSES
#finding max value for train and test loss
maxtrl_full = max(mean_trainlosses...)
maxtel_full = max(mean_testlosses...)

#normalizing data via max(loss) values 
norm_maxtrainloss = mean_trainlosses./maxtrl_full
norm_maxtestloss = mean_testlosses./maxtel_full

close("all")

#plotting normalized (by max) loss values
figname5 = "NormMean_FCNN_Loss_Vs_Batches";
PyPlot.xlabel("Number of epochs");
PyPlot.ylabel("Normalized Mean Losses");
PyPlot.plot(norm_maxtrainloss, label="Norm Mean Train Loss");
PyPlot.plot(norm_maxtestloss, label="Norm Mean Test Loss");
PyPlot.title("Normalized Mean FCNN Losses vs Number of Epochs");
PyPlot.legend(loc=5);
gcf()
#savefig("Mean_FCNN_Loss_Vs_Epochs")

close("all")



#NOW, TO TEST THE FCNN BY PLOTTING INDIVIDUAL RESULTS

xin = copy(SeisTrain_full);
# pass input to the autoencoder 
xout = FCAE(xin);
# safe-guard the size
@assert size(xin) == size(xout)

#axis labels
labelx = "Distance (m)"
labely = "Time (s)"

nx = 50;

# plot examples at particular section (example number) between 1:(no of examples)
section = 100;
xin_sec = reshape(xin[:,section],(nt,nx)); 
xout_sec = reshape(xout[:,section],(nt,nx));
clean_sec = reshape(newdce1[:,:,section], (nt,nx));

close("all")

#plotting clean, noisy and denoised data side-by-side
figname5 = "Train_Clean_Noisy_Denoised";
SeisPlotTX([clean_sec xin_sec xout_sec],cmap="gray",xlabel=labelx,ylabel=labely,title = "clean              noisy          denoised")
gcf()
#savefig("Train_Clean_Noisy_Denoised")

#getting the approx amount of noise
approx_noise = xin_sec - xout_sec;
#next formula should give us ~nothing (tells us how close our model is to real event)
approx_nothing = clean_sec - xout_sec;

close("all")

#plotting them all 
figname6 = "Train_5_in_1";
SeisPlotTX([clean_sec xin_sec xout_sec approx_noise approx_nothing],titlesize=15.5,cmap="gray",xlabel=labelx,ylabel=labely,title = "   clean   noisy  denoised ~noise ~nothing")
gcf()
#savefig("Train_5_in_1")