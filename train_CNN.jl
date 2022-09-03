#adding the required packages
using Flux, Flux.Optimise, Flux.Data
using Flux: mse
using Flux: @epochs
using Flux: params
using Plots, LinearAlgebra
using Images.ImageCore
using ImageTransformations
using Plots

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
function show_convautoencoder(input, lat_size,out_size,ConvEncoder,ConvDecoder;indexes=[1,3,5],figname="fcautoencoder_figure",fig_size=(8,8),nrow=3,ncol=3)
    a1 = 1.0;
    a2 = 1.0; #set to 0.0001 before training, and 1.0 after training
    #the convolutional latent space
    lat_cspace = ConvEncoder(input);
    lat_creshape = reshape(lat_cspace, lat_size...,:)

    #reconstruction
    outc = ConvDecoder(lat_cspace)
    out_creshape = reshape(outc, out_size...,:) 

    #reshaping input
    in_creshape = reshape(input, out_size..., :)

    #number of Plots
    nfigs = nrow*ncol

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


#setting name of training plot(s)
figname11 = "CNN_before_training";
figname12 = "CNN_after_training";

##TRAINING THE DATA ##
SeisTrain_conv, train_convloader = get_train_conv_seismic_batches();


#MAKING THE CNN - input is (300x50x1x150)
#defining the convolutional encoder
ConvEncoder = Chain(
    Conv((6,6), 1=>2, stride = 2,leakyrelu), #(148, 23, 2, 150)
    Conv((6,5), 2=>4, stride = 2, leakyrelu), #(72, 10, 4, 150)
    Conv((4,4), 4=>4, stride = 2, leakyrelu), #(35, 4, 4, 150)
    Conv((3,2), 4=>4, stride = 2, leakyrelu), #(17, 2, 4, 150)
    x-> reshape(x,17*2*4,:), #(136, 150)

    #defining the fully-connected encoder
    Dense(136,7,leakyrelu),
    #Dense(68,5,relu)
#output size is now (5, 150), which is the input size to decoder
)


ConvDecoder = Chain(
    #defining the fully-connected decoder
    #Dense(5,68,relu),
    Dense(7,136,leakyrelu),

    #defining the convolutional decoder
    x-> reshape(x,17,2,4,:), #(17, 2, 4, 150)
    ConvTranspose((3,2), 4=>4, stride = 2, leakyrelu), #(17, 2, 4, 150)
    ConvTranspose((4,4),4=>4, stride = 2, leakyrelu), #(35, 10, 4, 150)
    ConvTranspose((6,5),4=>2, stride = 2, leakyrelu), #(72, 23, 2, 150)
    ConvTranspose((6,6), 2=>1, stride = 2,identity), #(300, 50, 1, 150)
)

#combining the encoder and decoder
CAE = Chain(ConvEncoder,ConvDecoder);


close("all)")


#plotting something 
show_convautoencoder(SeisTrain_conv,
                (5,5),(300,50),
                ConvEncoder,ConvDecoder,
                indexes=[1,3,5])

#tight_layout()
gcf()
#savefig("CNN_before_training")
#savefig("CNN_after_training")

#defining cost function
cost(x,y) = Flux.mse(CAE(x),y); #x is the input, CAE(x) is the output, y is labels

#setting the optimizier
opt = Adam(0.001)#, (0.9, 0.8));

#epochs
epochs = 60;

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


#plotting the training and testing loss for CNN
figname13 = "CNN_Loss_Vs_Batches";
PyPlot.plot(trainconvlosses, label="Train Loss");
PyPlot.plot(testconvlosses, label="Test Loss");
PyPlot.xlabel("Number of batches");
PyPlot.ylabel("Loss");
PyPlot.title("CNN Losses vs Number of Batches");
PyPlot.legend(loc=1);
gcf()
#savefig("FCNN_Loss_Vs_Batches")

close("all")

#making the mean training loss
mean_trainconvlosses = [];
for i = 1:epochs
        first = 1 + (i-1)*5
        last = 5 + (i-1)*5
        push!(mean_trainconvlosses,mean(trainconvlosses[first:last]) )
end


#making the mean testing loss
mean_testconvlosses = [];
for i = 1:epochs
        first1 = 1 + (i-1)*2
        last1 = 2 + (i-1)*2
        push!(mean_testconvlosses,mean(testconvlosses[first1:last1]) )
end

#plotting the mean training testing CNN losses
figname14 = "Mean_CNN_Loss_Vs_Batches";
PyPlot.plot(mean_trainconvlosses, label="Mean Train Loss");
PyPlot.plot(mean_testconvlosses, label="Mean Test Loss");
PyPlot.xlabel("Number of epochs");
PyPlot.ylabel("Mean Loss");
PyPlot.legend(loc=1);
PyPlot.title("Mean CNN Losses vs Number of Epochs");
gcf()
#savefig("Mean_CNN_Loss_Vs_Batches")

# NORMALIZING THE MEAN LOSSES
#finding max value for train and test loss
max_mean_train_conv_loss = maximum(mean_trainconvlosses);
max_mean_test_conv_loss = max(mean_testconvlosses...);

#normalizing data via max(loss) values 
norm1 = mean_trainconvlosses./max_mean_train_conv_loss;
norm2 = mean_testconvlosses./max_mean_test_conv_loss;

close("all")

#plotting normalized (by max) loss values
figname15 = "NormMean_CNN_Loss_Vs_Batches";
PyPlot.xlabel("Number of epochs");
PyPlot.ylabel("Normalized Mean Losses");
PyPlot.plot(norm1, label="Norm Mean Train Loss");
PyPlot.plot(norm2, label="Norm Mean Test Loss");
PyPlot.title("Normalized Mean CNN Losses vs Number of Epochs");
PyPlot.legend(loc=3);
gcf()
#savefig("NormMean_CNN_Loss_Vs_Epochs")

close("all")


#NOW, TO TEST THE FCNN BY PLOTTING INDIVIDUAL RESULTS

# define the input
#tr_xinc = copy(SeisTrain_conv);
# pass input to the autoencoder 

tr_xoutc = CAE(tr_xinc);
# safe-guard the size
@assert size(tr_xinc) == size(tr_xoutc)

# plot examples at particular section (example number) between 1:(no of examples)
nx = 50;
sectionc = 20;

new_cleanc = newdce1c[:,:,1,sectionc];
newtr_xinc = tr_xinc[:,:,1,sectionc];
newtr_xoutc = tr_xoutc[:,:,1,sectionc];

close("all")

#plotting clean, noisy and denoised data side-by-side
figname16 = "Train_Conv_Clean_Noisy_Denoised"
SeisPlotTX([new_cleanc newtr_xinc newtr_xoutc],cmap="gray",xlabel=labelx,ylabel=labely,title = "clean              noisy          denoised")
gcf()
#savefig("Train_Conv_Clean_Noisy_Denoised")

#getting the amount of noise
approx_noise_conv = newtr_xinc - newtr_xoutc;

close("all")

#next formula should give us ~nothing (tells us how close our model is to real event)
approx_nothing_conv = new_cleanc - newtr_xoutc;

#plotting them all 
figname17 = "Train_Conv_5_in_1";
SeisPlotTX([new_cleanc newtr_xinc newtr_xoutc approx_noise_conv approx_noise_conv],cmap="gray",title = "   clean   noisy  denoised ~noise ~nothing")
gcf()
#savefig("Train_Conv_5_in_1")