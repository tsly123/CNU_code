### READ ME FIRST
This code is taken from https://github.com/tatsy/keras-generative with small modifications. Visit the github repo for more GAN models.
- The <b>Train_DCGAN.ipynb</b> notebook is for initiating the training <b>DCGAN</b> with <b>CelebA dataset</b> (image size 32x32).
- Download <b>Train_DCGAN.ipynb</b> and dataset <b>celebA_32.hdf5</b> from the [Google Drive](https://drive.google.com/open?id=1_WhJsXzsddysyzEuX8RFZqkyQqY5mPVD) to your Drive and run with Colab. If you have powerful PC, download folder <b>PC_code</b> (also in the Drive) and dataset to your machine and try. I also provide the <b>celebA_64.hdf5</b> (image size 32x32).
- Since the code will load all the dataset <b>celebA_32.hdf5</b> (600MB) or <b>celebA_64.hdf5</b> (2GB) into RAM during training, this could be a memory problem for your machine. One solution is to code your own on the fly dataset which will load only the batch of images each iteration.
- This is what i got after 20 epochs

Image size 64x64
![](DCGAN_epoch_0020_64.png?raw=true)

Image size 32x32
![](DCGAN_epoch_0020_32.png?raw=true)

### Some notes

You can change the zdim parameter which decide how big is the sampling vector for generator input. The higher 'zdim', the more detailed image is. Higher 'zdim' also come with longer converging time. For example, while the zdim = 100 will generate noise images for the first 160,000 samples (of total 202599), and start converging at the end of epoch 1, 'zdim = 50' shows face-like images at the first 40,000 samples.

#### Troubleshooting

When running the code, it will output as:

`Epoch #1 | 17920/202599 (8.85 %) | g_loss = 1.943672 | d_loss = 0.776618 | g_acc = 0.031250 | d_acc = 0.539062 | ETA: 14 min 12 sec`

where, `g_loss, d_loss, g_acc, d_acc` are generator and discriminator loss and accuracy, perspectively. While you should pay attention on the `g_loss, d_loss, d_acc`, the `g_acc` doesn't have much meaning.

There are chances that you will encounter some troubles where the `g_loss, d_loss, d_acc` <b>don't change over iterations</b>. The main reason for these failure cases is that your hyperparameters were initially bad randomized.

#### Solution

This code has been verified as working. Ideally, the `g_loss, d_loss, g_acc, d_acc` would change over iterations and d_accwould be somewhere around 0.3-0.7 as below.

`Epoch #2 | 156992 / 202599 ( 77.49 %) | g_loss = 0.711317 | d_loss = 0.694376 | g_acc = 0.140625 | d_acc = 0.476562 | ETA: 3 min 11 sec`

The networks often converge after 1-2 first epochs, depending on how you set zdim. So if the numbers don't change or d_acc = 1 after 5 epochs, <b>just restart the code and run it again</b>.
Run the code a few times and you will get the hang of it.

### HAVE FUN
