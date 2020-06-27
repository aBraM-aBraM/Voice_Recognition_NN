# Voice_Recognition_NN

Maker: Eyal Abramovitch

VIDEO DOWNLOAD LINK - https://f2h.io/zmqnut1db0vu

This is a voice recognition done using Convultional Neural Network.
It can be used to take a wav. file and detect which number is said.


SETUP:
First of all you need to make sure you have all the used libraries.
Secondly you need to have the exact hierarchy of datasets;
each folder (train, valid, test) should have 3 folders and within
every one of those 9 folders with names '0' to '9'.

The dataset used is a batch of spectrogram images I've created from a wav. dataset of voices

In order to predict you need to take your wav. file run it through wav_to_png.py and
then put the spectrogram image inside one of the folders of test.

train + valid + test: https://drive.google.com/file/d/1eKASxlNP_82ZjLodH1GbkCXq_2VYP9xC/view?usp=sharing
