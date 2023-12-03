SPEECH EMOTION RECOGNITION

Project purpose:
Discrete emotion classification of a speech.

All the parameters of audio, feature, model and training are exposed in their respective block in the file param.yaml and can be set from there.

The default parameters of the MFCC are set to parameters we got the best result by.
The default parameters of the Mel are set to the same value as the pretrained bert model.
(The pretrained bert model is trained on 80 dim mel with delta (total 160 feature dim))

After choosing feature in the file:
- run save_feature.py to save the desired features and create annotation file.
  If bert model is desired:
- git clone https://github.com/andi611/Mockingjay-Speech-Representation.git (to clone the acoustic bert model)

Put the scripts and the checkpoint file inside the cloned folder
- run train.py to start training 

The convolution attention model can be trained with either acoustic or bert features.
To train with the acoustic feature, it is enough to set the right annotation path for the feature in the training block.
To train with the bert feature, set 
bert_conv_att to True
fine_tune to True or False

