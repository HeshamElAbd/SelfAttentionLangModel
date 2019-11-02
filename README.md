# SelfAttentionLangModel
Language model based upon the Encoder units of the transformer. For Theortical back ground please refere to Attention is all you need 
paper @(https://arxiv.org/abs/1706.03762) and for detials regard the impelementation please refere to the source code here and to
google tutorial avilable at https://www.tensorflow.org/beta/tutorials/text/transformer. 

# Notes and Updates: 
## To DO: 
### Check the custom Training function with the Annotator and the Modeler 

## Current State: 
### The Modeler and Annotator Models are ready for deployment.

## Examples: 
from SelfAttentionLangModel.Models import EncoderModels

demoModel=EncoderModels.Modeler(embedding_dim=16,

                                         vocabulary_size=28,
                                         
                                         conditional_string_length=30,
                                         
                                         num_encoder_layer=6,
                                         
                                         num_heads=4,
                                         
                                         num_neuron_pointwise=32,
                                         rate=0.1,\n
                                         return_attent_weights=False
                                         )
                                         
demoModel is a keras model that can be used to be trained as usal using fit method or using a tf.GradientTape method along with custom training. 

