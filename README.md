# 7343 Homework 1

In this homework, your goal is to train a transformer model, called piano music composer, to generate piano music.  

## Data
The piano data (in midi format) can be downloaded from: 
https://storage.googleapis.com/magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0-midi.zip

By default, when unzipped, the data will be put into a directory named "maestro-v1.0.0".

The file "midi2seq.py" contains a set of functions that help to process the midi data and convert the data to sequences of events.   
The file "model_base.py" contains the base classes that you should inherit when implementing the following model classes.

## Composer
Implement a class called "Composer". It should be a subclass of the class ComposerBase. You must use the exact class name.

In this class, you should implement a language model by combining PyTorch nn.TransformerEncoder and 1 layer of fully connected (as token predictor). As a starter, the TransformerEncoder module can have 6 encoder layers, each having 8 heads and using d_model=256, dim_ff=512. Later, you should explore different values for these model parameters and see if a better composer can be obtained. When the "compose" member function is called, it should return a sequence of events. Randomness is require in the implementation of the compose function such that each call to the function should generate a different sequence. The function "seq2piano" in "midi2seq.py" can be used to convert the sequence into a midi object, which can be written to a midi file and played on a computer. Train the language model (autoregression) using the downloaded piano plays.

## Submit your work
Develop and train your model so that your model can compose reasonable piano music pieces at least 10 seconds long. Put all your code in a single file named "hw1.py" (*you must use this file name*) and submit the file in moodle. We will test your implementation using code similar to the following:
    
    from hw1 import Composer
    piano_seq = torch.from_numpy(process_midi_seq())
    loader = DataLoader(TensorDataset(piano_seq), shuffle=True, batch_size=bsz, num_workers=4)
    
    cps = Composer()
    for i in range(epoch):
        for x in loader:
            cps.train(x[0].cuda(0).long())
            
    midi = cps.compose(100)
    midi = seq2piano(midi)
    midi.write('piano1.midi')

Note that the above code trains your model from scratch. In addition, you should provide trained weights for your model. We may create your models by calling the constructor with "load_trained=True". In this case, your class constructor should: 
 - Download the trained weights from your google drive. (Do not upload the weights to moodle. Instead, you should store them on google drive.)
 - Load the trained weights into the model class object.

For example, if we do: m = Composer(load_trained=True), m should be a Composer model with the trained weights loaded. We should be able to call m.compose without training it and obtain a piano sequence from the downloaded trained model. 
 
 
