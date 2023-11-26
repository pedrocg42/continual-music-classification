# Embeddings Experiments
In this section, we use a music fundational model [MERT-95M](https://huggingface.co/m-a-p/MERT-v1-95M) to extract embeddings from different music datasets and for different downstream tasks. You can find the embeddings and their labels in the corresponding subdirectories. In the following sections we explain some information of the datasets used to give some context.

## GTZAN
The [GTZAN]{sdsd} dataset is normally used for music genre classification and it is formed by 1000 from 10 different genres:
- blues
- classical
- country
- disco
- hiphop
- jazz
- metal
- pop
- reggae
- rock

The dataset that we used is a curated version of the original dataset, this cause the total number of songs to drop to 930. 

### Class-Incremental Learning Scenarios
The CIL scenarios that we use with this datasets are:

```python
scenario1 = [
    ["metal", "pop"],
    ["country", "disco"],
    ["reggae", "rock"],
    ["blues", "classical"],
    ["hiphop", "jazz"],
]

scenario2 = [
    ["jazz", "rock"],
    ["classical", "hiphop"],
    ["reggae", "country"],
    ["metal", "blues"],
    ["pop", "disco"],
]

scenario3 = [
    ["hiphop", "metal"],
    ["reggae", "pop"],
    ["classical", "rock"],
    ["disco", "blues"],
    ["country", "jazz"],
]
```

## VocalSet-Singer
[VocalSet](http://ismir2018.ircam.fr/doc/pdfs/114_Paper.pdf) dataset used for singer recognition task. VocalSet is a singing voice dataset containing 10.1 hours
of recordings of professional singers demonstrating both
standard and extended vocal techniques in a variety of musical contexts.

### Class-Incremental Learning Scenarios
The CIL scenarios that we use with this datasets are:

```python
scenario1 = [
    ["female1", "female2", "male1", "male2"],
    ["female3", "female4", "male3", "male4"],
    ["female5", "female6", "male5", "male6"],
    ["female7", "female8", "male7", "male8"],
    ["female9", "male9", "male10", "male11"],
]

scenario2 = [
    ["female9", "male3", "female3", "female1"],
    ["female8", "male1", "male9", "female5"],
    ["female2", "male8", "female6", "male6"],
    ["male7", "male4", "male2", "female7"],
    ["male10", "female4", "male5", "male11"],
]

scenario3 = [
    ["female8", "male7", "male8", "female1"],
    ["male10", "female7", "male6", "male1"],
    ["female9", "female5", "male9", "female4"],
    ["female3", "male4", "male5", "female6"],
    ["male11", "female2", "male2", "male3"],
]
```

## VocalSet-Tech
In this case, we use VocalSet dataset too, but in this case we try to classify the technique that the singer is using instead of their identity. We follow the original work and choose the same subset of techniques from all the presented in their work.

### Class-Incremental Learning Scenarios
The CIL scenarios that we use with this datasets are:

```python
scenario1 = [
    ["vibrato", "straight"],
    ["belt", "breathy"],
    ["lip_trill", "spoken"],
    ["inhaled", "trill"],
    ["trillo", "vocal_fry"],
]

scenario2 = [
    ["belt", "trill"],
    ["vibrato", "inhaled"],
    ["breathy", "straight"],
    ["vocal_fry", "lip_trill"],
    ["spoken", "trillo"],
]

scenario3 = [
    ["spoken", "breathy"],
    ["straight", "inhaled"],
    ["lip_trill", "trillo"],
    ["vibrato", "vocal_fry"],
    ["trill", "belt"],
]
```

## NSynth
[NSynth](https://magenta.tensorflow.org/datasets/nsynth) is an audio dataset containing 305,979 musical notes, each with a unique pitch, timbre, and envelope. In our case, we use this dataset for instrument (family) classification.

### Class-Incremental Learning Scenarios
The CIL scenarios that we use with this datasets are:

```python
scenario1 = [
    ["bass", "brass"],
    ["flute", "guitar"],
    ["keyboard", "mallet"],
    ["organ", "reed"],
    ["string", "synth_lead", "vocal"],
]

scenario2 = [
    ["keyboard", "synth_lead"],
    ["mallet", "guitar"],
    ["reed", "string", "bass"],
    ["vocal", "brass"],
    ["organ", "flute"],
]

scenario3 = [
    ["guitar", "synth_lead", "organ"],
    ["string", "brass"],
    ["mallet", "reed"],
    ["keyboard", "bass"],
    ["flute", "vocal"],
]
```

