# STT-Swedish
This repository contains a scalable ML system to fine tune the [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper) transformer model for speech-to-text on the Swedish language, as part of the course ID2223 at KTH. There are three main components, namely the feature pipeline, training pipeline, and interactive UI that are explained in detail below. The rest of the readme explains how we improved the approach we were given as a starting point. 

## Feature Pipeline
The feature pipeline is implemented as a notebook that should be run on colab. It downloads the Swedish subset of the common voice dataset and extracts the features and labels that the Whisper model expects as input. It does the same for the NST dataset that can be used to extend the training dataset of the model (see section "Data-Centric Improvement" below). The data is stored on Google Drive and encompasses 16 GB for the common voice dataset and 6 GB for the additional data from NST.

Note that preprocessing the NST data can lead to RAM issuses on collab. We therefore advise to download the raw data to disk (still on collab) first and then restart the kernel and apply the pre-processing to the downloaded data. Otherwise the notebook may crash because it runs out of RAM.

## Training Pipeline
The training pipeline can either be run locally or on Google Collab. It downloads the pre-processed data from Google Drive (only if on collab, if local you should download it manually once) and starts training. Based on the model_variant chosen, the hyperparameters and dataset are modified in different ways. When training finished, the new model is pushed to Huggingface.

## Interactive UI
We provide an [interactive UI](https://huggingface.co/spaces/Neprox/STT-Swedish) on Huggingface where a user can translate Swedish audio into a language of their choosing. As a source for the audio, either an input to the audio can be provided or a link to a YouTube video (including the amount of seconds to translate). The UI performs feature extraction on the provided audio and transcribes it using the fine tuned Whisper model. It then calls the Google Translate API with the specified language to retrieve a translation of the spoken text. Unfortunately, querying the model takes a lot of time, as it is run on a CPU only and would most likely benefit from a GPU. You can try out the UI using this [YouTube video](https://www.youtube.com/watch?v=wIlOPJLhks4&ab_channel=Wikitongues). 

## Model Improvements

We trained a model using the example code provided in the lab description, that means using the common voice data set and hyperparameters specified in the example. The resulting model (and training history) can be found on the page of our [base model on Huggingface](https://huggingface.co/Neprox/STT-swedish-base-model). Notably, the WER converges very quickly to a value around 20 and we see that the model is overfitting strongly as the validation loss rises significantly while the training loss decreases. Based on this observation we argue that additional regularization is necessary. Below we describe the model-centric approach and the data-centric approach we used to improve the model. Both approaches seemed to act as regularizers and improved the model. Combining both approaches lead to a slightly better result over using them individually. The resulting final model can be found on [Huggingface](https://huggingface.co/Neprox/STT-Swedish-Whisper). The best performing hyperparameters depend on the specific dataset used and thus it is better to conduct Hyperparameter tuning on the full extended dataset. However, our limited computing resources don't allow for this luxury and thus we simply applied the best hyperparameters found on the base dataset to the full extended dataset.

### Model-Centric Improvement
To improve a model, a simple approach is to define value ranges for a set of hyperparameters to be tested within a grid search (or something similar). Because of the amount of computing power a single training run takes, it was necessary to prioritize testing the effect of only very few parameters. As described above, we chose to introduce a regularizer and tried out the following (individual) modifications to the base model:

**attention-dropout** of 0.3 improved the model's WER by roughly 0.5 points (see [Huggingface](https://huggingface.co/Neprox/STT-swedish-attention-dropout-model))

**weight_decay** of 0.001 lead to a much worse WER of 28+ (see [Huggingface](https://huggingface.co/axel-kaliff/STT-Whisper-Sv-Fine-Tune-Weight-Decay))

Additional parameters that could have been tested include **dropout on the fully connected layers** of the model. Especially lowering the **learning rate** could help finding a better local optimum with gradient descent. However, given that training time takes longer when adding regularization like attention-dropout or increasing the size of the dataset, this seemed impractical to test with our limited resoures.

### Data-Centric Improvement

Upon extensive research, we found the [NST Swedish ASR Database](https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-56/) which encompasses 124,2 GB of transcribed audio files of Swedish speakers. From this dataset we generated 6400 additional data samples and added them to the training set which constitutes an increase in size of 51.8% w.r.t. the original training dataset. At the same time, we left the test dataset untouched for better comparability with other training runs. Although we diluted the training set with data that could be substantially different from the test data, we observed an improvement of the WER to a value of 19.2. This model which is trained with the same hyperparameters as the base model but on more data, can be found on [Huggingface](https://huggingface.co/Neprox/STT-swedish-extended-dataset-model).

