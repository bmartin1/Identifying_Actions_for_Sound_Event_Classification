## Identifying Actions for Sound Event Classification

###### [Overview](#Identifying-Actions-for-Sound-Event-Classification) | [Abstract](#abstract) | [ESC-50 dataset](#ESC-50-dataset) | [Paper reviews](#Paper-reviews) | [Acknowledgements](#Acknowledgements)

## Abstract

In Psychology, actions are paramount for humans to identify sound events. In Machine Learning (ML), action recognition achieves high accuracy; however, it has not been asked whether  identifying actions can benefit Sound Event Classification (SEC), as opposed to mapping the audio directly to a sound event. Therefore, we propose a new Psychology-inspired approach for SEC that includes identification of actions via human listeners. To achieve this goal, we used crowdsourcing to have listeners identify 20 actions that in isolation or in combination may have produced any of the 50 sound events in the well-studied dataset ESC-50. The resulting annotations for each audio recording relate actions to a database of sound events for the first time. The annotations were used to create semantic representations called Action Vectors (AVs). We evaluated SEC by comparing the AVs with two types of audio features -- log-mel spectrograms and state-of-the-art audio embeddings. Because audio features and AVs capture different abstractions of the acoustic content, we combined them and achieved one of the highest reported accuracies (88%).

<img src="av_pipeline.png" alt="SEC pipeline" title="SEC pipeline" align="center" width="500" height="300" />

Typically, SEC takes the input audio, computes audio features and assigns a class label. We proposed to add an intermediate step where listeners identify actions in the audio. The identified actions are transformed into Action Vectors and are used for automatic SEC.

## Actions for ESC-50 dataset

In order to relate actions to sound events, we chose a well-studied sound event dataset called ESC-50. We selected 20 actions that in isolation or combination could have produced at least part (of most) of the 50 sound events.

||||||
| :--- | :--- | :--- | :--- | :--- |
|dripping |rolling |groaning |crumpling |wailing|
|splashing |scraping |gasping | blowing |calling |
|pouring   |exhaling |singing |exploding |ringing |
|breaking |vibrating |tapping |rotating |sizzling |


The [**ESC-50 dataset**](https://github.com/karolpiczak/ESC-50) is a sound event labeled collection of 2000 audio recordings suitable for benchmarking methods of environmental sound classification. The dataset consists of 5-second-long recordings organized into 50 semantical classes (with 40 examples per class) loosely arranged into 5 major categories:

| <sub>Animals</sub> | <sub>Natural soundscapes & water sounds </sub> | <sub>Human, non-speech sounds</sub> | <sub>Interior/domestic sounds</sub> | <sub>Exterior/urban noises</sub> |
| :--- | :--- | :--- | :--- | :--- |
| <sub>Dog</sub> | <sub>Rain</sub> | <sub>Crying baby</sub> | <sub>Door knock</sub> | <sub>Helicopter</sub></sub> |
| <sub>Rooster</sub> | <sub>Sea waves</sub> | <sub>Sneezing</sub> | <sub>Mouse click</sub> | <sub>Chainsaw</sub> |
| <sub>Pig</sub> | <sub>Crackling fire</sub> | <sub>Clapping</sub> | <sub>Keyboard typing</sub> | <sub>Siren</sub> |
| <sub>Cow</sub> | <sub>Crickets</sub> | <sub>Breathing</sub> | <sub>Door, wood creaks</sub> | <sub>Car horn</sub> |
| <sub>Frog</sub> | <sub>Chirping birds</sub> | <sub>Coughing</sub> | <sub>Can opening</sub> | <sub>Engine</sub> |
| <sub>Cat</sub> | <sub>Water drops</sub> | <sub>Footsteps</sub> | <sub>Washing machine</sub> | <sub>Train</sub> |
| <sub>Hen</sub> | <sub>Wind</sub> | <sub>Laughing</sub> | <sub>Vacuum cleaner</sub> | <sub>Church bells</sub> |
| <sub>Insects (flying)</sub> | <sub>Pouring water</sub> | <sub>Brushing teeth</sub> | <sub>Clock alarm</sub> | <sub>Airplane</sub> |
| <sub>Sheep</sub> | <sub>Toilet flush</sub> | <sub>Snoring</sub> | <sub>Clock tick</sub> | <sub>Fireworks</sub> |
| <sub>Crow</sub> | <sub>Thunderstorm</sub> | <sub>Drinking, sipping</sub> | <sub>Glass breaking</sub> | <sub>Hand saw</sub> |

## Citing

If you find this research or annotations useful please cite the paper below:

[Identifying Actions for Sound Event Classification](https://arxiv.org/abs/2104.12693)

      @inproceedings{elizalde2021identifying,
              title={Identifying actions for sound event classification},
              author={Elizalde, Benjamin and Revutchi, Radu and Das, Samarjit and Raj, Bhiksha and Lane, Ian and Heller, Laurie M},
              booktitle={2021 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
              pages={26--30},
              year={2021},
              organization={IEEE}
      }

For more research on Sound Event Classification with Machine Learning + Psychology refer to:

[Never-Ending Learning of Sounds - PhD thesis](https://kilthub.cmu.edu/ndownloader/files/25813502)

      @phdthesis{elizalde2020never,
            title={Never-Ending Learning of Sounds},
            author={Elizalde, Benjamin},
            year={2020},
            school={Carnegie Mellon University}
      }


## Reviews from WASPAA 2021

Reviewer 1: "This paper presents a new idea of annotating sound events [...] is very interesting and highly novel."

Reviewer 2: "This is a strong paper [...] well motivated, organized and written."

Reviewer 3: "The idea of using semantic information to help the learning process is quite interesting."

## Acknowledgements

Thanks to the different funding sources, Bosch Research Pittsburgh, Sense Of Wonder Group and CONACyT.
