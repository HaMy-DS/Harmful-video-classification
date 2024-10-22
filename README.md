# Harmful Videos Classification for Safer Online Social Network
**Abstract:** Video classification is a prominent research topic focusing
mainly on object, context, and action classification. However, recent advancements in speech processing, computer vision, and natural language
processing have presented new challenges in effectively classifying videos
using all three modalities of image, audio, and language, rather than
solely relying on one or two factors. In this study, we introduce our self-collected dataset, HarmfulVideoVN2023 including 1,589 videos that are
several seconds long, and investigate the effectiveness of image, audio,
and language features for classifying assumedly harmful videos, particularly for children according to our defined labels. We apply 3D CNN
models to video frames, transfer learning models for audio spectrograms
and transformers for textual features. Our experimental findings indicate
that training models using image frames decoded from videos demands
substantial computational resources, our best-recorded result is 80.77%
accuracy. In contrast, employing pre-trained models for images extracted
from audio spectrograms and textual features derived from video yields
accurate results and significantly reduces training time as well as computational cost (83.34% and 87.18% accuracy for audio spectrograms and
texts respectively)

<p align="center">
<img width="748" alt="{0AD0D917-05DA-4A3A-864C-29819C91B825}" src="https://github.com/user-attachments/assets/1862a2e7-f684-4b0f-91be-dcf191532897">
</p>

## Data collection

```source code/CrawVideo.ipynb``` is code for downloading our video dataset from YouTube using Google’s YouTube Data
API, with keywords used to search for and download videos based on the definition of each label.  Our labels of the dataset are defined as follows:

* Normal: Videos with normal content that do not violate community standards.
* Horrible: Videos with exciting and inappropriate content for children, including scary, horrifying, or frightening images.
* Offensive: Videos containing offensive, attacking, or mocking content towards individuals or groups. These videos may include disrespectful behavior, speech, or actions, racial discrimination, incitement to violence, and offensive content related to gender, religion, nationality, or other personal characteristics.
* Pornographic: Videos that may contain sensitive situations and actions, such as adult sexual behavior, advertising or displaying pornographic content, inappropriate or explicit body parts.
* Violence: Videos containing images or content with violent or physical violence towards humans or animals. These videos may include sensitive images, violence, pain, harm, or abuse. Examples include images of armed attacks, violent actions or fights, videos of animals being abused or attacked, witnessing accidents, disasters, or dangerous situations that cause harm to

| Label        | Train | Val | Test | Total |
|--------------|-------|-----|------|-------|
| Normal       | 257   | 31  | 31   | 319   |
| Horror       | 185   | 22  | 22   | 229   |
| Offensive    | 252   | 31  | 31   | 314   |
| Pornographic | 261   | 32  | 32   | 325   |
| Violence     | 322   | 40  | 40   | 402   |

<p align="center">
<img width="700" alt="{7AB86941-0E9B-4760-B049-09BC5F87992C}" src="https://github.com/user-attachments/assets/8c76be23-6c03-4d3d-9676-bc353298f388">
</p>

## Results

The results of the prediction models using image frames from video:

| Model                      | Accuracy | Precision | Recall | F1    |
|----------------------------|----------|-----------|--------|-------|
| 3D ResNet                   | 0.6558   | 0.7218    | 0.6611 | 0.6901|
| ResNet50 (IN)               | 0.1428   | 0.1429    | NaN    | 0.2501|
| Inception–ResNet-v2 (IN)    | 0.1429   | 0.1429    | NaN    | 0.2501|
| InceptionV3 (IN)            | 0.2013   | 0.2013    | NaN    | 0.3353|
| MoViNet (K600)              | 0.8077   | 0.7184    | 0.7551 | 0.7363|

We also implemented ```source code/Transfer_learning_with_VideoSwin.ipynb```. However, because of lacking computational resources, I can't finish training this model.

The results of the prediction models using spectrogram images, our code available at ```source code/Image classification for data spectrogram.ipynb```

| Model               | Accuracy | Precision | Recall | F1    |
|---------------------|----------|-----------|--------|-------|
| VGG16               | 0.7143   | 0.6819    | 0.7094 | 0.672 |
| VGG19               | 0.7051   | 0.6898    | 0.6784 | 0.6544|
| DenseNet121         | 0.3571   | 0.3472    | 0.5346 | 0.3039|
| DenseNet169         | 0.3333   | 0.2538    | 0.4595 | 0.1815|
| DenseNet201         | 0.4615   | 0.3286    | 0.4739 | 0.3146|
| ResNet50            | 0.7403   | 0.7499    | 0.7547 | 0.7389|
| ResNet50V2          | 0.3333   | 0.2000    | 0.0667 | 0.0100|
| Xception            | 0.4872   | 0.3671    | 0.3941 | 0.3273|
| InceptionV3         | 0.3571   | 0.3143    | 0.5034 | 0.2601|
| InceptionResNetV2   | 0.3462   | 0.2446    | 0.1162 | 0.1559|
| NASNetMobile        | 0.5769   | 0.5065    | 0.5531 | 0.5099|
| ConvNeXtTiny        | 0.8077   | 0.7110    | 0.6415 | 0.6735|
| ConvNeXtSmall       | 0.8334   | 0.7450    | 0.7746 | 0.7530|
| ConvNeXtLarge       | 0.8205   | 0.7124    | 0.6764 | 0.6921|
| ConvNeXtBase        | 0.7821   | 0.7009    | 0.8333 | 0.7192|
| EfficientNetB0      | 0.6538   | 0.5011    | 0.5639 | 0.4878|
| EfficientNetB1      | 0.7949   | 0.7214    | 0.7618 | 0.7358|
| EfficientNetB2      | 0.7561   | 0.6497    | 0.7217 | 0.6686|
| EfficientNetB3      | 0.8205   | 0.7310    | 0.8422 | 0.7437|
| EfficientNetB4      | 0.6923   | 0.5719    | 0.5278 | 0.5366|
| EfficientNetB7      | 0.4615   | 0.4856    | 0.5725 | 0.4029|
| EfficientNetV2M     | 0.6538   | 0.5798    | 0.6627 | 0.6013|
| EfficientNetV2L     | 0.7179   | 0.6047    | 0.6075 | 0.6075|

<p align="center">
<img width="700" alt="{A417FB66-AD68-4AB5-8D98-586A5C686B41}" src="https://github.com/user-attachments/assets/621c0646-5f6a-46df-9849-45a25f3f827c">
</p>

The results of the ```source code/simpletransformers.ipynb``` using text from Speech recognition:

| Model         | Accuracy | Precision | Recall  | F1     |
|---------------|----------|-----------|---------|--------|
| BiLSTM        | 0.5812   | 0.4736    | 0.5359  | 0.4827 |
| PhoBERT       | 0.8718   | 1.0000    | 0.8718  | 0.9315 |
| BERT          | 0.2949   | 0.1917    | 0.2949  | 0.1921 |
| RoBERTa       | 0.2821   | 0.0564    | 0.2000  | 0.0880 |
| XLM–RoBERTa   | 0.2821   | 0.0564    | 0.2000  | 0.0880 |
| CafeBERT      | 0.2821   | 0.0564    | 0.2000  | 0.0880 |
| ViSoBERT      | 0.6923   | 0.6357    | 0.6031  | 0.5983 |
| viBERT        | 0.7436   | 0.7671    | 0.6493  | 0.6414 |
| DistilBERT    | 0.2821   | 0.0564    | 0.2000  | 0.0880 |
| vELECTRA      | 0.5256   | 0.4768    | 0.4612  | 0.4536 |








  
