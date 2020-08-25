###Quantifying Uncertainty in Blood Oxygen Estimation Models from Real-World Data

<img src=misc/ppg.png>

Read the [full paper](misc/paper.pdf).

## Abstract
The blood oxygen level of a patient is an important clinical metric that is useful in the diagnosis and monitoring of respiratory illnesses, including **Covid-19**. Arterial oxygen saturation can be estimated through non-invasive methods with high levels of accuracy, such as by measuring the perfusion of blood to the skin. This peripheral oxygen saturation (SpO2) level is measured with a **pulse oximeter** device that leverages the different absorbance levels of oxy- and deoxy- haemoglobin at 600nm and 940nm wavelengths. A photoplethysmogram (PPG) signal is constructed from the raw input of the photodiode. This typically involves filtering frequencies outside of a specific range and correcting for the light source used. Unfortunately, this specialised hardware can be cost-prohibitive and under-supplied in times of crisis, such as the **Covid-19 pandemic**.

The open source [CoVital project](https://www.covital.org/) aims to recreate the accuracy of commercial pulse oximeters on any modern smartphone. The team of volunteers has worked to compile a dataset from multiple sources and is employing machine learning to produce accurate SpO2 estimation models that can be deployed in a smartphone app.

Training models for medical tasks introduces a host of unique challenges. Notably, **it is valuable to have not only predictions, but estimates of the confidence of those predictions** – which help clinicians and patients make better decisions about when to trust the model. Without such confidence estimates, even the most accurate models in evaluation may be dangerous in deployment. To address this issue in the context of smartphone-driven SpO2 estimation, we evaluated the use of **dropout techniques** in a hybrid **deep learning** model to generate 95% confidence intervals (CIs).

## Project Contributors
* [CoVital volunteer team](https://github.com/CoVital-Project/Spo2_evaluation/graphs/contributors)
* [Gianluca Truda](https://github.com/gianlucatruda)
* [Serafim Korovin](https://github.com/Serafim179)

## Referencing

BibTeX:

```
@article{truda2020covital
  title={Quantifying Uncertainty in Blood Oxygen Estimation Models from Real-World Data},
  author={Truda, Gianluca and Korovin, Serafim and Kantorik, Adam},
  year={2020}
}
```
