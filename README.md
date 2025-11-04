# P300_based_BCI
BCI for real time extraction and neurofeedback based on P300/CPP ampltudes 

## P300: A Neural Marker of Attention and Cognitive Processing

In this project, the Nback task is going to be used to study P300. 
P300 is an event-related potential (ERP) that occurs whenever a "rare/odd" stimuli appears. It is well established in literature and is studied either using the oddball paradigm task (visual or auditory) or
the Nback task. 

The P300 (or P3) is an **event-related potential (ERP)** component that appears roughly 300 ms after a rare, relevant, or surprising event in an EEG signal. It is typically observed as a positive deflection over **parietal scalp regions** (CP and P electrodes of the EEG, usually at Pz but it differs from subject to subject).
The P300 reflects **attentional resource allocation and working-memory updating** — in other words, how efficiently the brain detects and evaluates important stimuli. 
A larger P300 amplitude indicates stronger engagement of attention and memory processes, while a delayed or reduced P300 can signal slowed cognitive processing or diminished attention.

## The Nback Task: Measuring Working Memory and Cognitive Load

The Nback task is a widely used paradigm to measure working memory and attentional control. Participants view or hear a sequence of stimuli and must indicate when the current item matches one presented N steps earlier (e.g., 1-back, 2-back).
As N increases, the task becomes more demanding, requiring greater memory maintenance and mental updating. EEG recordings during Nback tasks allow researchers to track how brain activity — especially the P300 — changes with increasing cognitive load.

## P300 and Cognitive Decline

**Aging and neurodegenerative conditions** (e.g., mild cognitive impairment, Alzheimer’s disease) are often accompanied by **reductions in P300 amplitude** and **increases in P300 latency**. These changes reflect slower stimulus evaluation and **reduced attentional capacity**.
Therefore, studying the P300 provides a non-invasive neurophysiological window into cognitive decline. By analyzing P300 dynamics during tasks like the Nback, researchers can assess how well the brain maintains attention and working-memory function across time or after interventions.

## Project Overview

In this project, I investigate different neural stimulation techniques (tACS) as well as meditation as methods to enhance P300 amplitude during the Nback task using EEG.
This is paired with a Brain-Computer Interface (BCI) that provides real-time feedback on P300, using different pre-processing strategies (CCA,XDawn) and Machine Learning models (XGBoost) to boost the enhancement observed in P300. 

The first step will be to develop the **visual interface of the Nback Task** that you can find in the **Visual Interface** folder.

The second step is the offline data analysis and P300 extraction and visualization. In the **Offline** folder, you can find the function to load the data, extract the trial segments, filter the EEG data, visualize and plot P300 as well as get its metrics to be able to compare pre and post intervention (tACS or meditation). I will also train an XGBoost decoder model on the offline data that I will save and use for subject-specific BCI implementaion.

The third step is to start implementing the **Online BCI Visual Interface and real-time P300 decoding**. You will find the scripts and explanations for this in the **Online** folder. There you will also find script for evaluation of Online performance and statistical analysis.
