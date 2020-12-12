# Teach me to segment with mixed-supervision:when the student becomes the master

This repository contains the code of our recent paper: Teach me to segment with mixed-supervision:when the student becomes the master.


<br>
<img src="https://github.com/josedolz/MSL-student-becomes-master/blob/main/Images/IPMI-2021.png" />
<br>

## Dataset preparation

Download the dataset from the [ACDC Challenge](https://www.creatis.insa-lyon.fr/Challenge/acdc/) or your own data.

Then, you can find the instructions to pre-process the data and obtain partial labeled annotations in [this repository](https://github.com/LIVIAETS/SizeLoss_WSS)



If you just want to run your own dataset with this code, you simply need to convert the 3D volumes to 2D slices/images. Then, the structure to save the images should be (in trainSemi folder you can avoid having the 'WeaklyAnnotations' folder, but I suggest to create the same structure across all the folders):

```bash  
-| MainFolder
--|N (number of labeled images)
----| trainFull
--------| Img/
------------| patientXXXX_01_1.png
------------| patientXXXX_01_2.png
------------| ....
--------| GT/
------------| patientXXXX_01_1.png
------------| patientXXXX_01_2.png
------------| ....
--------| WeaklyAnnotations/
------------| patientXXXX_01_1.png
------------| patientXXXX_01_2.png
------------| ....
----| trainSemi
--------| Img/
------------| patientXXXX_01_1.png
------------| patientXXXX_01_2.png
------------| ....
--------| GT/
------------| patientXXXX_01_1.png
------------| patientXXXX_01_2.png
------------| ....
--------| WeaklyAnnotations/
------------| patientXXXX_01_1.png
------------| patientXXXX_01_2.png
------------| ....
----| val/
----| test/
```

Note: Val and Test folder will have the same structure as trainFull and trainSemi folders. Furthermore, note that we evaluated our method on the ACDC dataset, which contains two sets of images (systole and diastole) per exam. This explains the '_01_' in between the patient name and the slide number (patient_systole/diastole_slidenumber.png).

  
## Training 

The code for training will be released after publication of the paper.

## Trained models

We also share the trained networks for our two models, i.e., KL and KL+Ent, which can be found in [this link](https://drive.google.com/drive/folders/1CtyaAcg_-8zIxzzIcnuxD35iGbvEs52q?usp=sharing)


## Testing 

To test a given model, download the trained models in the previous link into a folder named 'TrainedModel'. Then, you can use the following script:

```
CUDA_VISIBLE_DEVICES=1 python runTest.py --modelName ./TrainedModel/path_and_name_of_your_model
```

NOTE: This script will segment both the validation and testing images, and generate 3D volumes from them. In order to be able to successfully run the whole script, and therefore compare to the GT, you need to create 3D GT volumes in .nifti format and saved them in the following path:

```bash  
-|Results
---|GT
------|val/Nifti
---------|patientXXX_YY.nii
---------|patientXXX_YY.nii
---------|......
------|test/Nift
---------|patientXXX_YY.nii
---------|patientXXX_YY.nii
---------|......
```

## Several results

Qualitative results comparing the proposed methods to other baselines for mixed-supervised learning, as well as to full supervision (upper bound) 

<br>
<img src="https://github.com/josedolz/MSL-student-becomes-master/blob/main/Images/IPMI-1.png" />
<br>

The top and bottom rows visualize respectively the prediction probability maps obtained by the proposed models

<br>
<img src="https://github.com/josedolz/MSL-student-becomes-master/blob/main/Images/IPMI-Entropy.png" />
<br>



## Requirements

- The code has been written in Python (3.6.9)
- Torch version: 1.6.0
- scikit-image: 0.15.0

### Versions
- Version 1.0. 
  * December,10th. 2020
   

You can contact me at: jose.dolz.upv@gmail.com