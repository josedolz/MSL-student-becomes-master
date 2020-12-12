import numpy as np
import torch
import torch.nn as nn
import torchvision
import os
import pdb
import nibabel as nib

from torch.autograd import Variable
from progressBar import printProgressBar
from os.path import isfile, join
from PIL import Image
from medpy.metric.binary import dc,hd95
#import scipy.spatial
#import scipy.io as sio
#import torch.nn.functional as F
#import time

def load_nii(imageFileName, printFileNames):
    if printFileNames == True:
        print (" ... Loading file: {}".format(imageFileName))

    img_proxy = nib.load(imageFileName)
    imageData = img_proxy.get_data()

    return (imageData, img_proxy)

def computeDSC(pred, gt):

    dscAll= []

    for i_b in range(pred.shape[0]):
        pred_id = pred[i_b,1,:]
        gt_id = gt[i_b,0,:]
        dscAll.append(dc(pred_id.cpu().data.numpy(),gt_id.cpu().data.numpy()))

    DSC = np.asarray(dscAll)

    return DSC.mean()


def getImageImageList(imagesFolder):
    if os.path.exists(imagesFolder):
       imageNames = [f for f in os.listdir(imagesFolder) if isfile(join(imagesFolder, f))]

    imageNames.sort()

    return imageNames

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def predToSegmentation(pred):
    Max = pred.max(dim=1, keepdim=True)[0]
    x = pred / Max
    return (x == 1).float()


def getTargetSegmentation(batch):
    # input is 1-channel of values between 0 and 1
    # output is 1 channel of discrete values : 0, 1, 2, 3, etc
    denom = 1.0 # To change if there are more than 2 classes

    return (batch / denom).round().long().squeeze()

def inferenceTraining(net, img_batch, modelName, mode, singleBranch = True):
    total = len(img_batch)

    net.eval()

    DSC = []
    DSCW = []

    softMax = nn.Softmax().cuda()
    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        image, labels, img_names = data

        input = to_var(image)
        Segmentation = to_var(labels)

        segmentation_prediction, segmentation_predictionW = net(input)

        pred_y = softMax(segmentation_prediction)
        segmentation_prediction_ones = predToSegmentation(pred_y)
        DSC.append(computeDSC(segmentation_prediction_ones, Segmentation))

        if singleBranch == False:
            pred_y_w = softMax(segmentation_predictionW)
            segmentation_prediction_ones_W = predToSegmentation(pred_y_w)
            DSCW.append(computeDSC(segmentation_prediction_ones_W, Segmentation))

        str_1 = img_names[0].split('/Img/')

        str_subj_main = str_1[1].split('_')

        modelName_str = modelName.split('models/')

        path = os.path.join('./Results/Images/', modelName_str[0])
        path_top = os.path.join(path, mode, 'top',str_subj_main[0],str_subj_main[1])
        path_bottom = os.path.join(path, mode, 'bottom',str_subj_main[0],str_subj_main[1])

        if not os.path.exists(path_top):
            os.makedirs(path_top)

        if not os.path.exists(path_bottom):
            os.makedirs(path_bottom)

        torchvision.utils.save_image(segmentation_prediction_ones[:,1].data * 255, os.path.join(path_top, str_subj_main[2]), padding=0)

        if singleBranch == False:
            torchvision.utils.save_image(segmentation_prediction_ones_W[:,1].data * 255, os.path.join(path_bottom, str_subj_main[2]), padding=0)

    printProgressBar(total, total, done="[Inference] Segmentation Done !")

    DSC = np.asarray(DSC)
    DSCW = np.asarray(DSCW)
    if singleBranch == True:
        return DSC.mean()
    else:
        return [DSC.mean(), DSCW.mean()]

def inferenceTesting(net, img_batch, modelName, modelType, mode, singleBranch = True):
    total = len(img_batch)

    net.eval()

    DSC = []
    DSCW = []

    softMax = nn.Softmax().cuda()
    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        image, labels, img_names = data


        idx = np.where(labels < 1.0)
        labels[idx] = 0.
        # img_names_ALL.append(img_names[0].split('/')[-1].split('.')[0])
        input = to_var(image)
        Segmentation = to_var(labels)

        if singleBranch == True:
            segmentation_prediction = net(input)
            segmentation_prediction = segmentation_prediction[0]
        else:
            segmentation_prediction, segmentation_predictionW = net(input)

        #if modelType == 2:
        #    segmentation_prediction = segmentation_predictionW
        #pdb.set_trace()
        pred_y = softMax(segmentation_prediction)
        segmentation_prediction_ones = predToSegmentation(pred_y)
        DSC.append(computeDSC(segmentation_prediction_ones, Segmentation))

        if singleBranch == False:
            pred_y_w = softMax(segmentation_predictionW)
            segmentation_prediction_ones_W = predToSegmentation(pred_y_w)
            DSCW.append(computeDSC(segmentation_prediction_ones_W, Segmentation))

        str_1 = img_names[0].split('/Img/')
        str_subj = str_1[1].split('.')
        str_subj_main = str_1[1].split('_')

        modelName_str = modelName.split('/')

        path = os.path.join('./Results/Images/', modelName_str[2])
        path_top = os.path.join(path, mode, 'top',str_subj_main[0],str_subj_main[1])
        path_bottom = os.path.join(path, mode, 'bottom',str_subj_main[0],str_subj_main[1])

        if not os.path.exists(path_top):
            os.makedirs(path_top)

        if not os.path.exists(path_bottom):
            os.makedirs(path_bottom)

        torchvision.utils.save_image(segmentation_prediction_ones[:,1].data * 255, os.path.join(path_top, str_subj_main[2]), padding=0)

        if singleBranch == False:
            torchvision.utils.save_image(segmentation_prediction_ones_W[:,1].data * 255, os.path.join(path_bottom, str_subj_main[2]), padding=0)

        #### For the GT
        #path = os.path.join('./Results/GT',mode,'/Png/')
        '''path = './Results/GT/' + mode + '/Png/'
        path_top = os.path.join(path,str_subj_main[0], str_subj_main[1])
        path_bottom = os.path.join(path, str_subj_main[0], str_subj_main[1])

        #pdb.set_trace()
        if not os.path.exists(path_top):
            os.makedirs(path_top)

        if not os.path.exists(path_bottom):
            os.makedirs(path_bottom)

        torchvision.utils.save_image(labels.data * 255,
                                     os.path.join(path_top, str_subj_main[2]), padding=0)
        torchvision.utils.save_image(labels.data * 255,
                                     os.path.join(path_bottom, str_subj_main[2]), padding=0)'''

    printProgressBar(total, total, done="[Inference] Segmentation Done !")

    DSC = np.asarray(DSC)
    DSCW = np.asarray(DSCW)

    if singleBranch == True:
        return DSC.mean()
    else:
        return [DSC.mean(), DSCW.mean()]

def reconstruct3D(modelName,modelType,mode,singleBranch = True):

    modelName_str = modelName.split('/')

    if modelType==1:
        path = os.path.join('./Results/Images/', modelName_str[2],mode,'top')
    else:
        path = os.path.join('./Results/Images/', modelName_str[2],mode,'bottom')

    subjNames = os.listdir(path)
    subjNames.sort()
    xSize = 256
    ySize = 256

    for s_i in range(len(subjNames)):
        folderNames = os.listdir(os.path.join(path,subjNames[s_i]))
        folderNames.sort()
        for f_i  in range(len(folderNames)):

            imgNames = os.listdir(os.path.join(path,subjNames[s_i],folderNames[f_i]))
            numImages = len(imgNames)
            vol_numpy = np.zeros((xSize, ySize, numImages))

            for t_i in range(numImages - 1):

                imagePIL = Image.open(os.path.join(path, subjNames[s_i], folderNames[f_i],str(t_i+1)+'.png')).convert('L')
                imageNP = np.array(imagePIL)
                vol_numpy[:, :, t_i] = imageNP

            xform = np.eye(4) * 2
            imgNifti = nib.nifti1.Nifti1Image(vol_numpy, xform)

            if modelType ==1:
                if not os.path.exists(os.path.join('Results/Images',modelName_str[2],'Nifti',mode,'top')):
                    os.makedirs(os.path.join('Results/Images',modelName_str[2],'Nifti',mode,'top'), exist_ok=True)

                niftiName = 'Results/Images/' + modelName_str[2] + '/Nifti/' + mode+ '/top/'+subjNames[s_i] + '_'+folderNames[f_i]
            else:
                if not os.path.exists(os.path.join('Results/Images', modelName_str[2], 'Nifti', mode, 'bottom')):
                    os.makedirs(os.path.join('Results/Images', modelName_str[2], 'Nifti', mode, 'bottom'), exist_ok=True)

                niftiName = 'Results/Images/' + modelName_str[2] + '/Nifti/' + mode + '/bottom/' + subjNames[s_i] + '_' + folderNames[f_i]

            nib.save(imgNifti, niftiName)

def reconstruct3D_GT(modelName,mode):

    path = './Results/GT/' + mode + '/Png/'

    subjNames = os.listdir(path)
    subjNames.sort()
    xSize = 256
    ySize = 256

    for s_i in range(len(subjNames)):
        folderNames = os.listdir(os.path.join(path,subjNames[s_i]))
        folderNames.sort()
        for f_i  in range(len(folderNames)):

            imgNames = os.listdir(os.path.join(path,subjNames[s_i],folderNames[f_i]))
            numImages = len(imgNames)
            vol_numpy = np.zeros((xSize, ySize, numImages))

            for t_i in range(numImages - 1):

                imagePIL = Image.open(os.path.join(path, subjNames[s_i], folderNames[f_i],str(t_i+1)+'.png')).convert('L')
                imageNP = np.array(imagePIL)
                vol_numpy[:, :, t_i] = imageNP

            xform = np.eye(4) * 2
            imgNifti = nib.nifti1.Nifti1Image(vol_numpy, xform)

            if not os.path.exists(os.path.join('Results/GT/',mode,'Nifti')):
                os.makedirs(os.path.join('Results/GT',mode, 'Nifti'), exist_ok=True)

            niftiName = 'Results/GT/'+mode+'/Nifti/' + subjNames[s_i] + '_'+folderNames[f_i]
            nib.save(imgNifti, niftiName)


def evaluate(modelName,mode,modelType):
    path_GT = './Results/GT/' + mode + '/Nifti/'

    modelName_str = modelName.split('/')
    if modelType==1:
        path_Pred = os.path.join('Results/Images/', modelName_str[2], 'Nifti',mode,'top')
    else:
        path_Pred = os.path.join('Results/Images/', modelName_str[2], 'Nifti', mode,'bottom')

    GT_names = getImageImageList(path_GT)
    Pred_names = getImageImageList(path_Pred)

    GT_names.sort()
    Pred_names.sort()

    numClasses = 1
    DSC = np.zeros((len(Pred_names), numClasses))
    HD = np.zeros((len(Pred_names), numClasses))

    for s_i in range(len(Pred_names)):
        path_Subj_GT = path_GT + '/' + GT_names[s_i]
        path_Subj_pred = path_Pred + '/' + Pred_names[s_i]

        [imageDataGT, img_proxy] = load_nii(path_Subj_GT, printFileNames=False)
        [imageDataCNN, img_proxy] = load_nii(path_Subj_pred, printFileNames=False)

        for c_i in range(numClasses):
            label_GT = np.zeros(imageDataGT.shape, dtype=np.int8)
            label_CNN = np.zeros(imageDataCNN.shape, dtype=np.int8)

            idx_GT = np.where(imageDataGT == 255)
            label_GT[idx_GT] = 1

            idx_CNN = np.where(imageDataCNN == 255)
            label_CNN[idx_CNN] = 1

            DSC[s_i,c_i] = dc(label_GT,label_CNN)
            HD[s_i,c_i] = hd95(label_GT,label_CNN)

    return [DSC,HD]
