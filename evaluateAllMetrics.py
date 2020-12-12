import sys
import pdb
from os.path import isfile, join
import os
import numpy as np
import nibabel as nib
import scipy.io as sio
from medpy.metric.binary import dc,hd, asd, assd, ravd,hd95
import argparse

import SimpleITK as sitk
import scipy.spatial

import logging


labels = {1: 'Foreground',
          }

def load_nii(imageFileName, printFileNames):
    if printFileNames == True:
        print (" ... Loading file: {}".format(imageFileName))

    img_proxy = nib.load(imageFileName)
    imageData = img_proxy.get_data()

    return (imageData, img_proxy)

def getImageImageList(imagesFolder):
    if os.path.exists(imagesFolder):
       imageNames = [f for f in os.listdir(imagesFolder) if isfile(join(imagesFolder, f))]

    imageNames.sort()

    return imageNames

def getHausdorff(testImage, resultImage):
    """Compute the 95% Hausdorff distance."""
    hd = dict()
    for k in labels.keys():
        lTestImage = sitk.BinaryThreshold(testImage, k, k, 1, 0)
        lResultImage = sitk.BinaryThreshold(resultImage, k, k, 1, 0)

        # Hausdorff distance is only defined when something is detected
        statistics = sitk.StatisticsImageFilter()
        statistics.Execute(lTestImage)
        lTestSum = statistics.GetSum()
        statistics.Execute(lResultImage)
        lResultSum = statistics.GetSum()
        if lTestSum == 0 or lResultSum == 0:
            hd[k] = None
            continue

        # Edge detection is done by ORIGINAL - ERODED, keeping the outer boundaries of lesions. Erosion is performed in 2D
        eTestImage = sitk.BinaryErode(lTestImage, (1, 1, 0))
        eResultImage = sitk.BinaryErode(lResultImage, (1, 1, 0))

        hTestImage = sitk.Subtract(lTestImage, eTestImage)
        hResultImage = sitk.Subtract(lResultImage, eResultImage)

        hTestArray = sitk.GetArrayFromImage(hTestImage)
        hResultArray = sitk.GetArrayFromImage(hResultImage)

        # Convert voxel location to world coordinates. Use the coordinate system of the test image
        # np.nonzero   = elements of the boundary in numpy order (zyx)
        # np.flipud    = elements in xyz order
        # np.transpose = create tuples (x,y,z)
        # testImage.TransformIndexToPhysicalPoint converts (xyz) to world coordinates (in mm)
        # (Simple)ITK does not accept all Numpy arrays; therefore we need to convert the coordinate tuples into a Python list before passing them to TransformIndexToPhysicalPoint().
        testCoordinates = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in
                           np.transpose(np.flipud(np.nonzero(hTestArray)))]
        resultCoordinates = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in
                             np.transpose(np.flipud(np.nonzero(hResultArray)))]

        # Use a kd-tree for fast spatial search
        def getDistancesFromAtoB(a, b):
            kdTree = scipy.spatial.KDTree(a, leafsize=100)
            return kdTree.query(b, k=1, eps=0, p=2)[0]

        # Compute distances from test to result and vice versa.
        dTestToResult = getDistancesFromAtoB(testCoordinates, resultCoordinates)
        dResultToTest = getDistancesFromAtoB(resultCoordinates, testCoordinates)
        hd[k] = max(np.percentile(dTestToResult, 95), np.percentile(dResultToTest, 95))

    return hd


def getVS(testImage, resultImage):
    """Volume similarity.

    VS = 1 - abs(A - B) / (A + B)

    A = ground truth in ML
    B = participant segmentation in ML
    """
    # Compute statistics of both images
    testStatistics = sitk.StatisticsImageFilter()
    resultStatistics = sitk.StatisticsImageFilter()

    vs = dict()
    for k in labels.keys():
        testStatistics.Execute(sitk.BinaryThreshold(testImage, k, k, 1, 0))
        resultStatistics.Execute(sitk.BinaryThreshold(resultImage, k, k, 1, 0))

        numerator = abs(testStatistics.GetSum() - resultStatistics.GetSum())
        denominator = testStatistics.GetSum() + resultStatistics.GetSum()

        if denominator > 0:
            vs[k] = 1 - float(numerator) / denominator
        else:
            vs[k] = None

    return vs


def getDSC(testImage, resultImage):
    """Compute the Dice Similarity Coefficient."""
    dsc = dict()
    for k in labels.keys():
        testArray = sitk.GetArrayFromImage(sitk.BinaryThreshold(testImage, k, k, 1, 0)).flatten()
        resultArray = sitk.GetArrayFromImage(sitk.BinaryThreshold(resultImage, k, k, 1, 0)).flatten()

        # similarity = 1.0 - dissimilarity
        # scipy.spatial.distance.dice raises a ZeroDivisionError if both arrays contain only zeros.
        try:
            dsc[k] = 1.0 - scipy.spatial.distance.dice(testArray, resultArray)
        except ZeroDivisionError:
            dsc[k] = None

    return dsc


def getImages(testFilename, resultFilename):
    """Return the test and result images, thresholded and pathology masked."""
    testImage = sitk.ReadImage(testFilename)
    resultImage = sitk.ReadImage(resultFilename)


    # Check for equality
    assert testImage.GetSize() == resultImage.GetSize()

    # Get meta data from the test-image, needed for some sitk methods that check this
    resultImage.CopyInformation(testImage)

    # Remove pathology from the test and result images, since we don't evaluate on that
    #pathologyImage = sitk.BinaryThreshold(testImage, 9, 11, 0, 1)  # pathology == 9 or 10

    #maskedTestImage = sitk.Mask(testImage, pathologyImage)  # tissue    == 1 --  8
    #maskedResultImage = sitk.Mask(resultImage, pathologyImage)

    images_array_t = sitk.GetArrayFromImage(testImage)
    images_array_r = sitk.GetArrayFromImage(resultImage)

    #pdb.set_trace()
    #images_array_r = images_array_r*255

    images_array_r = images_array_r / 255
    images_array_t = images_array_t / 255

    resultImage = sitk.GetImageFromArray(images_array_r)
    testImage = sitk.GetImageFromArray(images_array_t)
    #

    # Force integer
    if not 'integer' in resultImage.GetPixelIDTypeAsString():
        resultImage = sitk.Cast(resultImage, sitk.sitkUInt8)


    return testImage, resultImage

def runEvaluationITK(argv):
    GT_Folder = './Results/GT/'+args.mode +'/Nifti/'
    CNN_Folder = args.path

    GT_names = getImageImageList(GT_Folder)
    CNN_names = getImageImageList(CNN_Folder)



    DSC_All = np.zeros((len(GT_names),1))
    H95_All = np.zeros((len(GT_names),1))
    VS_All = np.zeros((len(GT_names),1))
    #pdb.set_trace()
    for i in range(len(CNN_names)):
        #print(' {} '.format(CNN_names[i]))
        testImage, resultImage = getImages(os.path.join(GT_Folder, CNN_names[i]), os.path.join(CNN_Folder, CNN_names[i]))
        dsc = getDSC(testImage, resultImage)
        h95 = getHausdorff(testImage, resultImage)
        vs = getVS(testImage, resultImage)
        for j in range(1):

             DSC_All[i,j]=dsc[j+1]
             H95_All[i,j]=h95[j+1]
             VS_All[i,j]=vs[j+1]

    #pdb.set_trace()

    meanDSC = np.mean(DSC_All,0)
    meanH95 = np.mean(H95_All,0)
    meanVS = np.mean(VS_All,0)

    stdDSC = np.std(DSC_All, 0)
    stdH95 = np.std(H95_All, 0)
    stdVS = np.std(VS_All, 0)

    nameLog = args.path.split('/Images/')
    nameLog = nameLog[1].split('/')
    nameLog = nameLog[0]

    logging.basicConfig(filename='./logs/' + nameLog + args.suffix + '.log', level=logging.INFO)
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)

    logging.info('################  Results in 3D ################## ')
    logging.info(' Model NAME:  %s ', nameLog)
    logging.info('## TESTING ##  ')
    logging.info(' TOP--> DSC: %.4f (%.4f) ', meanDSC, stdDSC)
    logging.info(' TOP--> HD95: %.4f (%.4f) ', meanH95, stdH95)
    logging.info(' TOP--> VS: %.4f (%.4f) ', meanVS, stdVS)
    logging.info('################   ################## ')
    logging.info('################   ################## ')
    logging.info('################  INDIVIDUAL ################## ')


    for i_m in range(len(DSC_All)):
        logging.info(' %d   DSC: %.4f   HD95: %.4f,   VS: %.4f ', i_m, DSC_All[i_m], H95_All[i_m],VS_All[i_m])

    print(' TOP--> DSC: {:.4f} ({:.4f}) '.format(meanDSC[0], stdDSC[0]))
    print(' TOP--> HD95: {:.4f} ({:.4f}) '.format(meanH95[0], stdH95[0]))
    print(' TOP--> VS: {:.4f} ({:.4f}) '.format(meanVS[0], stdVS[0]))

def load_nii(imageFileName, printFileNames):
    if printFileNames == True:
        print (" ... Loading file: {}".format(imageFileName))

    img_proxy = nib.load(imageFileName)
    imageData = img_proxy.get_data()

    return (imageData, img_proxy)


def runEvaluation(argv):
    GT_Folder = './Results/GT/'+args.mode +'/Nifti/'
    CNN_Folder = args.path

    GT_names = getImageImageList(GT_Folder)
    CNN_names = getImageImageList(CNN_Folder)

    numClasses = 1
    DSC_All = np.zeros((len(GT_names),numClasses))
    H95_All = np.zeros((len(GT_names),numClasses))
    VS_All = np.zeros((len(GT_names),numClasses))

    #pdb.set_trace()
    for i in range(len(CNN_names)):
        #print(' {} '.format(CNN_names[i]))
        #testImage, resultImage = getImages(os.path.join(GT_Folder, CNN_names[i]), os.path.join(CNN_Folder, CNN_names[i]))

        [testImage, img_proxy] = load_nii(os.path.join(GT_Folder, CNN_names[i]), printFileNames=False)
        [resultImage, img_proxy] = load_nii(os.path.join(CNN_Folder, CNN_names[i]), printFileNames=False)

        for c_i in range(numClasses):
            label_GT = np.zeros(testImage.shape, dtype=np.int8)
            label_CNN = np.zeros(resultImage.shape, dtype=np.int8)
            #idx_GT = np.where(imageDataGT == c_i+1)
            idx_GT = np.where(testImage == 255)
            label_GT[idx_GT] = 1
            #idx_CNN = np.where(imageDataCNN == c_i+1)
            idx_CNN = np.where(resultImage == 255)
            label_CNN[idx_CNN] = 1

            DSC_All[i,c_i] = dc(label_GT,label_CNN)
            #HD[s_i,c_i] = hd(label_GT,label_CNN)
            H95_All[i,c_i] = hd95(label_GT,label_CNN)
            #ASSD[s_i,c_i] = assd(label_GT,label_CNN)

    #pdb.set_trace()

    meanDSC = np.mean(DSC_All,0)
    meanH95 = np.mean(H95_All,0)
    meanVS = np.mean(VS_All,0)

    stdDSC = np.std(DSC_All, 0)
    stdH95 = np.std(H95_All, 0)
    stdVS = np.std(VS_All, 0)

    nameLog = args.path.split('/Images/')
    nameLog = nameLog[1].split('/')
    nameLog = nameLog[0]

    logging.basicConfig(filename='./logs/' + nameLog + args.suffix + '.log', level=logging.INFO)
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)

    logging.info('################  Results in 3D ################## ')
    logging.info(' Model NAME:  %s ', nameLog)
    logging.info('## TESTING ##  ')
    logging.info(' TOP--> DSC: %.4f (%.4f) ', meanDSC, stdDSC)
    logging.info(' TOP--> HD95: %.4f (%.4f) ', meanH95, stdH95)
    logging.info(' TOP--> VS: %.4f (%.4f) ', meanVS, stdVS)
    logging.info('################   ################## ')
    logging.info('################   ################## ')
    logging.info('################  INDIVIDUAL ################## ')


    for i_m in range(len(DSC_All)):
        logging.info(' %d   DSC: %.4f   HD95: %.4f,   VS: %.4f ', i_m, DSC_All[i_m], H95_All[i_m],VS_All[i_m])

    print(' TOP--> DSC: {:.4f} ({:.4f}) '.format(meanDSC[0], stdDSC[0]))
    print(' TOP--> HD95: {:.4f} ({:.4f}) '.format(meanH95[0], stdH95[0]))
    print(' TOP--> VS: {:.4f} ({:.4f}) '.format(meanVS[0], stdVS[0]))

    #pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='val', type=str)
    parser.add_argument("--path", default="-", type=str)
    parser.add_argument("--suffix", default="", type=str)
    args = parser.parse_args()
    #runEvaluationITK(args)
    runEvaluation(args)


