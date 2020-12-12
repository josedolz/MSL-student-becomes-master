import medicalDataLoader
import argparse
import random
import torchvision.transforms.functional as TF
import logging

from torch.utils.data import DataLoader
from torchvision import transforms
from progressBar import printProgressBar
from utils import *
from UNet_Base import *
from random import random, randint
from PIL import Image, ImageOps


def runSingleModel(args, val_loader, test_loader, name):
    # Initialize
    num_classes = 2
    modelName = args.modelName

    # initial_kernels = 64
    print("~~~~~~~~~~~ Creating the UNet model ~~~~~~~~~~")
    net = UNet_Mixed(num_classes)
    print(" Model Name: {}".format(args.modelName))
    net.load_state_dict(torch.load(args.modelName))
    print("--------model restored--------")

    net.eval()

    if torch.cuda.is_available():
        net.cuda()

    print("~~~~~~~~~~~ Starting the testing ~~~~~~~~~~")

    dsc, dscW = inferenceTesting(net, val_loader, modelName, args.modelType, 'val', False)
    dsct, dscWt = inferenceTesting(net, test_loader, modelName, args.modelType, 'test', False)
    print("###                                                       ###")
    print("###  [VAL]  Best Dice (top): {:.4f} (bot) {:.4f}  ###".format(dsc, dscW))
    print("###  [TEST] Best Dice (top): {:.4f} (bot) {:.4f}  ###".format(dsct, dscWt))
    print("###                                                       ###")
    print('##########  3D ###########')
    print('Reconstructing images into volumes....')

    modelType = 1
    reconstruct3D(modelName, modelType, 'val', False)
    reconstruct3D(modelName, modelType, 'test', False)
    modelType = 2
    reconstruct3D(modelName, modelType, 'val', False)
    reconstruct3D(modelName, modelType, 'test', False)

    modelType = 1
    dscValTop, hdValTop = evaluate(modelName, 'val', modelType)
    modelType = 2
    dscValBot, hdValBot = evaluate(modelName, 'val', modelType)
    modelType = 1
    dscTestTop, hdTestTop = evaluate(modelName, 'test', modelType)
    modelType = 2
    dscTestBot, hdTestBot = evaluate(modelName, 'test', modelType)

    print(' #########  Results in 3D ####### ')
    print(' ## VALIDATION ## ')
    print(' TOP--> DSC: {:.4f} ({:.4f}) BOTTOM--> DSC: {:.4f} ({:.4f})'.format(dscValTop.mean(),
                                                                               dscValTop.std(),
                                                                               dscValBot.mean(),
                                                                               dscValBot.std()))
    print(' TOP--> HD95: {:.4f} ({:.4f}) BOTTOM--> HD95: {:.4f} ({:.4f})'.format(hdValTop.mean(),
                                                                               hdValTop.std(),
                                                                               hdValBot.mean(),
                                                                               hdValBot.std()))

    print(' ## TESTING ## ')
    print(' TOP--> DSC: {:.4f} ({:.4f}) BOTTOM--> DSC: {:.4f} ({:.4f})'.format(dscTestTop.mean(),
                                                                               dscTestTop.std(),
                                                                               dscTestBot.mean(),
                                                                               dscTestBot.std()))
    print(' TOP--> HD95: {:.4f} ({:.4f}) BOTTOM--> HD95: {:.4f} ({:.4f})'.format(hdTestTop.mean(),
                                                                               hdTestTop.std(),
                                                                               hdTestBot.mean(),
                                                                               hdTestBot.std()))
    metrics =[]
    metrics.append(dscValTop.mean())
    metrics.append(dscValTop.std())
    metrics.append(dscValBot.mean())
    metrics.append(dscValBot.std())
    metrics.append(hdValTop.mean())
    metrics.append(hdValTop.std())
    metrics.append(hdValBot.mean())
    metrics.append(hdValBot.std())

    metrics.append(dscTestTop.mean())
    metrics.append(dscTestTop.std())
    metrics.append(dscTestBot.mean())
    metrics.append(dscTestBot.std())
    metrics.append(hdTestTop.mean())
    metrics.append(hdTestTop.std())
    metrics.append(hdTestBot.mean())
    metrics.append(hdTestBot.std())

    return metrics

def runTesting(args):
    print('-' * 40)
    print('~~~~~~~~  Starting the training... ~~~~~~')
    print('-' * 40)


    batch_size_val = 1
    root_dir = '../Preliminary/ACDC-2D-All/'+str(args.numSubj)+'/'

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    val_set = medicalDataLoader.MedicalImageDataset('val',
                                                     root_dir,
                                                     transform=transform,
                                                     mask_transform=mask_transform,
                                                     equalize=False)

    val_loader = DataLoader(val_set,
                             batch_size=batch_size_val,
                             num_workers=5,
                             shuffle=False)

    test_set = medicalDataLoader.MedicalImageDataset('test',
                                                    root_dir,
                                                    transform=transform,
                                                    mask_transform=mask_transform,
                                                    equalize=False)

    test_loader = DataLoader(test_set,
                            batch_size=batch_size_val,
                            num_workers=5,
                            shuffle=False)

    ###### Check now how split the string ####
    folderName = args.modelName
    nameLog = folderName.split('/')

    nameLog = nameLog[2]

    if not os.path.exists('./logs/'):
        os.makedirs('./logs/')

    logging.basicConfig(filename='./logs/'+nameLog+'.log', level=logging.INFO)
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)

    metrics = runSingleModel(args, val_loader, test_loader, args.modelName)

    logging.info('################  Results in 3D ################## ')
    logging.info(' Model NAME:  %s ', args.modelName)
    logging.info('## VALIDATION ##  ')
    logging.info(' TOP--> DSC: %.4f (%.4f) BOTTOM--> DSC: %.4f (%.4f)', metrics[0], metrics[1], metrics[2], metrics[3])
    logging.info(' TOP--> HD95: %.4f (%.4f) BOTTOM--> HD95: %.4f (%.4f)', metrics[4], metrics[5], metrics[6],metrics[7])
    logging.info('## TESTING ##  ')
    logging.info(' TOP--> DSC: %.4f (%.4f) BOTTOM--> DSC: %.4f (%.4f)', metrics[8], metrics[9], metrics[10], metrics[11])
    logging.info(' TOP--> HD95: %.4f (%.4f) BOTTOM--> HD95: %.4f (%.4f)', metrics[12], metrics[13], metrics[14], metrics[15])


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--modelName",default="Test_Model",type=str)
    parser.add_argument("--numBranches", default=2, type=int)
    parser.add_argument("--modelType", default=2, type=int)
    parser.add_argument("--numSubj", default=3, type=int)
    args=parser.parse_args()
    runTesting(args)
