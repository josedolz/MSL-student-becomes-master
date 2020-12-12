import medicalDataLoader
import argparse
import random
import warnings
import os
import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from progressBar import printProgressBar
from utils import *
from UNet_Base import *
from PIL import Image, ImageOps

warnings.filterwarnings("ignore")

seed_value= 0

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value

os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set `pytorch` pseudo-random generator at a fixed value
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

##### LOSSES  ######
def partialCE_loss(pred,target):
    eps = 1e-20
    loss = -(torch.log(pred[:, 1, :, :] + eps) * target).sum()/(target.sum()+eps)
    return loss

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.mean()
        return b


def runTraining(args):
    print('-' * 40)
    print('~~~~~~~~  Starting the training... ~~~~~~')
    print('-' * 40)

    batch_size = args.batch_size
    batch_size_val = 1
    lr = args.lr

    epoch = args.epochs
    times_weaklabels = args.xtimes_weak # indicates how much larger the batch size for weakly labels is

    root_dir = '../OmniSupervision/ACDC-2D-All/'+str(args.numSub)+'/'

    print(' Dataset: {} '.format(root_dir))

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Create the data loaders
    train_set_full = medicalDataLoader.MedicalImageDataset('trainFull',
                                                      root_dir,
                                                      transform=transform,
                                                      mask_transform=mask_transform,
                                                      augment=False,
                                                      equalize=False)

    train_loader_full = DataLoader(train_set_full,
                              batch_size=batch_size,
                              worker_init_fn=np.random.seed(0),
                              num_workers=0,
                              shuffle=True)

    train_set_weak = medicalDataLoader.MedicalImageDataset('trainSemi',
                                                           root_dir,
                                                           transform=transform,
                                                           mask_transform=mask_transform,
                                                           augment=False,
                                                           equalize=False)

    train_loader_weak = DataLoader(train_set_weak,
                                   batch_size=times_weaklabels*batch_size,
                                   worker_init_fn=np.random.seed(0),
                                   num_workers=0,
                                   shuffle=True)

    val_set = medicalDataLoader.MedicalImageDataset('val',
                                                    root_dir,
                                                    transform=transform,
                                                    mask_transform=mask_transform,
                                                    equalize=False)

    val_loader = DataLoader(val_set,
                            batch_size=batch_size_val,
                            worker_init_fn=np.random.seed(0),
                            num_workers=0,
                            shuffle=False)

    test_set = medicalDataLoader.MedicalImageDataset('test',
                                                    root_dir,
                                                    transform=transform,
                                                    mask_transform=mask_transform,
                                                    equalize=False)

    test_loader = DataLoader(test_set,
                            batch_size=batch_size_val,
                            worker_init_fn=np.random.seed(0),
                            num_workers=0,
                            shuffle=False)
                                                                    
    # Initialize
    num_classes = 2

    print("~~~~~~~~~~~ Creating the UNet model ~~~~~~~~~~")
    net = UNet_Mixed(num_classes)
    print(" Model Name: {}".format(args.modelName))
    print("Total params: {0:,}".format(sum(p.numel() for p in net.parameters() if p.requires_grad)))

    # Initialize model weights
    net.apply(weights_init)


    softMax = nn.Softmax()
    CE_loss = nn.CrossEntropyLoss()
    KD_loss = nn.KLDivLoss()
    entropy_loss = HLoss()
    l2_loss = nn.MSELoss()

    if torch.cuda.is_available():
        net.cuda()
        softMax.cuda()
        CE_loss.cuda()
        KD_loss.cuda()
        entropy_loss.cuda()
        l2_loss.cuda()

    '''try:
        net = torch.load('./model/Best.pkl')
        print("--------model restored--------")
    except:
        print("--------model not restored--------")
        pass'''

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.99))

    ### To save statistics ####
    lossTotalTraining = []
    DSC_Fully = []
    DSC_Weakly = []
    DSC_Weakly_train = []
    DSC_Weakly_test = []
    DSC_Full_test = []
    BestDice = .0
    BestDice_bottom = .0
    BestDiceTest_top = .0
    BestDiceTest_bottom = .0
    BestEpoch = 0

    print("~~~~~~~~~~~ Starting the training ~~~~~~~~~~")

    if args.modelType == 1:
        modelName = 'KL' + '_Subj_' + str(args.numSub) + '_KL' + str(args.weight_KL) + '_T' + str(
            args.temperature) + '_x_' + str(args.xtimes_weak) + '_w_' + str(args.weight_Weakly) + '_Run_' + args.run
    else:
        modelName = 'Omni' + '_Subj_' + str(args.numSub) + '_KL' + str(args.weight_KL) + '_T' + str(
            args.temperature) + '_wEnt_' + str(args.weight_Entropy) + '_x_' + str(args.xtimes_weak) + '_w_' + str(
            args.weight_Weakly) + '_Run_' + args.run

    directory = 'Results/Statistics/' + modelName

    if os.path.exists(directory):
        print(' ############### [WARNING] The folder already exists!!!! ############### ')
        pdb.set_trace()
    else:
        os.makedirs(directory)

    for i in range(epoch):
        net.train()
        lossEpoch = []
        DSCEpoch = []
        DSCEpoch_w = []
        num_batches = len(train_loader_full)
        for j, dataF in enumerate(train_loader_full):
            imageF, labelsF, labelsF_weak, img_namesF = dataF

            # Get the next iteration for weakly/partially labeled data(not the cleanest solution)
            imageW, labelsW, img_namesW = next(iter(train_loader_weak))

            # prevent batchnorm error for batch of size 1
            if imageF.size(0) != batch_size:
                continue

            idx = np.where(labelsF < 1.0)
            labelsF[idx] = 0.0


            optimizer.zero_grad()
            inputF = to_var(imageF)
            inputW = to_var(imageW)

            labelsF = to_var(labelsF)
            labelsF_weak = to_var(labelsF_weak)
            labelsW = to_var(labelsW)

            ################### Train ###################
            net.zero_grad()

            segmentation_predictionF, segmentation_predictionW = net(torch.cat((inputF,inputF,inputW),dim=0))

            ### Let's get the segmentation of each branch
            segmentation_predictionF = segmentation_predictionF[:batch_size]
            segmentation_predictionW_fromFull = segmentation_predictionW[batch_size:2*batch_size]
            segmentation_predictionW_fromWeak = segmentation_predictionW[2*batch_size:]

            # Losses
            # 1 - full supervision
            Segmentation_class_F = getTargetSegmentation(labelsF)
            CE_loss_F = CE_loss(segmentation_predictionF, Segmentation_class_F)

            predClass_yF = softMax(segmentation_predictionF)
            pred_F = predToSegmentation(predClass_yF)

            dscF = computeDSC(pred_F,labelsF)

            # 2 - partial CE (i.e., only on labeled pixels)

            # 2.1 - On labels coming from the whole labeled dataset
            Segmentation_class_W_fromFull = getTargetSegmentation(labelsF_weak)
            predClass_yFW = softMax(segmentation_predictionW_fromFull)
            my_partialCE_loss_F = partialCE_loss(predClass_yFW,Segmentation_class_W_fromFull)

            # 2.2 - On labels coming from the WEAKLY labeled dataset
            Segmentation_class_W = getTargetSegmentation(labelsW)
            predClass_yWW = softMax(segmentation_predictionW_fromWeak)
            my_partialCE_loss_W = partialCE_loss(predClass_yWW, Segmentation_class_W)

            # Compute DSC on training images (fully labeled only - this does not affect the training)
            pred_FW = predToSegmentation(predClass_yFW)
            dscFW = computeDSC(pred_FW, labelsF)

            # 3 - Distill knowledge between upper branch and lower branch (Re-check this)
            T = args.temperature
            alpha = 0.5
            dist_loss = KD_loss(F.log_softmax(predClass_yF / T, dim=1), F.softmax(predClass_yFW / T, dim=1)) * (alpha * T * T)

            # 4 - Entropy loss
            eloss = entropy_loss(torch.cat((segmentation_predictionW_fromFull,segmentation_predictionW),dim=0))

            if args.modelType ==1:
                lossTotal = CE_loss_F + args.weight_Weakly*(my_partialCE_loss_F + my_partialCE_loss_W) + args.weight_KL * dist_loss
            else:
                lossTotal = CE_loss_F + args.weight_Weakly*(my_partialCE_loss_F + my_partialCE_loss_W) + args.weight_KL * dist_loss + args.weight_Entropy * eloss

            lossTotal.backward()

            optimizer.step()

            lossEpoch.append(lossTotal.cpu().data.numpy())
            DSCEpoch.append(dscF)
            DSCEpoch_w.append(dscFW)

            printProgressBar(j + 1, num_batches,
                             prefix="[Training] Epoch: {} ".format(i),
                             length=15,
                             suffix=" Mean Dice: (F) {:.4f}, (W) {:.4f}".format(dscF,dscFW))


        DSC_Weakly_train.append(dscFW.mean())
        lossEpoch = np.asarray(lossEpoch)
        lossEpoch = lossEpoch.mean()

        DSCEpoch = np.asarray(DSCEpoch)
        DSCEpoch = DSCEpoch.mean()

        lossTotalTraining.append(lossEpoch)

        printProgressBar(num_batches, num_batches,
                             done="[Training] Epoch: {}, LossG: {:.4f}, DSC: {:.4f}".format(i,lossEpoch,DSCEpoch))

        # Compute and save statistics
        dsc, dscW = inferenceTraining(net, val_loader, modelName, 'val', False)

        DSC_Fully.append(dsc)
        DSC_Weakly.append(dscW)

        np.save(os.path.join(directory, 'Losses.npy'), lossTotalTraining)
        
        np.save(os.path.join(directory, 'DSC_Fully_Val.npy'), DSC_Fully)
        np.save(os.path.join(directory, 'DSC_Weakly_Val.npy'), DSC_Weakly)
        np.save(os.path.join(directory, 'DSC_Weakly_Train.npy'), DSC_Weakly_train)

        currentDice = dscW

        print("[val] DSC: (F): {:.4f} (W): {:.4f}  ".format(dsc,dscW))

        if currentDice > BestDice:
            BestDice = currentDice
            BestDice_bottom = dscW
            BestEpoch = i

            if currentDice > 0.4:
                print("###   Evaluating in Test  ###")
                dsct, dscWt = inferenceTraining(net, test_loader, modelName, 'test', False)
                BestDiceTest_top = dsct
                BestDiceTest_bottom = dscWt
                if not os.path.exists('./models/'+modelName):
                    os.makedirs('./models/'+modelName)

                torch.save(net.state_dict(), './models/'+modelName+'/'+str(i)+'_Epoch')
                print(" [TEST] Dice (Top): {:.4f} Dice (Bottom) {:.4f}  ###".format(dsct, dscWt))


        print("###                                                       ###")
        print("###  [VAL]  Best Dice (t): {:.4f} (bot) {:.4f} at epoch {}  ###".format(BestDice, BestDice_bottom, BestEpoch))
        print("###  [TEST] Best Dice (t): {:.4f} (bot) {:.4f}  ###".format(BestDiceTest_top, BestDiceTest_bottom))
        print("###                                                       ###")

        DSC_Weakly_test.append(BestDiceTest_bottom)
        DSC_Full_test.append(BestDiceTest_top)

        np.save(os.path.join(directory, 'DSC_Weakly_test.npy'), DSC_Weakly_test)
        np.save(os.path.join(directory, 'DSC_Full_test.npy'), DSC_Full_test)

        if i % (BestEpoch + 100) == 0 and i>0:
            for param_group in optimizer.param_groups:
                lr = lr*0.5
                param_group['lr'] = lr
                print(' ----------  New learning Rate: {}'.format(lr))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--arch_type",default=1,type=int)
    parser.add_argument("--modelName",default="Test_Model",type=str)
    parser.add_argument('--batch_size',default=8,type=int)
    parser.add_argument('--epochs',default=1000,type=int)
    parser.add_argument('--lr',default=0.0001,type=float)
    parser.add_argument('--xtimes_weak',default=1,type=int)
    parser.add_argument('--numSub',default=3,type=int)
    parser.add_argument('--weight_KL',default=20,type=int)
    parser.add_argument('--weight_Entropy',default=1,type=int)
    parser.add_argument('--weight_Weakly',default=0.1,type=float)
    parser.add_argument('--temperature',default=2,type=float)
    parser.add_argument('--modelType',default=4,type=float)
    parser.add_argument('--run', default="A", type=str)
    args=parser.parse_args()
    runTraining(args)
