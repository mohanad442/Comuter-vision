import os 
import scipy.io as sio

def loadData(name):
    if name == 'IP':
        data = sio.loadmat(os.path.join("Final_Version", 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join("Final_Version", 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SA':
        data = sio.loadmat('Salinas_corrected.mat')['salinas_corrected']
        labels = sio.loadmat('Salinas_gt.mat')['salinas_gt']
    elif name == 'PU':
        data = sio.loadmat(os.path.join("Final_Version", 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join("Final_Version", 'PaviaU_gt.mat'))['paviaU_gt']
    return data, labels