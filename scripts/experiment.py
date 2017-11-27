#from lasso import Lasso
from discretization import Discretization
from pertubation import conditional_perturbQuant,perturbQuali
from sklearn import svm
from sklearn.linear_model import Lasso
import numpy as np
import copy
import itertools
import pandas as pd

flattening_binaryObs = lambda slices: list(
itertools.chain.from_iterable(slices))

def selectSlice_randomly(slices):
    randomN = np.random.random()
    slicesLen= [len(slc) for slc in slices]
    sliceSumLen = sum(slicesLen)
    slicesProb = [sliceLen/sliceSumLen  for sliceLen in slicesLen]
    slicesCum = np.cumsum(slicesProb)
    bools = slicesCum > randomN
    positions = np.where(bools == True)[0]
    featureIdx = positions[0]
    return featureIdx, slices[featureIdx]

def gen_binaryData(slices,nData_toGen=10):
    _Z = []
    slice_vals = [x[0] for x in slices]
    slice_types = [x[1] for x in slices]
    generatedBinary_obs = copy.deepcopy(slice_vals)

    for genIdx in range(nData_toGen):
        idx, sliceToPerturb = selectSlice_randomly(generatedBinary_obs)
        SliceType = slice_types[idx]

        if SliceType == "quant":
            pert_slice = conditional_perturbQuant(sliceToPerturb)
        else:
            pert_slice = perturbQuali(sliceToPerturb)

        new_obs_slices = copy.deepcopy(slice_vals)
        new_obs_slices[idx] = np.array(pert_slice)
        new_obs = [val for l in new_obs_slices
                for val in l]
        new_obs = np.array(new_obs)
        _Z.append(new_obs)
    return np.array(_Z)

def decode_binaryData(Z,obj):
    _X = []
    for obs in Z:
        #flat_binaryObs = np.array(flattening_binaryObs(obs))
        decoded_obs = obj.inverse_transform(obs)
        _X.append(decoded_obs)
    return _X

def obs_importance(obs,generatedData):
    ''' given the obs of reference and a generatedData
        returns the importance of the generatedData
        in relation to the obs
    '''
    sqrDistance = np.sqrt( np.sum( (obs -generatedData)**2 ) )
    importance = np.exp(-sqrDistance)
    return importance

def explain(disc_catX,disc_contX,real_features,real_y,obj,ft_types,Allcat_slices):
    classifier = svm.SVC(gamma=0.001)
    classifier.fit(real_features,real_y)
    sintetic_obs = list()

    for obs_idx,disc_cont_obs in enumerate(disc_contX):
        cat_slices = [(observations[0][obs_idx],"categoricals") for observations in Allcat_slices]
        cont_slices = [(slic,"quant") for slic in obj.get_slices_from_discretized_sample(disc_cont_obs)]
        contBins = 0
        for cSlic,type_ in cont_slices:
            contBins+= len(cSlic)
        slices = [tpl for type_slices in [cont_slices,cat_slices]
                          for tpl in type_slices ]
        Z_prime = gen_binaryData(slices)
        Zcont_prime=np.array([genObs[:contBins] for genObs in Z_prime])
        Zcat_prime=np.array([genObs[contBins:] for genObs in Z_prime])
        Xcont_prime = np.array(decode_binaryData(Zcont_prime,obj))
        X_prime = np.concatenate((Xcont_prime,Zcat_prime),axis=1)
        importances = [ obs_importance(real_features[obs_idx],x) for x in X_prime ]
        sintetic_obs.append(X_prime)
        Y_prime = classifier.predict(X_prime)

        ZwithImportance = [ z*np.sqrt(importances[pos]) for pos,z in enumerate(Z_prime)]
        YwithImportance = [ y*np.sqrt(importances[pos]) for pos,y in enumerate(Y_prime)]
        explainer = Lasso(alpha=0.001)
        explainer.fit(ZwithImportance,YwithImportance)

    sintetic_dataset = np.concatenate(sintetic_obs, axis=0)

    print('sintetic dataset has shape {0}'.format(sintetic_dataset.shape))
    Y_prime = classifier.predict(sintetic_dataset)
    from collections import Counter
    c = Counter(Y_prime)
    print('classifying sintetic observations...')
    print(c)
def loadAustralian():
	X,y =[],[]
	with open("data/australian.dat.txt") as f:
		for line in f:
			data = [float(val) for val in line.replace("\n","").split(" ")]
			X.append(data[:-1])
			y.append(data[-1])
	return np.array(X),np.array(y)

def loadNewsGroup():
	from sklearn.datasets import fetch_20newsgroups
	cats = ['alt.atheism', 'sci.space']
	newsgroups_train = fetch_20newsgroups(subset='train', categories=cats)
from sklearn.datasets import load_iris

def experimentAustralian():
    aus_X,aus_y=loadAustralian() 
    continousFt = [1,2,6]
    ft_types = {"continous":continousFt,"categoricals":[ idx for idx in range(len(aus_X[0])) if idx not in continousFt]}
    obj = Discretization(aus_X[:,ft_types["continous"]], {0:7,1:3, 2:10 })
    continousX_t = obj.fit_transform()
    categoricalX_t = np.ones((len(continousX_t), 1))
    cat_slices = []
    for ftIdx in ft_types["categoricals"]:
        categoricals = pd.get_dummies(aus_X[:,ftIdx]).values
        aus_X = np.concatenate((aus_X,categoricals),axis=1)
        cat_slices.append((categoricals,"categoricals"))
        categoricalX_t = np.concatenate((categoricalX_t,categoricals),axis=1)
    categoricalX_t = categoricalX_t[:,1:]
    aus_X = np.delete(aus_X,ft_types["categoricals"],axis=1)
    explain(categoricalX_t,continousX_t,aus_X,aus_y,obj,ft_types,cat_slices)

experimentAustralian()
