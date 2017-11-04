from discretization import Discretization
from pertubation import conditional_perturbQuant,perturbQuali
#from LIME import *
from sklearn.linear_model import Lasso
import numpy as np
import copy

def flattening_binaryObs(slices):
    flat_binaryObs= []
    for ft in [ft_values for ft_values,ft_type in slices]:
        flat_binaryObs.extend(ft)    
    flat_binaryObs = np.array(flat_binaryObs)
    return flat_binaryObs

def selectSlice_randomly(slices):
        randomN = np.random.random()
        slicesLen= [len(slc) for slc in slices ]
        sliceSumLen = sum(slicesLen)
        slicesProb = [sliceLen/sliceSumLen  for sliceLen in slicesLen ]
        slicesCum = np.cumsum(slicesProb)
        bools = slicesCum < randomN
        positions = np.where(bools == True )[0]
        if len(positions) > 0:
            featureIdx = positions[-1]
        else:
            featureIdx = 0
        return slices[featureIdx]

def gen_binaryData(slices,nData_toGen=10):
    _Z = []
    generatedBinary_obs = copy.copy(slices)
    for genIdx in range(nData_toGen):
        sliceToPerturb,SliceType=  selectSlice_randomly(generatedBinary_obs)
        if SliceType == "quant":
            conditional_perturbQuant(sliceToPerturb)
        else:
            perturbQuali(slice_)
        _Z.append(generatedBinary_obs)
    return _Z

def decode_binaryData(Z):
    _X = []
    for obs in Z:
        flat_binaryObs = flattening_binaryObs(obs)
        decoded_obs = obj.inverse_transform(flat_binaryObs)
        _X.append(decoded_obs)
    return _X




from sklearn.datasets import load_iris
data = load_iris()
obj = Discretization(data.data, {0:7,1:3, 2:10, 3:4})
X_t = obj.fit_transform()

from sklearn import svm
classifier = svm.SVC(gamma=0.001)
classifier.fit(data.data,data.target)

sample = X_t[:,:]
for obs in sample:
    slices = [(val,"quant") for val in obj.get_slices_from_discretized_sample(obs)]
    Z_prime = gen_binaryData(slices)
    X_prime = decode_binaryData(Z_prime)
    Y_prime = classifier.predict(X_prime)
    print(X_prime)
    #explainer = Lasso.fit(X_prime,Y_prime)
    break
	#y = model.predict(decodedBinaryData)
'''data.data[65,:]
array([ 6.7,  3.1,  4.4,  1.4])
obj.inverse_transform(sample),array([6.66863463,3.054,4.28622542,1.38881982])'''

