from discretization import Discretization
from pertubation import conditional_perturbQuant, perturbQuali
from sklearn.linear_model import Lasso
import numpy as np
import copy
import itertools
import matplotlib.pyplot as plt

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
            pert_slice = perturbQuali(slice_)

        new_obs = copy.deepcopy(slice_vals)
        new_obs[idx] = np.array(pert_slice)
        _Z.append(new_obs)
    return _Z

def decode_binaryData(Z):
    _X = []
    for obs in Z:
        flat_binaryObs = np.array(flattening_binaryObs(obs))
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
np.random.shuffle(sample)
sintetic_obs = list()
bin_obs = list()

for obs in sample:
    slices = [(val,"quant") for val in obj.get_slices_from_discretized_sample(obs)]
    Z_prime = np.array(gen_binaryData(slices))
    bin_obs.append([np.array(flattening_binaryObs(z)) for z in Z_prime])
    X_prime = np.array(decode_binaryData(Z_prime))
    sintetic_obs.append(X_prime)

sintetic_dataset = np.concatenate(sintetic_obs, axis=0)
bin_dataset = np.concatenate(bin_obs, axis=0)
print('sintetic dataset has shape {0}'.format(sintetic_dataset.shape))

Y_prime = classifier.predict(sintetic_dataset)
from collections import Counter
c = Counter(Y_prime)

print('classifying sintetic observations...')
print(c)
explainer = Lasso(alpha=0.05, fit_intercept=True)
print('explaining...')

explainer.fit(bin_dataset, Y_prime)
pred = explainer.predict(bin_dataset)

idx = np.random.randint(0,bin_dataset.shape[0])
obs = bin_dataset[idx]
fig, ax = plt.subplots()
ax.set_xticks(np.arange(100))
plt.bar(np.arange(obs.shape[0]), explainer.coef_)
plt.title('observation {0}, y = {1}, y_hat = {2}'.format(idx, Y_prime[idx],
    explainer.predict(obs.reshape(1,-1))))
plt.grid()
plt.show()
