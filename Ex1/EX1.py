# Team name: Toivo Wuoti

import csv

from sklearn import neighbors
import pickle
import numpy as np

def read_data():

    with open("training_x.dat", 'rb') as pickleFile:
        x_tr = pickle.load(pickleFile)

    with open("training_y.dat", 'rb') as pickleFile:
        y_tr = pickle.load(pickleFile)

    with open("validation_x.dat", 'rb') as pickleFile:
        x_val = pickle.load(pickleFile)

    for i in range(len(x_tr)):
        x_tr[i] = x_tr[i][:,:,0].flatten()

    for i in range(len(x_val)):
        x_val[i] = x_val[i][:,:,0].flatten()


    return x_tr, y_tr, x_val


x_tr, y_tr, x_val = read_data()

clf = neighbors.KNeighborsClassifier(n_neighbors=1, algorithm="kd_tree")
clf.fit(x_tr, y_tr)
y_pred = clf.predict(x_val)


results = np.vstack( (np.arange(1, len(y_pred)+1), y_pred) ).T
np.savetxt("results.csv", results, delimiter=",", header="Id,Class", fmt="%s", comments="")