import os.path

import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler


def load_data_from_file(file_path, file_name, input_dim, sample_per_file, sample_start, use_channel):
    x = np.empty((0, input_dim))
    matlab_file = scipy.io.loadmat(file_path)
    keys = [key for key in matlab_file if key.startswith(file_name[0:8])]
    assert len(keys) > 0
    array_key = keys[0]
    array = matlab_file[array_key]
    for i in range(sample_per_file):
        data = array[sample_start + i * input_dim:sample_start + (i + 1) * input_dim, use_channel].reshape(1, -1)
        x = np.concatenate((x, np.array(data)))
    return x


def load_domain(speed, load, input_dim, sample_per_file, sample_start, use_channel):
    x = np.empty((0, input_dim))
    y = np.empty(0)
    clazz = ["0A", "1A", "2A", "3A", "4A", "5A", "6A"]
    for c in clazz:
        file_name = f"C{c}_{speed}_{load}"
        filepath = f"D:/DIRG dataset/{file_name}_1.mat"
        if os.path.exists(filepath):
            print(f"load {file_name}")
            data = load_data_from_file(filepath, file_name, input_dim, sample_per_file, sample_start, use_channel)
            if c == "0A":
                l = 0
            if c == "1A":
                l = 1
            if c == "2A":
                l=2
            if c == "3A":
                l=3
            if c == "4A":
                l = 4
            if  c == "5A":
                l=5
            if c == "6A":
                l=6
            label = [l] * data.shape[0]
            x = np.concatenate((x, data))
            y = np.concatenate((y, np.array(label)))
        else:
            print(f"file:{file_name} does not exist!")
            continue
    return x, y


def save_domains(sample_size, sample_per_file, sample_start, use_channel):
    domain_idx = 0
    for speed in ["100", "200", "300", "400", "500"]:
        for load in ["000", "500", "700", "900"]:
            x = np.empty((0, sample_size))
            y = np.empty(0)
            data, label = load_domain(speed, load, sample_size, sample_per_file, sample_start, use_channel)
            x = np.concatenate((x, data))
            y = np.concatenate((y, label))
            if x.shape[0]>0:
                np.save(f"DIRG/domain_{domain_idx}_x.npy", x)
                np.save(f"DIRG/domain_{domain_idx}_y.npy", y)
                domain_idx += 1


#save_domains(512,1000,0,2)

def load_domains(domain_nums, input_dim):
    x = np.empty((0, input_dim))
    y = np.empty(0)
    domains = []
    domain=0
    scaler = StandardScaler()
    for domain_idx in domain_nums:
        #data = np.load(f"data/DIRG/domain_{domain_idx}_x.npy")
        #label = np.load(f"data/DIRG/domain_{domain_idx}_y.npy")
        data = np.load(f"DIRG/domain_{domain_idx}_x.npy")
        label = np.load(f"DIRG/domain_{domain_idx}_y.npy")


        x = np.concatenate((x, data))
        y = np.concatenate((y, label))
        [domains.append(domain) for _ in range(data.shape[0])]
        domain+=1
    x = scaler.fit_transform(x)
    return {"data": x, "label": y, "domain": np.array(domains)}


# d = load_domains([1], 512)
# print(np.unique(d["label"], return_counts=True))
# print(np.unique(d["data"], return_counts=True))
