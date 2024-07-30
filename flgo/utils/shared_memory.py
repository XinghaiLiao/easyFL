import multiprocessing
import os
import time
import uuid
import flgo.utils.fflow as fuf
import torch
import numpy as np
import torch.utils.data as tud
import sys
import pickle

def get_dict_size(obj, seen=None):
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for key, value in obj.items():
            size += sys.getsizeof(key) + get_dict_size(value, seen)
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            size += get_dict_size(item, seen)
    return size

TYPE_CANDIDATES = ['int', 'float', 'str', 'ndarray', 'Tensor', 'list', 'tuple', 'dict', 'int64', 'float64']

class TmpDataset(tud.Dataset):
    def __init__(self, sharable_data, types):
        super().__init__()
        data = []
        for i in range(len(types)):
            if isinstance(types[i], dict):
                etype = types[i]['etype']
                shapes = types[i]['shape']
                offsets = [0] + np.cumsum([np.prod(s) for s in shapes]).tolist()
                tmp_data = [sharable_data[i][offsets[k]:offsets[k + 1]].reshape(shapes[k]) for k in range(len(shapes))]
                if etype =='Tensor':
                    tmp_data = [torch.from_numpy(tdi) for tdi in tmp_data]
                elif etype =='list':
                    tmp_data = [tdi.tolist() for tdi in tmp_data]
                elif etype=='tuple':
                    tmp_data = [tuple(tdi.tolist()) for tdi in tmp_data]
                data.append(tmp_data)
            elif types[i]=='pickle':
                data.append(pickle.loads(sharable_data[i].tobytes()))
            else:
                if types[i] == 'ndarray':
                    data.append(sharable_data[i])
                elif types[i] == 'Tensor':
                    data.append(torch.from_numpy(sharable_data[i]))
                elif types[i] in ['int', 'float', 'str']:
                    data.append(sharable_data[i].tolist())
                elif types[i] in ['int64', 'float64']:
                    data.append(sharable_data[i])
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, i):
        return tuple(d[i] for d in self.data)

def _check_vector_shapes(vec_list):
    if not vec_list: return True
    reference_shape = vec_list[0].shape
    for tensor in vec_list[1:]:
        if tensor.shape != reference_shape:
            return False
    return True

def dataset2sharable(dataset):
    first_item = dataset[0]
    item_size = len(first_item)
    if not isinstance(first_item, tuple): first_item = tuple(first_item)
    element_types = [type(ei).__name__ if type(ei) not in TYPE_CANDIDATES else 'unknown' for ei in first_item]
    dataset_size = len(dataset)
    res = [[] for _ in range(item_size)]
    for i in range(dataset_size):
        for j in range(item_size):
            res[j].append(dataset[i][j])
    for j in range(item_size):
        if element_types[j] in ['int', 'float', 'str', 'int64', 'float64']:
            res[j] = np.array(res[j])
        elif element_types[j] == 'ndarray':
            if _check_vector_shapes(res[j]):
                res[j] = np.stack(res[j])
            else:
                shapes = [rjk.shape for rjk in res[j]]
                element_types[j] = {'etype': element_types[j], 'shape': shapes}
                res[j] = np.concat([rjk.reshape((-1, )) for rjk in res[j]])
        elif element_types[j] == 'Tensor':
            if _check_vector_shapes(res[j]):
                res[j] = torch.stack(res[j]).numpy()
            else:
                shapes = [tuple(rjk.shape) for rjk in res[j]]
                element_types[j] = {'etype': element_types[j], 'shape': shapes}
                res[j] = torch.cat([rjk.view(-1) for rjk in res[j]]).numpy()
        elif element_types[j] in ['list', 'tuple']:
            try:
                new_resj = [np.array(rjk) for rjk in res[j]]
                if _check_vector_shapes(new_resj):
                    new_resj = np.stack(new_resj)
                else:
                    shapes = [rjk.shape for rjk in new_resj]
                    element_types[j] = {'etype': element_types[j], 'shape': shapes}
                    new_resj = np.concat([rjk.reshape((-1, )) for rjk in new_resj])
                res[j] = new_resj
            except:
                res[j] = pickle.dumps(res[j])
                element_types[j] = 'pickle'
        else:
            res[j] = np.frombuffer(pickle.dumps(res[j]), dtype=np.uint8)
            element_types[j] = 'pickle'
    return res, element_types

def sharable2dataset(res, element_types):
    return TmpDataset(res, element_types)

def create_meta_for_dataset(sharable_data, name):
    shm_name_list = []
    shm_dtype_list = []
    shm_shape_list = []
    for i, sdi in enumerate(sharable_data):
        shm_name = name+f"{uuid.uuid4()}"
        shm_name_list.append(shm_name)
        shm_dtype_list.append(sdi.dtype.name)
        shm_shape_list.append(sdi.shape)
        dst = np.memmap(shm_name, mode='w+', dtype=sdi.dtype, shape=sdi.shape)
        np.copyto(dst, sdi)
    return shm_name_list, shm_dtype_list, shm_shape_list

def create_meta_for_task(task_data):
    task_meta = {}
    for party in task_data:
        task_meta[party] = {}
        for data_name in task_data[party]:
            data = task_data[party][data_name]
            if data is None: continue
            sharable_data, etypes = dataset2sharable(data)
            shm_name_list, shm_type_list, shm_shape_list= create_meta_for_dataset(sharable_data, "_".join([party, data_name]))
            task_meta[party][data_name] = {
                "name": shm_name_list,
                "dtype": shm_type_list,
                "shape": shm_shape_list,
                'element_type': etypes,
            }
    return task_meta

def load_dataset_from_meta(name, dtype, shape, element_type):
    sharable_data = []
    target_element_types = []
    for name_i, dtype_i, shape_i, element_type_i in zip(name, dtype, shape, element_type):
        np_array = np.memmap(name_i, mode='r+', dtype=dtype_i, shape=shape_i)
        sharable_data.append(np_array)
        target_element_types.append(element_type_i)
    return sharable2dataset(sharable_data, target_element_types)

def load_taskdata_from_meta(task_meta):
    task_data = {}
    for party in task_meta:
        task_data[party] = {}
        for data_name in task_meta[party]:
            task_data[party][data_name] = load_dataset_from_meta(**task_meta[party][data_name])
    return task_data



if __name__=='__main__':
    def worker(task_meta):
        pid = os.getpid()
        task_data = load_taskdata_from_meta(task_meta)
        selected_party = 'server'
        dataset = task_data[selected_party]['test']
        dataloader = tud.DataLoader(dataset, batch_size=2)
        for batch_id, batch_data in enumerate(dataloader):
            x, y = batch_data
            print(f"Process {pid} Xid = {id(x.numpy())} {pid} Y = {id(y.numpy())}")
            if batch_id >= 3: break
        return True
    class TenDataset(tud.Dataset):
        def __init__(self, n: int = 5, l: int = 3, r: int = 8):
            self.n = n
            self.x = [torch.rand((torch.randint(l, r, (1,)).item())) for i in range(n)]
            self.y = torch.randint(0, 9, (n,)).long()

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            return self.x[i], self.y[i]


    class NPDataset(tud.Dataset):
        def __init__(self, n: int = 5, l: int = 3, r: int = 8):
            self.n = n
            self.x = [torch.rand((torch.randint(l, r, (1,)).item())).numpy() for i in range(n)]
            self.y = torch.randint(0, 9, (n,)).long().numpy()

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            return self.x[i], self.y[i]


    class DictData(tud.Dataset):
        def __init__(self, n: int = 5, l: int = 3, r: int = 8):
            self.n = n
            self.x = [{'x': torch.rand((torch.randint(l, r, (1,)).item())).numpy()} for i in range(n)]
            self.y = torch.randint(0, 9, (n,)).long().numpy()

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            return self.x[i], self.y[i]
    NUM_WORKERS = 3
    # task = './my_task'
    DATA = NPDataset
    # task_data = fuf.load_task_data(task)
    task_data = {'server':{'test': DATA(), 'val':None}, 'Client01':{'train':DATA(), 'val':DATA(), 'test':None}}
    # DEBUG: create_meta_for_task
    task_meta = {}
    for party in task_data:
        task_meta[party] = {}
        for data_name in task_data[party]:
            data = task_data[party][data_name]
            if data is None: continue
            sharable_data, etypes = dataset2sharable(data)
            new_data = sharable2dataset(sharable_data, etypes)
            print(f"Consistency for {party} {data_name}: {all([all(new_data[i][0] == data[i][0]) for i in range(len(data))])}")
            # shm_name_list, shm_type_list, shm_shape_list= create_meta_for_dataset(sharable_data, "_".join([party, data_name]))
            # task_meta[party][data_name] = {
            #     "name": shm_name_list,
            #     "dtype": shm_type_list,
            #     "shape": shm_shape_list,
            #     'element_type': etypes,
            # }
    # task_meta = create_meta_for_task(task_data)
    # worker(task_meta)
    # pros = []
    # for i in range(NUM_WORKERS):
    #     p = multiprocessing.Process(target=worker,  args=(task_meta, ))
    #     pros.append(p)
    # for p in pros: p.start()
    # for p in pros: p.join()
