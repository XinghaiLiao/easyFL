import os
from .config import train_data # config必须包含train_data
try:
    from .config import test_data # config可选包含test_data和val_data
except:
    test_data = None
try:
    from .config import val_data
except:
    val_data = None
try:
    import ujson as json
except:
    import json
try:
    from .config import DataLoader as MyDataloader
except:
    MyDataloader = None

from .config import data_to_device, eval, compute_loss

import flgo.benchmark.base as fbb
import torch.utils.data
import os

class TaskGenerator(fbb.FromDatasetGenerator):
    def __init__(self):
        super(TaskGenerator, self).__init__(benchmark=os.path.split(os.path.dirname(__file__))[-1],
                                            train_data=train_data, val_data=val_data, test_data=test_data)

class TaskPipe(fbb.FromDatasetPipe):
    TaskDataset = torch.utils.data.Subset
    def __init__(self, task_path, train_data, val_data=None, test_data=None):
        super(TaskPipe, self).__init__(task_path, train_data=train_data, val_data=val_data, test_data=test_data)

    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_datas)) # 生成用户的名字
        feddata = {'client_names': client_names}                         # 记录用户的名字属性为client_names
        for cid in range(len(client_names)): feddata[client_names[cid]] = {'data': generator.local_datas[cid],} # 记录每个用户的本地数据划分信息，以其名字为关键字索引
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf: # 保存为data.json文件到任务目录中
            json.dump(feddata, outf)
        return

    def load_data(self, running_time_option: dict) -> dict:
        train_data = self.train_data
        test_data = self.test_data
        val_data = self.val_data
        if val_data is None:
            server_data_test, server_data_val = self.split_dataset(test_data, running_time_option['test_holdout'])
        else:
            server_data_test, server_data_val = test_data, val_data
        task_data = {'server': {'test': server_data_test, 'val': server_data_val}}
        for cid, cname in enumerate(self.feddata['client_names']):
            cdata = self.TaskDataset(train_data, self.feddata[cname]['data'])
            cdata_train, cdata_val = self.split_dataset(cdata, running_time_option['train_holdout'])
            if running_time_option['train_holdout']>0 and running_time_option['local_test']:
                cdata_val, cdata_test = self.split_dataset(cdata_val, 0.5)
            else:
                cdata_test = None
            task_data[cname] = {'train':cdata_train, 'val':cdata_val, 'test': cdata_test}
        return task_data

class TaskCalculator(fbb.BasicTaskCalculator):
    r"""
    Support task-specific computation when optimizing models, such
    as putting data into device, computing loss, evaluating models,
    and creating the data loader
    """

    def __init__(self, device, optimizer_name='sgd'):
        super(TaskCalculator, self).__init__(device, optimizer_name)
        self.device = device
        self.optimizer_name = optimizer_name
        self.criterion = None
        self.DataLoader = MyDataloader if MyDataloader is not None else torch.utils.data.DataLoader
        self.collect_fn = None

    def to_device(self, data, *args, **kwargs):
        return data_to_device(data)

    def get_dataloader(self, dataset, batch_size=64, *args, **kwargs):
        return self.DataLoader(dataset, batch_size=batch_size, **kwargs,)

    def test(self, model, data, *args, **kwargs):
        return eval(model, data, *args, **kwargs)

    def compute_loss(self, model, data, *args, **kwargs):
        return compute_loss(model, data, *args, **kwargs)