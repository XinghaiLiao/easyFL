import os
from flgo.benchmark.toolkits.cv.classification import GeneralCalculator, HierFromDatasetPipe
from flgo.benchmark.base import HierFromDatasetGenerator
from .config import train_data
try:
    from .config import test_data
except:
    test_data = None
try:
    from .config import val_data
except:
    val_data = None

class TaskGenerator(HierFromDatasetGenerator):
    def __init__(self):
        super(TaskGenerator, self).__init__(benchmark=os.path.split(os.path.dirname(__file__))[-1],
                                            train_data=train_data, val_data=val_data, test_data=test_data)
        self.scene = 'hierarchical'

class TaskPipe(HierFromDatasetPipe):
    def __init__(self, task_path):
        super(TaskPipe, self).__init__(task_path, train_data=train_data, val_data=val_data, test_data=test_data)

TaskCalculator = GeneralCalculator