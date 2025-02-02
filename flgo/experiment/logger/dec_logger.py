from flgo.experiment.logger import BasicLogger
import numpy as np
import flgo.simulator.base as ss

class DecLogger(BasicLogger):
    r"""Simple Logger. Only evaluating model performance on testing dataset and validation dataset."""
    def initialize(self):
        """This method is used to record the stastic variables that won't change across rounds (e.g. local data size)"""
        for c in self.participants:
            self.output['client_datavol'].append(len(c.train_data))

    """This logger only records metrics on validation dataset"""
    def log_once(self, *args, **kwargs):
        global_test_metrics = []
        local_val_metrics = []
        local_test_metrics = []
        for c in self.participants:
            local_val_metrics.append(c.test(flag = 'val'))
            local_test_metrics.append(c.test(flag='test'))
            global_test_metrics.append(self.coordinator.test(c.model, flag = 'test'))
        val_met_name = list(local_val_metrics[0].keys()) if local_val_metrics[0] is not None else []
        if len(val_met_name)>0:
            val_met = {met_name: [] for met_name in val_met_name}
            for cid in range(len(self.participants)):
                for k in val_met_name:
                    val_met[k].append(local_val_metrics[cid][k])
            for k in val_met_name:
                self.output['val_'+k].append(np.mean(val_met[k]))
        test_met_name = list(local_test_metrics[0].keys()) if local_test_metrics[0] is not None else []
        if len(test_met_name)>0:
            test_met = {met_name: [] for met_name in test_met_name}
            for cid in range(len(self.participants)):
                for k in test_met_name:
                    test_met[k].append(local_test_metrics[cid][k])
            for k in test_met_name:
                self.output['local_test_'+k].append(np.mean(test_met[k]))
        gtest_name = list(global_test_metrics[0].keys()) if global_test_metrics[0] is not None else []
        if len(gtest_name)>0:
            gtest_met = {met_name: [] for met_name in gtest_name}
            for cid in range(len(self.participants)):
                for k in gtest_name:
                    gtest_met[k].append(global_test_metrics[cid][k])
            for k in gtest_name:
                self.output['global_test_'+k].append(np.mean(gtest_met[k]))
        # output to stdout
        self.show_current_output()