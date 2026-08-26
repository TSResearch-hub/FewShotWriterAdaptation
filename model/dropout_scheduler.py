from torch.nn import Dropout, Dropout2d
import numpy as np


class DropoutScheduler:

    def __init__(self, encoder_config, fcn_encoder):
        """
        T: number of gradient updates to converge
        """

        self.teta_list = list()
        self.init_teta_list(fcn_encoder)

        self.start_dropout_rate = encoder_config.start_dropout_rate
        self.end_dropout_rate = encoder_config.end_dropout_rate
        self.start_epoch = getattr(encoder_config, "start_epoch_dropout", 0)
        self.end_epoch = encoder_config.end_epoch_dropout
       
    def init_teta_list(self, fcn_encoder):
        #for model_name in models.keys():
        self.init_teta_list_module(fcn_encoder)

    def init_teta_list_module(self, module):
        for child in module.children():
            if isinstance(child, Dropout) or isinstance(child, Dropout2d):
                self.teta_list.append([child, child.p])
            else:
                self.init_teta_list_module(child)

    def update_dropout_rate(self, current_epoch):
        self.current_dropout_rate = self.expo_dropout_update(current_epoch)
        for (module, p) in self.teta_list:
            module.p = self.current_dropout_rate

    def expo_dropout_update(self, current_epoch):
        """
        Exponential increase of the dropout rate over epochs.

        Dropout starts at start_dropout_rate and grows exponentially
        up to end_dropout_rate between start_epoch and end_epoch.

        Args:
            current_epoch (int): current epoch (0-indexed)
            self.start_epoch (int): epoch from which the increase begins
            self.end_epoch (int): epoch at which the increase stops
            self.start_dropout_rate (float): initial dropout value
            self.end_dropout_rate (float): final dropout value
        """

        # Before the start: no increase yet
        if current_epoch <= self.start_epoch:
            new_dropout = self.start_dropout_rate

        # After the end: final rate reached
        elif current_epoch >= self.end_epoch:
            new_dropout = self.end_dropout_rate

        else:
            # Normalized progress (between 0 and 1)
            progress = (current_epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)

            # Exponential growth of the form (1 - exp(-10x))
            growth = 1 - np.exp(-10 * progress)

            # Interpolation between start and end dropout
            new_dropout = self.start_dropout_rate + \
                        (self.end_dropout_rate - self.start_dropout_rate) * growth

        return new_dropout
    