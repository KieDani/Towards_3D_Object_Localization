from datetime import datetime
from helper import map_environment_name
import os

from helper import get_logs_path as glp


class BaseConfig(object):
    def __init__(self):
        super(BaseConfig, self).__init__()
        self.lr_coord = 1e-5
        self.lr_forec = 1e-7
        self.warmup_iterations = 0
        self.temperature = 0.1
        self.reprojection_weight = 1.
        self.depth_mode = 'depthmap'
        self.date_time = datetime.now().strftime("%m%d%Y-%H%M%S")
        self.BATCH_SIZE = 8
        self.NUM_EPOCHS = 400
        self.ident = None
        self.ema_decay = 0.98
        self.seed = 42

        self.lr_coord = None
        self.timesteps = None
        self.loss_coord_title = None
        self.forec_title = None
        self.environment_name = None
        self.folder = None
        self.sin_title = None



    def get_identifier(self):
        if self.ident is None:
            identifier = f'sintitle:{self.sin_title}_lossmode:{self.lossmode}_{self.date_time}'
        else:
            identifier = self.ident
        return identifier

    def get_logs_path(self, debug=True):
        identifier = self.get_identifier()
        if debug == False:
            logs_path = os.path.join(glp(), 'logs')
        else:
            logs_path = os.path.join(glp(), 'logs_tmp')
        if self.folder is not None:
            logs_path = os.path.join(logs_path, self.folder, self.environment_name, identifier)
        else:
            logs_path = os.path.join(logs_path, self.environment_name, identifier)
        return logs_path

    def get_pathforsaving(self):
        identifier = self.get_identifier()
        #tmp = 'parcour' if 'parcour' in self.environment_name else self.environment_name
        ident = os.path.join(self.folder, self.environment_name, identifier) #if self.folder is None else os.path.join(tmp, self.folder, identifier)
        return ident




class MyConfig(BaseConfig):
    def __init__(self, lr_coord, timesteps, loss_coord_title, forec_title, sin_title, environment_name, folder, lossmode):
        super(MyConfig, self).__init__()
        self.lr_coord = lr_coord
        self.timesteps = timesteps
        self.loss_coord_title = loss_coord_title
        self.forec_title = forec_title
        self.environment_name = map_environment_name(environment_name)
        self.folder = folder
        self.sin_title = sin_title
        self.lossmode = lossmode


class EvalConfig(BaseConfig):
    def __init__(self, sin_title, environment_name, identifier, folder):
        super(EvalConfig, self).__init__()
        self.sin_title = sin_title
        self.environment_name = map_environment_name(environment_name)
        self.ident = identifier
        self.folder = folder
        self.BATCH_SIZE = 1






if __name__ == '__main__':
    config = MyConfig()
    pass
