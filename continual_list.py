from builder import create_continual_index_list
from util import DataFolder

if __name__ == '__main__':
    # data_folder = '~/codes/OPP-PersonReID/Market-1501/pytorch/train_all'
    # data_folder = '~/codes/OPP-PersonReID/DukeMTMC-reID/pytorch/train_all'
    data_folder = '~/codes/OPP-PersonReID/MARS/bbox_train'
    # data_folder = '~/codes/OPP-PersonReID/MSMT17/pytorch/train_all'

    _train_dataset = DataFolder(root=data_folder)

    # continual_index_list = create_continual_index_list('Market-1501', _train_dataset)
    # continual_index_list = create_continual_index_list('DukeMTMC-ReID', _train_dataset)
    continual_index_list = create_continual_index_list('MARS', _train_dataset)
    # continual_index_list = create_continual_index_list('MSMT17', _train_dataset)
    pass
