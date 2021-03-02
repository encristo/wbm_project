from wbm_class import *
from wbm_util_func import *
np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})

# DATASET LOAD
# PRIVATE DATASET LOAD
my_wbm_raw = np.loadtxt('./dataset/private_dataset_rawdata.csv', delimiter=',', dtype='int')  # shape : (520, 968)
my_label_raw = np.loadtxt('./dataset/private_dataset_label.csv', delimiter=',', dtype='int')  # shape : (520, 1)
# shape (160, 968): 원래 dataset 에서 모양이 애매한 wafer 들 (label : 99 로 표기) 제외, 160 개 wafer 를 가져옴
private_label = my_label_raw[my_label_raw != 99]
private_wbmdata = my_wbm_raw[my_label_raw != 99]
private_label_map_type_list = np.array(['C+R', 'Left_Top', 'Center', 'L+R', 'Edge', 'Donut', 'Random'])
wbm_2244 = WBM(private_wbmdata,
               private_label,
               private_label_map_type_list,
               map_shape=(22, 44), fail_rate_limit=(0, 1), norm_factor=10)

# infile = open('./dataset/wm811k_2626', 'rb')
# wm811k_2626 = pickle.load(infile, encoding='latin1')
# wm811k_2626_data = np.loadtxt('./dataset/wm811k_2626_data_none_100ea.csv', delimiter=',', dtype='uint8')
# wm811k_2626_label = np.loadtxt('./dataset/wm811k_2626_label_none_100ea.csv', delimiter=',', dtype='uint8')
# wbm_2626 = WBM(wm811k_2626_data,
#                wm811k_2626_label,
#                wm811k_2626.map_type_list,
#                map_shape=(26, 26), fail_rate_limit=(0, 0.5), norm_factor=10)


if __name__ == '__main__':

    model_2244 = MODEL(wbm_2244)
    model_2244.get_dpgmm(infer_method='mc')
    model_2244.get_skldm()
    n_cg = 37
    model_2244.get_cg(n_cg=n_cg)
    model_2244.update_sim_mtx_dict_euclidean()
    model_2244.set_para_wmhd(weight_type='type_2', m=0.1, s=0.1)

    print('================')
    print('end')
