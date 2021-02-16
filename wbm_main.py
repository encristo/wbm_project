from wbm_class import *
from wbm_util_func import *

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
               map_shape=(22, 44), fail_rate_limit=0.01, norm_factor=10)

if __name__ == '__main__':

    model_2244 = MODEL(wbm_2244)
    model_2244.get_similarity_euclidean(wbm_2244.target_wf_list)
    model_2244.get_similarity_JSD(wbm_2244.target_wf_list, dpgmm_infer_method='vi', n_cg=9, cov_type='real')
    model_2244.update_dict_sim_score()
    model_2244.export_runtime_dict_to_json()
    model_2244.get_similarity_WMHD([0], weight_type='type_2', m=1, s_out_rate=0.1)
    model_2244.update_dict_sim_score()
    model_2244.export_runtime_dict_to_json()

    print('================')
    print('end')

    dpgmm_mc = DPGMM_MC(0, wbm_2244)
    dpgmm_mc.cluster()