import json

from wbm_class import *
from wbm_util_func import *

# DATASET LOAD
# PRIVATE DATASET LOAD
my_wbm_raw = np.loadtxt('./dataset/private_dataset_rawdata.csv', delimiter=',', dtype='int')  # shape : (520, 968)
my_label_raw = np.loadtxt('./dataset/private_dataset_label.csv', delimiter=',', dtype='int')  # shape : (520, 1)
# shape (160, 968): 원래 dataset 에서 모양이 애매한 wafer 들 (label : 99 로 표기) 제외, 160 개 wafer 를 가져옴
private_label = my_label_raw[my_label_raw != 99]
private_wbmdata = my_wbm_raw[my_label_raw != 99]
private_label_map_type_list = ['C+R', 'Left_Top', 'Center', 'L+R', 'Edge', 'Donut', 'Random']

# PUBLIC DATASET LOAD : MAP SHAPE : (26, 26) 'none' type 은 제외
infile = open('./dataset/wm811k_dataset', 'rb')
wm811k_dataset = pickle.load(infile, encoding='latin1')
wm811k_2626 = WM811K(wm811k_dataset,
                     map_shape=(26, 26), n_valid=533, label_count_lim=5, map_type_exclude=['none'], verbose=False)
wbm_2626 = WBM(wm811k_2626.data, wm811k_2626.label_list, wm811k_2626.map_type_list, (26, 26))

if __name__ == '__main__':
    wbm_2244 = WBM(private_wbmdata,
                   private_label,
                   private_label_map_type_list,
                   map_shape=(22, 44), fail_rate_limit=0.01)
    model_2244 = MODEL(wbm_2244)

    get_similarity_euclidean(model_2244, model_2244.wbm_obj.target_wf_list)
    get_similarity_JSD(model_2244, model_2244.wbm_obj.target_wf_list)
    get_similarity_SKL(model_2244, model_2244.wbm_obj.target_wf_list)
    model_2244.update_dict_sim_score()
    get_similarity_WMHD(model_2244, [0], weight_type='type_2', m=1, s_out_rate=0.1)
    model_2244.update_dict_sim_score()
    with open('./results/data.json', 'w') as fw:
        fw.write(json.dumps(model_2244.sim_score_dict, indent=4, sort_keys=True))

    print('================')
    print('end')
