import os
import time

MODEL_NAME = 'transformer_modified_zero_with_summed_bond_embeddings_only_sell'

TIME_DIR = time.strftime('%Y_%m_%d_%H_%M_%S')
IS_TRAIN = True

# DATA_ROOT_DIR = r'D:\Data\share_mine_laptop\community_detection\data'
DATA_ROOT_DIR = r'/fs/clip-scratch/yusenlin/data'

__group_dir = os.path.join(DATA_ROOT_DIR, 'input_data')
__group_dir_dealer_prediction = os.path.join(DATA_ROOT_DIR, 'input_data_dealer_prediction')
group_name = 'group_k_means_split_by_date'
group_name_dealer_prediction = 'group_k_means_cluster_4_feat_1_trace_count_2_volume_3_num_dealer_split_by_date'
group_param_name = 'no_day_off_no_distinguish_buy_sell_use_transaction_count_only_sell'
group_param_name_dealer_prediction = 'no_day_off_no_distinguish_buy_sell_use_transaction_count'
group_file_name = 'group_1'
group_path = os.path.join(__group_dir, group_name, group_param_name, group_file_name)
group_path_dealer_prediction = os.path.join(__group_dir_dealer_prediction, group_name_dealer_prediction,
                                            group_param_name_dealer_prediction, group_file_name)

freq_level = 0

LOG = {
    'group_name': group_name,
    'group_param_name': group_param_name,
    'group_file_name': group_file_name,
}
