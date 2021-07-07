output_dir = "../saved_models/"
figure_folder = "../figures/"

# dataset paths
train_df_path = "/home/kaushikdas/aashish/multi-digit-dataset/train/labels_1_to_8.csv"
train_image_base_path = "/home/kaushikdas/aashish/multi-digit-dataset/train/combined_1_to_8_random_real_new/"
test_df_path = "/home/kaushikdas/aashish/multi-digit-dataset/test/labels_1_to_8.csv"
test_image_base_path = "/home/kaushikdas/aashish/multi-digit-dataset/test/combined_1_to_8_random_real_new/"
# Local test paths
# train_df_path = "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/output_2/test/labels_1.csv"
# train_image_base_path = "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/output_2/test/1_reshape/"
# test_df_path = "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/output_2/test/labels_1.csv"
# test_image_base_path = "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/output_2/test/1_reshape/"

# train_df_path = "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/output_2/train/labels_1_to_8.csv"
# train_image_base_path = "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/output_2/train/combined_1_to_8_real/"
# test_df_path = "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/output_2/test/labels_1_to_8.csv"
# test_image_base_path = "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/output_2/test/combined_1_to_8_real/"


model_folder_name =  "multi_digit_model_1_to_8_real_new"
model_json_file_name =  "multi_digit_model_1_to_8_real_new_json.json"
model_weights_file_name =  "multi_digit_model_1_to_8_real_new_weights.h5"
model_tflite_name =  "worksheet.multi_digit_model_1_to_8_real_new.tflite"

# dataloader params
img_height = 28
img_width = 168
num_time_steps = 42  # img_width//4
max_digit_length = 8
shuffle = True

# model params
batch_size = 256
epochs = 10
early_stopping_patience = 3


