output_dir = "../saved_models/"
figure_folder = "../figures/"

# dataset paths
train_df_path =         "/home/kaushikdas/aashish/multi-digit-dataset/output_3/train/labels_1_to_8.csv"
train_image_base_path = "/home/kaushikdas/aashish/multi-digit-dataset/output_3/train/combined_1_to_8_random_real/"
test_df_path =          "/home/kaushikdas/aashish/multi-digit-dataset/output_3/test/labels_1_to_8.csv"
test_image_base_path =  "/home/kaushikdas/aashish/multi-digit-dataset/output_3/test/combined_1_to_8_random_real/"
# Local test paths
# train_df_path = "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/output_2/test/labels_1.csv"
# train_image_base_path = "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/output_2/test/1_reshape/"
# test_df_path = "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/output_2/test/labels_1.csv"
# test_image_base_path = "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/output_2/test/1_reshape/"

# train_df_path = "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/output_2/train/labels_1_to_8.csv"
# train_image_base_path = "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/output_2/train/combined_1_to_8_real/"
# test_df_path = "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/output_2/test/labels_1_to_8.csv"
# test_image_base_path = "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/output_2/test/combined_1_to_8_real/"


model_folder_name =  "multi_digit_model_1_to_8_real"
model_json_file_name =  "multi_digit_model_1_to_8_real_json.json"
model_weights_file_name =  "multi_digit_model_1_to_8_real_weights.h5"
model_tflite_name =  "worksheet.multi_digit_model_1_to_8_real.tflite"

# dataloader params
img_height = 28
img_width = 140
num_time_steps = 35  # img_width//4
max_digit_length = 8
shuffle = True

# model params
batch_size = 256
epochs = 10
early_stopping_patience = 3


