output_dir = "../saved_models/"
figure_folder = "../figures/"

# dataset paths
train_df_path =         "/home/kaushikdas/aashish/multi-digit-dataset/output_comma_2/output_mix/train/labels_1_to_8_comma.csv"
train_image_base_path = "/home/kaushikdas/aashish/multi-digit-dataset/output_comma_2/output_mix/train/combined_1_to_8_comma_real/"
test_df_path =          "/home/kaushikdas/aashish/multi-digit-dataset/output_comma_2/output_mix/test/labels_1_to_8_comma.csv"
test_image_base_path =  "/home/kaushikdas/aashish/multi-digit-dataset/output_comma_2/output_mix/test/combined_1_to_8_comma_real/"
test_image_actual_shape_base_path = "/home/kaushikdas/aashish/multi-digit-dataset/output_comma_2/output_mix/test/combined_1_to_8_comma_actual_shape_real/"
# Local test paths
# train_df_path = "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/output_2/test/labels_1.csv"
# train_image_base_path = "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/output_2/test/1_reshape/"
# test_df_path = "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/output_2/test/labels_1.csv"
# test_image_base_path = "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/output_2/test/1_reshape/"

# train_df_path = "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/output_2/train/labels_1_to_8.csv"
# train_image_base_path = "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/output_2/train/combined_1_to_8_real/"
# test_df_path = "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/output_2/test/labels_1_to_8.csv"
# test_image_base_path = "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/output_2/test/combined_1_to_8_real/"


model_folder_name =  "multi_digit_model_1_to_8_comma"
model_json_file_name =  "multi_digit_model_1_to_8_comma_json.json"
model_weights_file_name =  "multi_digit_model_1_to_8_comma_weights.h5"
model_tflite_name =  "worksheet.multi_digit_model_1_to_8_comma.tflite"

# dataloader params
img_height = 28
img_width = 168
num_time_steps = 42  # img_width//4
max_digit_length = 11
shuffle = True

# model params
num_classes = 12
batch_size = 256
epochs = 10
early_stopping_patience = 3


