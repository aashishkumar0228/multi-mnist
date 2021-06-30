import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt


def check_dataset(data):
    for batch in data:
        print("x_train.shape",batch['image'].shape)
        print("y_train.shape",batch['label'].shape)
        print("input_length_arr.shape",batch['input_length'].shape)
        print("label_length_arr.shape",batch['label_length'].shape)
        break

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    results = results.numpy()
    # Iterate over the results and get back the text
    output_preds = []
    for res in results:
        pred_str = ""
        for i in res:
            if i != -1:
                pred_str += str(i)
        output_preds.append(pred_str)
    return output_preds

def get_loss_plot(history_1, figure_folder=None):
    plt.plot(history_1.epoch,history_1.history['loss'],label="loss") # Loss curve for training set
    plt.plot(history_1.epoch,history_1.history['val_loss'],label="val_loss") # Loss curve for validation set
    plt.title("Loss Curve",fontsize=18)
    plt.xlabel("Epochs",fontsize=15)
    plt.ylabel("Loss",fontsize=15)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(figure_folder + '/loss.png')

def create_loss_csv(history_1, figure_folder=None):
    loss_df = pd.DataFrame(list(zip(history_1.epoch, history_1.history['loss'], history_1.history['val_loss'])), 
                            columns = ['Epoch', 'Train_Loss', 'Val_loss'])
    loss_df.to_csv(figure_folder + "/loss.csv", index=None)


def editDistDP(str1, str2): 
    m = len(str1)
    n = len(str2)
    # Create a table to store results of subproblems 
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)] 
  
    # Fill d[][] in bottom up manner 
    for i in range(m + 1): 
        for j in range(n + 1): 
  
            # If first string is empty, only option is to 
            # insert all characters of second string 
            if i == 0: 
                dp[i][j] = j    # Min. operations = j 
  
            # If second string is empty, only option is to 
            # remove all characters of second string 
            elif j == 0: 
                dp[i][j] = i    # Min. operations = i 
  
            # If last characters are same, ignore last char 
            # and recur for remaining string 
            elif str1[i-1] == str2[j-1]: 
                dp[i][j] = dp[i-1][j-1] 
  
            # If last character are different, consider all 
            # possibilities and find minimum 
            else: 
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert 
                                   dp[i-1][j],        # Remove 
                                   dp[i-1][j-1])    # Replace 
  
    return dp[m][n] 

def get_edit_distance_freq(df):
    edit_distance_freq = {}
    wrong_count = 0
    for i in range(0, df.shape[0]):
        if df['labels'][i] is not None:
            edit_distance = editDistDP(df['labels'][i], df['preds'][i])
            if edit_distance_freq.get(edit_distance) is None:
                edit_distance_freq[edit_distance] = 1
            else:
                edit_distance_freq[edit_distance] += 1
            if edit_distance > 2:
                print(df['file_name'][i],df['labels'][i], df['preds'][i], i)
            if df['labels'][i] != df['preds'][i]:
                wrong_count += 1
    
    return edit_distance_freq, wrong_count

def show_edit_distance_freq_graph(edit_distance_freq, title, figure_folder=None):
    plt.figure(figsize=(7,5))
    plt.bar(edit_distance_freq.keys(),edit_distance_freq.values(),width = 0.8,color="orange")
    for i in edit_distance_freq:
        plt.text(i,edit_distance_freq[i],str(edit_distance_freq[i]),horizontalalignment='center',fontsize=10)

    plt.tick_params(labelsize = 14)
    plt.xticks(range(0,8))
    plt.xlabel("Edit Distance",fontsize=16)
    plt.ylabel("Frequency",fontsize=16)
    plt.title(title)
    plt.savefig(figure_folder + "/" + title + ".png")
    
def get_word_leng_freq(df):
    correct_word_length_freq = {}
    wrong_word_length_freq = {}
    for i in range(0, df.shape[0]):
        if df['labels'][i] == df['preds'][i]:
            word_length = len(df['labels'][i])
            if correct_word_length_freq.get(word_length) is None:
                correct_word_length_freq[word_length] = 1
            else:
                correct_word_length_freq[word_length] += 1
        else:
            word_length = len(df['labels'][i])
            if wrong_word_length_freq.get(word_length) is None:
                wrong_word_length_freq[word_length] = 1
            else:
                wrong_word_length_freq[word_length] += 1
        
    return correct_word_length_freq, wrong_word_length_freq

def show_word_length_freq_graph(freq_var, title, figure_folder=None):
    plt.figure(figsize=(7,5))
    plt.bar(freq_var.keys(),freq_var.values(),width = 0.8,color="orange")
    for i in freq_var:
        plt.text(i,freq_var[i],str(freq_var[i]),horizontalalignment='center',fontsize=10)

    plt.tick_params(labelsize = 14)
    plt.xticks(range(0,10))
    plt.xlabel("word length",fontsize=16)
    plt.ylabel("Frequency",fontsize=16)
    plt.title(title)
    plt.savefig(figure_folder + "/" + title + ".png")