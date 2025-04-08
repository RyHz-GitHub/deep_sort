import os
import pandas as pd
from collections import Counter

# Compare all Detections from the updated Version with the original result that was submitted to the MOT Challenge.
# Result txt.file format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

# There are some lines that are different. However they mostly differ by Track ID.
# After eliminating the Track ID the difference left appears to be only a switch in rows. The Track ID change results in some cases where the order changes.

original_dir = "MOTresults"
new_dir      = "results"

def compare_by_String(output = True):
    differences = []
    for sequence in os.listdir("MOTresults"):
        original_sequence_path = os.path.join(original_dir, sequence)
        new_sequence_path = os.path.join(new_dir, sequence)

        with open(original_sequence_path, 'r') as orig, open(new_sequence_path, 'r') as new:
            
            n=0
            unequal_lines = []
            for orig_line,new_line in zip(orig,new):
                if orig_line != new_line:
                    unequal_lines.append(n)
                n+=1

            if output == True:
                print(f"Detections in file {sequence.split('.')[0]}: {n} ")
                print(f"There are {len(unequal_lines)} unequal lines")
                print(f"Percentage unequal lines: {round(len(unequal_lines)/(n)*100,2)} % \n") 
            differences.append(unequal_lines)
    return differences

# comparison by string which leads to edge case differences (8 cases of 0.00 != -0.00) in MOT16-03, <bb_top> column, track_ids: 192, 317
# comparison with pandas, compares the values as numbers thus + or - 0 doesn't matter

def compare_with_pandas(output=True):
    differences_dataframes = []
    for sequence in os.listdir("MOTresults"):
        original_sequence_path = os.path.join(original_dir, sequence)
        new_sequence_path = os.path.join(new_dir, sequence)

        df_orig = pd.read_csv(original_sequence_path, header = None)
        df_orig.columns=["frame", "track_id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"]
        df_orig_dropid = df_orig.drop(columns="track_id")
        df_orig_sorted = df_orig_dropid.sort_values(by=list(df_orig_dropid.columns)).reset_index(drop=True)
        
        df_new = pd.read_csv(new_sequence_path, header = None)
        df_new.columns=["frame", "track_id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"]
        df_new_dropid = df_new.drop(columns="track_id")
        df_new_sorted = df_new_dropid.sort_values(by=list(df_new_dropid.columns)).reset_index(drop=True)
        
        df_diff = df_orig.compare(df_new,keep_equal=True, result_names=('orig', 'new'))
        df_diff_dropid = df_orig_dropid.compare(df_new_dropid,keep_equal=True, result_names=('orig', 'new'))
        df_diff_sorted = df_orig_sorted.compare(df_new_sorted,keep_equal=True, result_names=('orig', 'new'))

        df_diff.to_csv(f'Metrics\compare_result_lines\\raw\{sequence.split(".")[0]}.csv')
        df_diff_dropid.to_csv(f'Metrics\compare_result_lines\dropid\{sequence.split(".")[0]}.csv')
        df_diff_sorted.to_csv(f'Metrics\compare_result_lines\sorted\{sequence.split(".")[0]}.csv')
        
        n = df_orig.shape[0]
        amount_diff_raw  = df_diff.shape[0]
        amount_diff_dropid = df_diff_dropid.shape[0]
        amount_diff_sorted = df_diff_sorted.shape[0]

        differences_dataframes.append(df_diff)

        if output==True:
            print(f"Detections in file {sequence.split('.')[0]}: {n}\n")

            print(f"There are {amount_diff_raw} unequal lines")
            print(f"Percentage unequal lines: {round(amount_diff_raw/n*100,2)} % \n") 

            print(f"There are {amount_diff_dropid} that differ by more than just Track ID")
            print(f"Percentage differing lines ignoring ID: {round(amount_diff_dropid/n*100,2)} % \n") 

            print(f"There are {amount_diff_sorted} that aren't just switched around.")
            print(f"Percentage differing lines ignoring order: {round(amount_diff_sorted/n*100,2)} % \n")

            print('----------------------------------------------------------------- \n')
        if sequence == "MOT16-03.txt":
            df03 = df_diff
    
    return differences_dataframes


# 8 extra differences happen, because original has value: 0.00 vs -0.00 on new version, if you compare the lines by string
def diff_pandas_string():
    differences = compare_by_String(output=False)
    differences_dataframes = compare_with_pandas(output=False)
    a = (differences[1])
    b = differences_dataframes[1].index.to_list()

    d = list(set(a) - set(b))
    d.sort()
    print(d)

# count the amount of detections for each track_id
# the id switches could result in cases where the original track_id was able to track the object but the new version couldn't
# e.g. 112, 112 , 112  -> 113, 113, 114 
# ensure track_ids are "only" switched around with no change in occurences

def count_track_ids():
    dict_diffs = []
    for sequence in os.listdir("MOTresults"):
        original_sequence_path = os.path.join(original_dir, sequence)
        new_sequence_path = os.path.join(new_dir, sequence)

        df_orig = pd.read_csv(original_sequence_path, header = None)
        df_orig.columns=["frame", "track_id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"]
        df_new = pd.read_csv(new_sequence_path, header = None)
        df_new.columns=["frame", "track_id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"]

        # list all track_ids
        track_ids_orig = df_orig['track_id'].to_list()
        track_ids_new = df_new['track_id'].to_list()

        dict_orig = Counter(track_ids_orig)
        dict_new = Counter(track_ids_new)

        dict_diff = []

        for key in range(1 , max(max(list(dict_orig)),max(list(dict_new)))):
            if dict_orig.get(key) != dict_new.get(key):
                dict_diff.append([(key,dict_orig.get(key)),(key,dict_new.get(key))])
        dict_diffs.append(dict_diff)

    for x in dict_diffs:
        # print(x) # few enough switches to check it by hand, other than last file
        orig_counts = []
        new_counts = []
        for i in range(len(x)):
            orig_counts.append(x[i][0][1])
            new_counts.append(x[i][1][1])

        if Counter(orig_counts) == Counter(new_counts): # make sure the amount of occurences fit each other
            print("check passed")
        else:
            print("check failed")



#compare_by_String()
#compare_with_pandas()
#diff_pandas_string()
#count_track_ids()