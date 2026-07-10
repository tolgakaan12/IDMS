import json

import h5py
import numpy as np
import os
import re
import sys


def fix_stack(s):
    import os

    command = 'echo | set /p nul=' + s.replace('\n', ',').replace(' ', ',').replace(',,', ',').replace(',,', ',').\
        replace(',,', ',').replace('[,', '[').strip() + '| clip'
    os.system(command)
    return


def get_min_loss(s):
    import pickle
    import os
    import numpy as np
    with open(s, "rb") as input_file:
        h = pickle.load(input_file)
    text = str([np.min(d['val_loss']) for d in h])
    command = 'echo | set /p nul=' + text.strip() + '| clip'
    os.system(command)


def document_model(dir_path, end, model, history, use_json=True, **kwargs):
    os.makedirs(dir_path, exist_ok=True)
    model.save(os.path.join(dir_path, 'model_' + str(end) +'.h5'))
    import pickle
    with open(os.path.join(dir_path, 'history_' + str(end) + '.pickle'), 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    str_sum = model_summary_to_string(model) + '\n'
    if not use_json:
        str_sum += kw_summary(**kwargs)
        print_to_file(str_sum, os.path.join(dir_path, 'summary_' + str(end) + '.txt'))
    else:
        dict_out = {"architecture": str_sum.splitlines()}
        dict_out.update(kwargs)
        save_dict(os.path.join(dir_path, 'summary_' + str(end) + '.json'), convert_numpy_to_list(dict_out))


def convert_numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        # Convert numpy array to list
        return obj.tolist()
    elif isinstance(obj, list):
        # Recursively convert each element in the list
        return [convert_numpy_to_list(item) for item in obj]
    elif isinstance(obj, dict):
        # Recursively convert each value in the dictionary
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    else:
        # Return the object unchanged if it is neither a numpy array, list, nor dict
        return obj if isinstance(obj, (int, float, bool, str)) else str(obj)


def model_summary_to_string(model):
    str_list = []
    model.summary(print_fn=lambda x: str_list.append(x))
    return "\n".join(str_list)


def print_to_file(str_in, filepath):
    with open(filepath, 'w+') as f:
        f.write(str_in)


def kw_summary(**kwargs):
    str_out = str()
    for key, item in kwargs.items():
        str_out += key + ': ' + str(item) + '\n'
    return str_out


def summary(k, scores, kernel, drop, model, epochs, batch, files):
    print(scores)
    m, st = np.mean(scores), np.std(scores)

    print('MSE: {0:.3f} (+/-{1:.3f})'.format(-m, st))
    print('K-fold: {0:.0f}'.format(k))

    # region Self-documentation
    from datetime import datetime
    summary_dir = os.environ.get('IDMS_SUMMARY_DIR', 'Summaries')
    os.makedirs(summary_dir, exist_ok=True)

    dir_path, filename, ends = incr_file(summary_dir, 'model_summary', '.txt')
    summary_path = os.path.join(dir_path, filename)
    print(summary_path)

    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    with open(summary_path, 'w+') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write("\nDate: {}\n"
                "File: {}\n"
                "Scores: {}\n"
                "MSE: {:.3f} (+/-{:.3f})\n"
                "Dropout (if applicable): {:.3f}\n"
                "Kernel (if applicable): {:.0f}, {:.0f}\n"
                "Epochs: {}, Batch size: {}\n"
                "K: {:.0f}\n".format(dt_string, os.path.basename(sys.argv[0]),
                                     scores, m, st, drop, kernel[0], kernel[1], epochs, batch, k)
                + "\n\nUsing Files:\n")
        for file in files:
            f.write(file + "\n")


def incr_file(dir_path, file_name, ext):
    ends = [int(re.search(r'(\d+)$', str(os.path.splitext(f)[0])).group(0))
            for f in os.listdir(dir_path) if f.endswith(ext) and file_name in f]
    if not ends:
        ends = [0]
    filename = f'{file_name}{max(ends) + 1}{ext}'
    return dir_path, filename, ends


### WINDOWS STYLE ###
# def summary(k, scores, kernel, drop, model, data_path, epochs, batch, files):
#     print(scores)
#     m, st = np.mean(scores), np.std(scores)

#     print('MSE: {0:.3f} (+/-{1:.3f})'.format(-m, st))
#     print('K-fold: {0:.0f}'.format(k))

#     # region Self-documentation

#     file_name = incr_file(r'C:\Users\win10\Desktop\Projects\CYB\PyCYB\Summaries', r'model_summary', '.txt')

#     ends = [int(re.search(r'(\d+)$', str(os.path.splitext(f)[0])).group(0))
#             for f in os.listdir(r'C:\Users\win10\Desktop\Projects\CYB\PyCYB\Summaries') if f.endswith('.txt')]
    
#     if not ends:
#         ends = [0]
#     print(r'C:\Users\win10\Desktop\Projects\CYB\PyCYB\Summaries\model_summary' +
#           str(max(ends) + 1) + '.txt')
#     from datetime import datetime
#     dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

#     with open(r'C:\Users\win10\Desktop\Projects\CYB\PyCYB\Summaries\model_summary' +
#               str(max(ends) + 1) + '.txt', 'w+') as f:
#         model.summary(print_fn=lambda x: f.write(x + '\n'))
#         f.write("\nDate: {}\n"
#                 "File: {}\n"
#                 "Scores: {}\n"
#                 "MSE: {:.3f} (+/-{:.3f})\n"
#                 "Dropout (if applicable): {:.3f}\n"
#                 "Kernel (if applicable): {:.0f}, {:.0f}\n"
#                 "Epochs: {}, Batch size: {}\n"
#                 "K: {:.0f}\n".format(dt_string, os.path.basename(sys.argv[0]),
#                                      scores, m, st, drop, kernel[0], kernel[1], epochs, batch, k)
#                 + "\n\nUsing Files:\n")
#         for file in files:
#             f.write(file + "\n")
# # 
# 
# def incr_file(dir_path, file_name, ext):
#     ends = [int(re.search(r'(\d+)$', str(os.path.splitext(f)[0])).group(0))
#             for f in os.listdir(dir_path) if f.endswith(ext)
#             and file_name in f]
#     if not ends:
#         ends = [0]
#     return dir_path + '\\', file_name + str(max(ends) + 1) + ext, ends
#
# def incr_dir(dir_path, dir_name, make=True):
#     ends = [int(re.search(r'(\d+)', d).group(0)) for d in next(os.walk(dir_path))[1] if dir_name in d]
#     if not ends:
#         ends = [0]
#     new_dir = dir_path + '\\' + dir_name + str(max(ends) + 1)
#     if make:
#         os.makedirs(new_dir)
#     return new_dir, ends
# 
# def get_file_names(dir_path, task=None):
#     return [dir_path + '\\' + file for file in sorted([f for f in os.listdir(dir_path) if f.endswith('.json')])
#             if task is None or task in file]
# 
# def load_dict_stack(path, task='None'):
#     dict_stack = list()
# 
#     def f_check(f):
#         return np.any([n in f for n in task or task == 'None'])
#     for file in sorted([f for f in os.listdir(path) if f.endswith('.json') and f_check(f)]):
#         with open(path + '\\' + file) as json_file:
#             dict_data = json.load(json_file)
#             dict_stack.append(dict_data)
#     return dict_stack
# 
# def load_emg_stack(path, task='None', n_channels=8):
#     emg_stack = list()
#     def f_check(f):
#         return np.any([n in f for n in task or task == 'None'])
#
#     for file in sorted([f for f in os.listdir(path) if f.endswith('.json') and f_check(f)]):
#         with open(path + '\\' + file) as json_file:
#             dict_data = json.load(json_file)
#             emg_stack.append(np.array(dict_data["EMG"]))
#     return emg_stack


def incr_dir(dir_path, dir_name, make=True):
    ends = [int(re.search(r'(\d+)', d).group(0)) for d in next(os.walk(dir_path))[1] if dir_name in d]
    if not ends:
        ends = [0]
    new_dir = os.path.join(dir_path, f'{dir_name}{max(ends) + 1}')
    if make:
        os.makedirs(new_dir)
    return new_dir, ends


def get_file_names(dir_path, task=None):
    return [os.path.join(dir_path, file) for file in sorted([f for f in os.listdir(dir_path) if f.endswith('.json')])
            if task is None or task in file]


def load_dict(file_path):
    with open(file_path) as json_file:
        dict_data = json.load(json_file)
    return dict_data


def load_dict_stack(path, task='None'):
    dict_stack = list()

    def f_check(f):
        return np.any([n in f for n in task or task == 'None'])

    for file in sorted([f for f in os.listdir(path) if f.endswith('.json') and f_check(f)]):
        with open(os.path.join(path, file)) as json_file:
            dict_data = json.load(json_file)
            dict_stack.append(dict_data)
    return dict_stack


def save_dict(file_path, dict_in):
    with open(file_path, 'w') as fp:
        json.dump(dict_in, fp, indent=4)
    return


def load_emg_stack(path, task='None', n_channels=8):
    emg_stack = list()
    def f_check(f):
        return np.any([n in f for n in task or task == 'None'])

    for file in sorted([f for f in os.listdir(path) if f.endswith('.json') and f_check(f)]):
        with open(os.path.join(path, file)) as json_file:
            dict_data = json.load(json_file)
            emg_stack.append(np.array(dict_data["EMG"]))
    return emg_stack

def load_emg(file_path):
    import json
    import numpy as np
    with open(file_path, 'r') as f:
        data = json.load(f)
    return np.array(data["EMG"]).T

# Function to recursively save the dictionary
def save_dict_to_hdf5(file_path, dictionary):
    def recursive_save(group, dictionary):
        for key, item in dictionary.items():
            if isinstance(item, dict):  # If the item is a dictionary, create a subgroup
                subgroup = group.create_group(key)
                recursive_save(subgroup, item)
            else:
                group[key] = item

    # Saving the dictionary inside an HDF5 file
    with h5py.File(file_path, 'w') as hdf_file:
        recursive_save(hdf_file, dictionary)


# Function to recursively load the dictionary
def load_dict_from_hdf5(file_path):
    def recursive_load(group):
        result = {}
        for key, item in group.items():
            if isinstance(item, h5py.Group):  # If the item is a group, recursively load it
                result[key] = recursive_load(item)
            else:
                result[key] = item[()]  # Extract the value from the dataset
        return result

    # Loading the dictionary from an HDF5 file
    with h5py.File(file_path, 'r') as hdf_file:
        return recursive_load(hdf_file)


if __name__ == '__main__':
    my_dict = {
        "array": [1, 2, 3, 4, 5],
        "value": 42,
        "nested_dict": {"a": 1, "b": 2}
    }

    file_path = 'data_test.h5'
    save_dict_to_hdf5(file_path, my_dict)

    loaded_dict = load_dict_from_hdf5(file_path)
    print(loaded_dict)