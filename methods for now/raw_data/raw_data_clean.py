import os
import pandas as pd
import numpy as np
import csv

'''Check if the line start with number'''
def is_number(line):
    try: 
        int(line[0])
    except ValueError:
        return False
    return True

'''check if this line is a MSG'''
def is_msg(line):
    if line[0:3] == 'MSG':
        return True
    return False 

'''check if this is an empty line'''
def is_newline(line):
    if line == '' or line == '\n':
        return True
    return False

'''Check if is a fix behavior'''
def is_fix(line):
    if "SFIX" in line or "EFIX" in line:
        return True
    return False

'''Check if is a sacc behavior'''
def is_sacc(line):
    if "SSACC" in line or "ESACC" in line:
        return True
    return False

'''check if it is a blink behavior'''
def is_blink(line):
    if "SBLINK" in line or "EBLINK" in line:
        return True
    return False    

'''Check if the coordinate [x, y] is on the picture'''
def check_inpic(x, y, img_msg):
    [mid_x, mid_y, width, height] = img_msg[1:]
    start_x = float(mid_x) - float(width) / 2
    start_y = float(mid_y) - float(height) / 2
    end_x = float(mid_x) + float(width) / 2
    end_y = float(start_y) + float(height) / 2
    try:
        x = float(x)
        y = float(y)
        if x >= start_x and x <= end_x and y >= start_y and y <= end_y:
            return True
        return False
    except:
        return False

'''handle forced memory event'''
def process_forcedmemory(msg_marker, msg, data, current_data_index, current_msg_index):
    assert("forcedmemory_start" in msg[current_msg_index][3])
    start_msg_index = current_msg_index
    event_name = msg[start_msg_index][3]
    img_msg = []
    start_time = msg[start_msg_index][2]
    end_time = -1
    while(1):
        current_msg_index += 1
        if "IMGLOAD CENTER" in msg[current_msg_index][3]:
            img_msg = msg[current_msg_index][3].split(" ")[3:] 
            # img_msg: [pic_loc, x, y, width, height]
        elif "forcedmemory_end" in msg[current_msg_index][3]:
            end_time = msg[current_msg_index][2]
            break

    while data[current_data_index][1] <= start_time:
        current_data_index += 1

    while data[current_data_index][1] <= end_time:
        msg_marker[current_data_index][0] = event_name
        current_x = data[current_data_index][2]
        current_y = data[current_data_index][3]
        if check_inpic(current_x, current_y, img_msg):
            msg_marker[current_data_index][1] = img_msg[0]
        else:
            msg_marker[current_data_index][1] = "not on pics"
        current_data_index += 1 
    return [current_data_index, current_msg_index, msg_marker]
 
'''check if the coordiante [x, y] is in the box'''
def check_inbox(x, y, box_msg):
    [start_x, start_y, end_x, end_y] = box_msg
    start_x = float(start_x)
    start_y = float(start_y)
    end_x = float(end_x)
    end_y = float(end_y)
    try:
        x = float(x)
        y = float(y)
        if x >= start_x and x <= end_x and y >= start_y and y <= end_y:
            return True
        return False
    except:
        return False

'''find which picture has been fixed'''
def fix_on_picnum(x, y, img_msg_bottom, img_msg_left, img_msg_right):
    if check_inpic(x, y, img_msg_bottom):
        return 0
    elif check_inpic(x, y, img_msg_left):
        return 1
    elif check_inpic(x, y, img_msg_right):
        return 2
    else:
        return -1

'''find which box has been chosen'''
def fix_on_boxnum(x, y, box_msg_bottom, box_msg_left, box_msg_right):
    if check_inbox(x, y, box_msg_bottom):
        return 0
    elif check_inbox(x, y, box_msg_left):
        return 1
    elif check_inbox(x, y, box_msg_right):
        return 2
    else:
        return -1

'''handle forced choice evenr'''
def process_forcedchoice(msg_marker, msg, data, current_data_index, current_msg_index):
    assert("forcedchoice_start" in msg[current_msg_index][3])
    start_msg_index = current_msg_index
    event_name = msg[start_msg_index][3]
    img_msgs = []
    box_msgs = []
    start_time = msg[start_msg_index][2]
    end_time = -1
    while(1):
        current_msg_index += 1
        if "IMGLOAD CENTER" in msg[current_msg_index][3]:
            img_msgs.append(msg[current_msg_index][3].split(" ")[3:])
            # img_msg: [pic_loc, x, y, width, height]
        elif "DRAWBOX" in msg[current_msg_index][3]:
            box_msgs.append(msg[current_msg_index][3].split(" ")[5:])
            # img_msg: [x, y, width, height]
        elif "forcedchoice_end" in msg[current_msg_index][3]:
            end_time = msg[current_msg_index][2]
            break
    
    while data[current_data_index][1] <= start_time:
        current_data_index += 1

    while data[current_data_index][1] <= end_time:
        msg_marker[current_data_index][0] = event_name
        current_x = data[current_data_index][2]
        current_y = data[current_data_index][3]
        fix_pic_num = fix_on_picnum(current_x, current_y, img_msgs[0], img_msgs[1], img_msgs[2])
        fix_box_num = fix_on_boxnum(current_x, current_y, box_msgs[0], box_msgs[1], box_msgs[2])
        # 0: bottom, 1: left, 2: right, -1: none
        if fix_pic_num != -1:
            msg_marker[current_data_index][1] = img_msgs[fix_pic_num][0]
        elif fix_box_num != -1:
            if fix_box_num == 0:
                msg_marker[current_data_index][1] = "box_bottom"
            elif fix_box_num == 1:
                msg_marker[current_data_index][1] = "box_left"
            elif fix_box_num == 2:
                msg_marker[current_data_index][1] = "box_right"
        else:
            msg_marker[current_data_index][1] = "out"
        current_data_index += 1 
    return [current_data_index, current_msg_index, msg_marker]

def clean_calibration(file_to_parse):
    # file_to_parse = "./J030701.asc"
    print("start parsing file %s"%file_to_parse)
    file = open(file_to_parse, 'r')
    lines = file.read().splitlines(True)
    header_split = 0
    for i in range(0, len(lines) - 2):
        if is_msg(lines[i]) & is_newline(lines[i+1]) & is_number(lines[i+2]):
            header_split = i + 2
            break

    original_removed_header = np.array(lines[header_split:])
    return original_removed_header

def extract_raw_to_data_movement(removed_header):
    msg = []
    fix = []
    sacc = []
    blink = []
    miss = []
    for i in reversed(range(np.size(removed_header))):
        if is_number(removed_header[i]):
            continue
        else:
            if is_msg(removed_header[i]):
                msg.append(removed_header[i])
            elif is_blink(removed_header[i]):
                blink.append(removed_header[i])
            elif is_fix(removed_header[i]):
                fix.append(removed_header[i])
            elif is_sacc(removed_header[i]):
                sacc.append(removed_header[i])
            else:
                miss.append(removed_header[i])
            removed_header = np.delete(removed_header, i, 0)
    
    nparr = np.asarray(removed_header)
    out_arr = np.char.split(nparr)
    arr = np.asarray([np.array(line) for line in out_arr])
    reverse_arr = arr[:, :-1]
    pd.DataFrame(reverse_arr).to_csv('data.csv', index_label = "Index", header  = ['Time','x','y','pupil size','CR'])

    for i in range(len(msg)):
        msg[i] = msg[i].replace('\t', ' ')
    split_msg = np.char.split(msg)
    for i in range(len(split_msg)):
        if len(split_msg[i])>3:
            for str in split_msg[i][3:]:
                split_msg[i][2] += " "
                split_msg[i][2] += str
                split_msg[i].remove(str)
    np_msg = np.asarray([np.array(line) for line in split_msg])
    reverse_msg = np_msg[::-1]
    pd.DataFrame(reverse_msg).to_csv('msg.csv', index_label = "Index", header  = ['Type','Time','Desceiption'])

    for i in range(len(fix)):
        fix[i] = fix[i].replace('\t', ' ')
    split_fix = np.char.split(fix)
    for i in reversed(range(np.shape(split_fix)[0])):
        if split_fix[i][0] == "SFIX":
            split_fix = np.delete(split_fix, i, 0)
    np_efix = np.asarray([np.array(line) for line in split_fix])
    reverse_efix = np_efix[::-1]
    pd.DataFrame(reverse_efix).to_csv('efix.csv', index_label = "Index", header  = ['Type','Eye','Start_time','End_time','Duration','Avg_x', 'Avg_y', 'Angle'])

    for i in range(len(miss)):
        miss[i] = miss[i].replace('\t', ' ')
    split_miss = np.char.split(miss)
    with open('miss.csv', 'w') as f:
        write = csv.writer(f)
        write.writerows(split_miss)

    for i in range(len(sacc)):
        sacc[i] = sacc[i].replace('\t', ' ')
    split_sacc = np.char.split(sacc)
    for i in reversed(range(np.shape(split_sacc)[0])):
        if split_sacc[i][0] == "SSACC":
            split_sacc = np.delete(split_sacc, i, 0)
    np_esacc = np.asarray([np.array(line) for line in split_sacc])
    reverse_esacc = np_esacc[::-1]
    pd.DataFrame(reverse_esacc).to_csv('esacc.csv', index_label = "Index", header  = ['Type','Eye','Start_time','End_time','Duration','cood_1', 'cood_2', 'cood_3', 'cood_4','cood_5', 'cood_6'])
        
    for i in range(len(blink)):
        blink[i] = blink[i].replace('\t', ' ')
    split_blink = np.char.split(blink)
    for i in reversed(range(np.shape(split_blink)[0])):
        if split_blink[i][0] == "SBLINK":
            split_blink = np.delete(split_blink, i, 0)
    np_blink = np.asarray([np.array(line) for line in split_blink])
    reverse_blink = np_blink[::-1]
    pd.DataFrame(reverse_blink).to_csv('eblink.csv', index_label = "Index", header  = ['Type','Eye','Start_time','End_time','Duration'])


def match_movement_data():
    headers = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6']
    data_f = pd.read_csv('data.csv', sep=',', skiprows = 1, header=None, names=headers, dtype={"col1": int, "col2": float, "col3": object, "col4": object, "col5": float, "col6": float})
    data = data_f.to_numpy()
    # Fixation
    headers = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9']
    fix_f = pd.read_csv('efix.csv', sep=',', skiprows = 1, header=None, names=headers, dtype={"col1": int, "col2": object, "col3": object, "col4": float, "col5": float, "col6": float, "col7": float, "col8": float, "col9": float})
    fix_marker = fix_f.to_numpy()

    # Blink
    headers = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6']
    blink_f = pd.read_csv('eblink.csv', sep=',', skiprows = 1, header=None, names=headers, dtype={"col1": int, "col2": object, "col3": object, "col4": float, "col5": float, "col6": float})
    blink_marker = blink_f.to_numpy()

    #sacc
    headers = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10', 'col11', 'col12']
    sacc_f = pd.read_csv('esacc.csv', sep=',', skiprows = 1, header=None, names=headers, dtype={"col1": int, "col2": object, "col3": object, "col4": float, "col5": float, "col6": float, "col7": float, "col8": float, "col9": float, "col10": float, "col11": float, "col12": float})
    sacc_marker = sacc_f.to_numpy()

    data_marker = np.empty((np.shape(data)[0], 2), dtype=object)
    data_marker[:] = ""

    fix_index = 0
    blink_index = 0
    sacc_index = 0
    for i in range(np.shape(data)[0]):
        record = data[i]
        current_time = record[1]
        if fix_index < len(fix_marker) and fix_marker[fix_index][4] < current_time:
            fix_index += 1
        if blink_index < len(blink_marker) and blink_marker[blink_index][4] < current_time:
            blink_index += 1
        if sacc_index < len(sacc_marker) and sacc_marker[sacc_index][4] < current_time:
            sacc_index += 1

        if fix_index < len(fix_marker) and current_time >= fix_marker[fix_index][3] and current_time <= fix_marker[fix_index][4]:
            data_marker[i] = ["fix", fix_marker[fix_index][0]]
        elif blink_index < len(blink_marker) and current_time >= blink_marker[blink_index][3] and current_time <= blink_marker[blink_index][4]:
            data_marker[i] = ["blink", blink_marker[blink_index][0]]
        elif sacc_index < len(sacc_marker) and current_time >= sacc_marker[sacc_index][3] and current_time <= sacc_marker[sacc_index][4]:
            data_marker[i] = ["sacc", sacc_marker[sacc_index][0]]
        else:
            continue
    
    new_data = np.concatenate((data, data_marker), axis = 1)
    pd.DataFrame(new_data).to_csv('match_data.csv', index = False, header  = ['Index', 'Time','x','y','pupil size','CR', 'event marker', 'event index'])


def match_behavior_data():
    headers = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8']
    data_f = pd.read_csv('match_data.csv', sep=',', skiprows = 1, header=None, names=headers, dtype={"col1": int, "col2": float, "col3": object, "col4": object, "col5": float, "col6": float, "col7": object, "col8": object})
    data = data_f.to_numpy()
    headers = ['col1', 'col2', 'col3', 'col4']
    msg_f = pd.read_csv('msg.csv', sep=',', skiprows = 1, header=None, names=headers, dtype={"col1": int, "col2": object, "col3": float, "col4": object})
    msg = msg_f.to_numpy()

    msg_marker = np.empty((np.shape(data)[0], 2), dtype=object)
    msg_marker[:] = ""

    current_msg_index = 0
    current_data_index = 0
    while current_msg_index < np.shape(msg)[0]:
        if "forcedchoice_start" in msg[current_msg_index][3]:
            [current_data_index, current_msg_index, msg_marker] = process_forcedchoice(msg_marker, msg, data, current_data_index, current_msg_index)
        elif "forcedmemory_start" in msg[current_msg_index][3]:
            [current_data_index, current_msg_index, msg_marker] = process_forcedmemory(msg_marker, msg, data, current_data_index, current_msg_index)
        current_msg_index += 1
    pd.DataFrame(msg_marker).to_csv('msg_marker.csv', index = True, header  = ['event', 'fix'])

    matched_data_pic = np.concatenate((data, msg_marker), axis = 1)
    pd.DataFrame(matched_data_pic).to_csv('final_data.csv', index = False, header  = ['Index', 'Time','x','y','pupil size','CR', 'movement_marker', 'movement_index', 'pic_marker', 'behavior'])


if __name__ == '__main__':
    removed_header = clean_calibration("./J030701.asc")
    extract_raw_to_data_movement(removed_header)
    match_movement_data()
    match_behavior_data()