#!/usr/bin/python3
import sqlite3
import numpy as np
import configparser
from glob import glob
from time import sleep, strftime
from datetime import datetime

config = configparser.ConfigParser()
config.read('../config.ini')

def on_off_bin(ac):
    if ac == 'on':
        return 1
    else:
        return 0

def fmt_tup(tup):
    setpoint = float(config['RLCONTROL']['SETPOINT'])
    return ((tup[1] - setpoint) / 20, tup[2] / 100, tup[3] / 2, on_off_bin(tup[4]))

def data_stream():
    conn = sqlite3.connect(config['PATHS']['DB_PATH'], timeout=30000)
    c = conn.cursor()
    res = c.execute("""select date, temp, humid, vpd, ac from environ order by rowid desc limit 5;""")
    data = [val for val in res]
    c.close()
    data = data[::-1]
    tm = (int(data[0][0][11:13]) - 12.) / 12
    data = list(map(fmt_tup, data))
    data = [x for v in data for x in v]
    return data + [tm]

def vpd_lin_trans(vpd, min_v=1.0, max_v=2.0, max_rate=1.0, min_rate=0.15):
    vpd = (np.clip(vpd, min_v, max_v) - min_v) / (max_v - min_v)
    return 1  / (vpd * (max_rate - min_rate) + min_rate)

def on_off(dev_idx, duration):
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("localhost", 9000))
    data = dev_idx + " on"
    sock.sendall(data.encode())
    result = sock.recv(1024).decode()
    print(result)
    sock.close()

    sleep(duration)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("localhost", 9000))
    data = dev_idx + " off"
    sock.sendall(data.encode())
    result = sock.recv(1024).decode()
    print(result)
    sock.close()

def day_time():
    start_hour = int(config['RLCONTROL']['ON_START'])
    end_hour = int(config['RLCONTROL']['ON_DURATION'])
    end_hour = (start_hour + end_hour) % 24
    cur_hour = datetime.now().time().hour
    return (cur_hour < end_hour) or (cur_hour >= start_hour)

