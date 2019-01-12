#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pylinux.py
# Create date : 2013-12-13 13:52
# Modified date : 2018-10-24 22:31
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################

import sys
import os
import platform
import re
import socket
import time
import fcntl
import struct
from collections import namedtuple

if sys.version > '3':
    import urllib.request as urllib2
    import subprocess as commands
    from pybase import etc
    from pybase import pyfile
else:
    import urllib2
    import commands
    import etc
    import pyfile

def _get_long_value(value):
    if sys.version > '3':
        return int(value)
    return long(value)

def get_linux_apps_list(name="linux_softs",path=etc.ENV_FILE_PATH):
    con = os.popen("dpkg -l").read()
    path = "%s/python%s/" % (path, platform.python_version())
    pyfile.write_file(con=con, name=name, path=path)

#获取服务器ip

def get_inner_ip():
    '''获取内网ip'''
    host_name = socket.getfqdn(socket.gethostname())
    ip = socket.gethostbyname(host_name)
    return ip

def get_outter_ip():
    '''获取外网ip'''
    outter_ip = re.search(r"\d+\.\d+\.\d+\.\d+", urllib2.urlopen("http://www.whereismyip.com").read()).group(0)
    return outter_ip

def get_host_name():
    '''获取主机名字'''
    host_name = socket.getfqdn(socket.gethostname())
    return host_name

def get_system_net_name():
    '''获取计算机网络名称'''
    net_name = platform.node()
    return net_name

#   def show_system_net_name():
#       '''获取系统网络名称'''
#       name = get_system_net_name()
#       return name

def get_system_name():
    '''获取系统名称'''
    system_name = platform.system()
    return system_name

def get_system_kernel_info():
    '''获取系统内核信息'''
    info = os.popen("uname -a")
    info = info.read()
    return info


def get_system_release_info():
    '''获取系统发行信息'''
    info = os.popen("cat /etc/issue")
    info = info.read()
    return info

def get_system_name_version():
    '''获取系统名字和版本信息'''
    version = platform.platform()
    return version

def get_platform_unname():
    '''获取总体系统信息'''
    u_info = platform.uname()
    return u_info

def get_sys_info(name="sys_info",path=etc.ENV_FILE_PATH):
    con = ""

    doc = get_host_name.__doc__
    ret = get_host_name()
    con = "%s%s:%s\n" % (con,doc,ret)

    doc = get_system_net_name.__doc__
    ret = get_system_net_name()
    con = "%s%s:%s\n" % (con,doc,ret)

    doc = get_system_name.__doc__
    ret = get_system_name()
    con = "%s%s:%s\n" % (con,doc,ret)

    doc = get_system_kernel_info.__doc__
    ret = get_system_kernel_info()
    con = "%s%s:%s\n" % (con,doc,ret)

    doc = get_system_release_info.__doc__
    ret = get_system_release_info()
    con = "%s%s:%s\n" % (con,doc,ret)

    doc = get_system_name_version.__doc__
    ret = get_system_name_version()
    con = "%s%s:%s\n" % (con,doc,ret)

    doc = get_platform_unname.__doc__
    ret = get_platform_unname()
    con = "%s%s:%s\n" % (con,doc,ret)

    doc = get_supported_dists.__doc__
    ret = get_supported_dists()
    con = "%s%s:%s\n" % (con,doc,ret)

    doc = get_architecture.__doc__
    ret = get_architecture()
    con = "%s%s:%s\n" % (con,doc,ret)

    doc = get_cpu_count.__doc__
    ret = get_cpu_count()
    con = "%s%s:%s\n" % (con,doc,ret)

    path = "%s/python%s/" % (path, platform.python_version())
    pyfile.write_file(con=con, name=name, path=path)

def get_cpu_count():
    '''获取cpu核数'''
    f = open("/proc/cpuinfo")
    lines = f.readlines()
    f.close()
    res = 0
    for line in lines:
        line = line.lower().lstrip()
        if line.startswith('processor'):
            res = res +1
    return res

def get_supported_dists():
    '''获取支持的盘类型信息'''
    info = platform._supported_dists
    return info

def get_architecture():
    '''获取操作系统位数'''
    info = platform.architecture()
    return info


def get_process_info():
    '''获取当前进程的信息'''
    pid = os.getpid()
    res = commands.getstatusoutput('ps aux|grep ' + str(pid))[1].split('\n')
    p = re.compile(r'\s+')
    l = p.split(res[0])
    info = {'user':l[0],
            'pid':l[1],
            'cpu':l[2],
            'mem':l[3],
            'vsa':l[4],
            'rss':l[5],
            'tty':l[6],
            'stat':l[7],
            'start':l[8],
            'time':l[9],
            'command':' '.join(l[10:]).strip()}
    return info

def read_cpu_info():
    f = open('/proc/stat')
    lines = f.readlines()
    f.close()
    counters = None
    for line in lines:
        line = line.lstrip()
        counters = line.split()
        if len(counters) < 5:
            continue
        if counters[0].startswith('cpu'):
            break
    total = 0
    idle = None
    if sys.version > '3':
        for i in range(1, len(counters)):
            total = total + _get_long_value(counters[i])
        idle = _get_long_value(counters[4])
   #     return {'total':total, 'idle':idle}
    else:
        for i in xrange(1, len(counters)):
            total = total + _get_long_value(counters[i])
        idle = _get_long_value(counters[4])
    return {'total':total, 'idle':idle}

def _calc_cpu_usage(counters1, counters2):
    idle = counters2['idle'] - counters1['idle']
    #print "cpu idle:%s" % idle
    total = counters2['total'] - counters1['total']
    #print "cpu total:%s" % total
    return 100 - (idle * 100)/total

def get_use_cpu():
    counters1 = read_cpu_info()
    time.sleep(5)
    counters2 = read_cpu_info()
    return _calc_cpu_usage(counters1, counters2)

def read_mem_info():
    res = {'total':0, 'free':0, 'buffers':0, 'cached':0}
    f = open('/proc/meminfo')
    lines = f.readlines()
    f.close()
    i = 0
    for line in lines:
        if i == 4:
            break
        line = line.lstrip()
        mem_item = line.lower().split()
        if mem_item[0] == 'memtotal:':
            res['total'] = _get_long_value(mem_item[1])
            i = i + 1
            continue
        elif mem_item[0] == 'memfree:':
            res['free'] = _get_long_value(mem_item[1])
            i = i + 1
            continue
        elif mem_item[0] == 'buffers:':
            res['buffers'] = _get_long_value(mem_item[1])
            i = i + 1
            continue
        elif mem_item[0] == 'cached:':
            res['cached'] = _get_long_value(mem_item[1])
            i = i + 1
            continue
    return res

def calc_mem_usage(counters):
    used = counters['total'] - counters['free'] - counters['buffers'] - counters['cached']
    total = counters['total']
    return used * 100 / total

def get_use_mem():
    counters = read_mem_info()
    return calc_mem_usage(counters)

def read_net_info(dev):
    f = open('/proc/net/dev')
    lines = f.readlines()
    f.close()
    res = {'in':0, 'out':0}
    for line in lines:
        if line.lstrip().startswith(dev):
            #for centos
            line = line.replace(':', ' ')
            items = line.split()
            res['in'] = _get_long_value(items[1])
            res['out'] = _get_long_value(items[len(items)/2 + 1])
    return res

def get_use_net():
    '''先要获取去设备'''
    return read_net_info('eth0')

def get_dev_address(dev):
    '''获取网卡设备的ip地址'''
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    a = s.fileno()
    b = 0x8915
    c = struct.pack('256s', dev[:15])
    res = fcntl.ioctl(a, b, c)[20:24]
    return socket.inet_ntoa(res)


def get_cpu_info():
    cpuinfo = {}
    procinfo = {}
    nprocs = 0

    with open('/proc/cpuinfo') as f:
        for line in f:
            if not line.strip():
                cpuinfo['proc%s' % nprocs] = procinfo
                nprocs = nprocs + 1
                procinfo = {}
            else:
                if len(line.split(':')) == 2:
                    procinfo[line.split(':')[0].strip()] = line.split(':')[1].strip()
                else:
                    procinfo[line.split(':')[0].strip()] = ''
    return cpuinfo

def disk_partitions(is_all=False):
    '''获取当前操作系统下的所有磁盘'''
    disk_ntuple = namedtuple('partition', 'device mountpoint fstype')
    usage_ntuple = namedtuple('usage', 'total used free percent')
    phydevs = []
    f = open("/proc/filesystems", 'r')
    for line in f:
        if not line.startswith("nodev"):
            phydevs.append(line.strip())
    #print(phydevs)

    retlist = []
    f = open('/etc/mtab', "r")
    for line in f:
        if not is_all and line.startswith('none'):
            continue
        fields = line.split()
        device = fields[0]
        mountpoint = fields[1]
        fstype = fields[2]
        if not is_all and fstype not in phydevs:
            continue
        if device == 'none':
            device = ''

        ntuple = disk_ntuple(device, mountpoint, fstype)
        retlist.append(ntuple)
    return retlist

def get_fs_info(path):
    '''统计某磁盘使用情况，返回对象'''
    hddinfo = os.statvfs(path)
    total = hddinfo.f_frsize * hddinfo.f_blocks
    free = hddinfo.f_frsize * hddinfo.f_bavail
    used = hddinfo.f_frsize * (hddinfo.f_blocks - hddinfo.f_bfree)
    try:
        percent = (float(used) / total) * 100
    except ZeroDivisionError:
        percent = 0
    return {'total' : int(float(total)),
            'free' : int(float(free)),
            'used' : int(float(used)),
            'percent' : int(float(percent)),}

def get_mounted_disks():
    disk_list = []
    with open('/proc/mounts', 'r') as f:
        mounts = f.readlines()

    for mount in mounts:
        if mount.startswith('/dev/'):
            disk_dict = {}
            mount = mount.split()
            dev_path = mount[0]
            target = mount[1]
            fs_type = mount[2]
            if target == '/':
                disk_dict['file_system_category'] = 'root'
            else:
                disk_dict['file_system_category'] = 'logical'

            disk_dict['mount_target'] = target
            disk_dict['fs_type'] = fs_type
            disk_dict['real_path'] = os.path.realpath(dev_path)
            dic = get_fs_info(target)
            for item in dic:
                disk_dict[item] = dic[item]

            disk_list.append(disk_dict)

    return disk_list

def get_memory_stat():
    '''内存信息 / meminfo'''
    mem = {}
    f = open("/proc/meminfo")
    lines = f.readlines()
    f.close()
    for line in lines:
        if len(line) < 2:
            continue
        name = line.split(':')[0]
        var = line.split(':')[1].split()[0]
        mem[name] = _get_long_value(var) * 1024

    mem['MemUsed'] = mem['MemTotal'] - mem['MemFree'] - mem['Buffers'] - mem['Cached']
    return mem

def load_stat():
    '''cpu负载 /loadavg'''
    loadavg = {}
    f = open("/proc/loadavg")
    con = f.read().split()
    f.close()
    loadavg['lavg_1'] = con[0]
    loadavg['lavg_5'] = con[1]
    loadavg['lavg_15'] = con[2]
    loadavg['nr'] = con[3]
    loadavg['last_pid'] = con[4]
    return loadavg

def uptime_stat():
    uptime = {}
    f = open("/proc/uptime")
    con = f.read().split()
    f.close()
    all_sec = float(con[0])
    minute, hour, day = 60, 3600, 86400
    uptime['day'] = int(all_sec / day)
    uptime['hour'] = int((all_sec % day) / hour)
    uptime['minute'] = int((all_sec % hour)/ minute)
    uptime['second'] = int(all_sec % minute)
    uptime['Free rate'] = float(con[1]) /float(con[0])
    return uptime

# pylint: disable=bad-continuation
def get_net_stat():
    net = []
    f = open("/proc/net/dev")
    lines = f.readlines()
    f.close()
    for line in lines[2:]:
        con = line.split()

        intf = dict(zip(
                        (
                        'ReceiveBytes',
                        'ReceivePackets',
                        'ReceiveErrs',
                        'ReceiveDrop',
                        'ReceiveFifo',
                        'ReceiveFrames',
                        'ReceiveCompressed',
                        'ReceiveMulticast',
                        'TransmitBytes',
                        'TransmitPackets',
                        'TransmitErrs',
                        'TransmitDrop',
                        'TransmitFifo',
                        'TransmitFrames',
                        'TransmitCompressed',
                        'TransmitMulticast',
                        ),
                        (
                        con[0].split(":")[1],
                        int(con[1]),
                        int(con[2]),
                        int(con[3]),
                        int(con[4]),
                        int(con[5]),
                        int(con[6]),
                        int(con[7]),
                        int(con[8]),
                        int(con[9]),
                        int(con[10]),
                        int(con[11]),
                        int(con[12]),
                        int(con[13]),
                        int(con[14]),
                        int(con[15]),
                        )
                       )
                )
        net_item = {}
        net_item['interface'] = con[0].split(":")[0]
        net_item['stat'] = intf
        net.append(net_item)

    return net
# pylint: enable=bad-continuation
