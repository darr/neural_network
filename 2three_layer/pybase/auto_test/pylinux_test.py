#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pylinux_test.py
# Create date : 2018-10-10 00:28
# Modified date : 2018-10-24 22:07
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pybase import pylinux
from pybase import pylog

def _show_test(con,value):
    if False:
        print("%s:%s" % (con,value))

def _get_inner_ip_test():
    info = pylinux.get_inner_ip()
    pylog.test_compare_value(info,"127.0.1.1")
    _show_test("内网ip",info)

def _get_outter_ip_test():
    ip = pylinux.get_outter_ip()
    _show_test("外网ip",info)

def _get_host_name_test():
    info = pylinux.get_host_name()
    _show_test("主机名字",info)

def _get_system_net_name_test():
    info = pylinux.get_system_net_name()
    _show_test("计算机网络名称",info)

def _get_system_name_test():
    info = pylinux.get_system_name()
    _show_test("系统名称",info)

def _get_system_kernel_info_test():
    info = pylinux.get_system_kernel_info()
    _show_test("系统内核信息",info)

def _get_system_release_info_test():
    info = pylinux.get_system_release_info()
    _show_test("系统发行信息",info)

def _get_system_name_version_test():
    info = pylinux.get_system_name_version()
    _show_test("系统名字和版本信息",info)

def _get_platform_uname_test():
    info = pylinux.get_platform_unname()
    _show_test("总体系统信息",info)

def _get_supported_dists_test():
    info = pylinux.get_supported_dists()
    _show_test("支持的盘类型信息",info)

def _get_architecture_test():
    info = pylinux.get_architecture()
    _show_test("操作系统位数",info)

def _get_process_info_test():
    info = pylinux.get_process_info()
    _show_test("当前进程信息",info)

def _read_cpu_info_test():
    info = pylinux.read_cpu_info()
    _show_test("读取cpu信息",info)

def _get_use_cpu_test():
    info = pylinux.get_use_cpu()
    _show_test("cpu使用",info)

def _get_use_mem_test():
    info = pylinux.get_use_mem()
    _show_test("内存使用",info)

def _get_use_net_test():
    info = pylinux.get_use_net()
    _show_test("使用网络",info)

def _get_cpu_count_test():
    info = pylinux.get_cpu_count()
    _show_test("cpu核数",info)

def _get_cpu_info_test():
    info = pylinux.get_cpu_info()
    _show_test("cpu信息",info)

def _disk_partitions_test():
    info = pylinux.disk_partitions()
    _show_test("当前操作系统下所有磁盘",info)

def _get_mounted_disks_test():
    info = pylinux.get_mounted_disks()
    _show_test("挂在的磁盘",info)

def _get_memory_stat_test():
    info = pylinux.get_memory_stat()
    _show_test("内存信息/meminfo",info)

def _load_stat_test():
    info = pylinux.load_stat()
    _show_test("cpu 负载/loadavg",info)

def _uptime_stat_test():
    info = pylinux.uptime_stat()
    _show_test("开机时间状态",info)

def _get_net_stat_test():
    info = pylinux.get_net_stat()
    _show_test("获取网络状态",info)

def _test_get_linux_apps_list():
    pylinux.get_linux_apps_list()

def _test_get_sys_info():
    pylinux.get_sys_info()

def test():
    _get_inner_ip_test()
    #_get_outter_ip_test() #局域网机器运行会失败 需要有外网ip的机器
    _get_host_name_test()
    _get_system_net_name_test()
    _get_system_name_test()
    _get_system_kernel_info_test()
    _get_system_release_info_test()
    _get_system_name_version_test()
    _get_platform_uname_test()
    _get_supported_dists_test()
    _get_architecture_test()
    _get_process_info_test()
    _read_cpu_info_test()
    #_get_use_cpu_test()
    _get_use_mem_test()
    _get_use_net_test()
    _get_cpu_count_test()
    #_get_cpu_info_test()
    _disk_partitions_test()
    _get_mounted_disks_test()
    #_get_memory_stat_test()
    _load_stat_test()
    _uptime_stat_test()
    _get_net_stat_test()
    _test_get_linux_apps_list()
    _test_get_sys_info()
