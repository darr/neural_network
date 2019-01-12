#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pyproxy.py
# Create date : 2014-01-10 16:14
# Modified date : 2014-01-12 11:58
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################

import urllib
import json
import socket

def get_enable_proxy_list(page='1', page_size='10'):
    url = get_proxy_interface_url(page, page_size)
    ret = ''
    try:
        ret = urllib.urlopen(url)
    except Exception, e:
        print e
    json_str = ret.read()
    dic = json.loads(json_str)
    count = 0
    if dic['result'] == "ok":
        data= dic['data']['data']
    return data

def get_proxy_interface_url(page, page_size):
    ip = '42.96.136.234'
    port = '8181'
    url = "http://%s:%s/pyproxy/query?page=%s&page_size=%s" %(ip, port, page, page_size)
    return url

def get_url_with_proxy(proxy, url="http://www.baidu.com", timeout=5):
    proxy_ip = proxy[1]
    proxy_port = proxy[2]
    ret = ''
    socket.setdefaulttimeout(timeout)
    try:
        ret = urllib.urlopen(url,proxies = {'http':'http://%s:%s' % (proxy_ip, proxy_port)})
    except Exception,e:
        print 'get web html %s' % e
        return False

    return ret.read()

def work():
    list = get_enable_proxy_list()
    for item in list:
        print item
        #url = get_proxy_interface_url(page='1', page_size='10')
        url = "http://www.hbsc.cn"
        print get_url_with_proxy(item, url)

def run():
    work()

if __name__ == '__main__':
    run()

