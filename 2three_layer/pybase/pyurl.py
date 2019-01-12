#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pyurl.py
# Create date : 2014-01-14 15:48
# Modified date : 2014-01-17 16:06
# Author : DARREN
# Describe : get url with proxy
# Email : lzygzh@126.com
#####################################

import pyproxy
import random
import pypickle
import string
import pylog
import urllib
import re

delete_proxy_count = 0

def get_page_size():
    return "10"

def get_proxy_page_file_name():
    return "proxy_page"

def get_retry_times():
    return 5

def get_proxy_list_min_count():
    return 80

def get_max_delete_count():
    return 10

def get_proxy_list_file_name():
    return "proxy_list"

def get_check_url():
    return "http://www.mianbao.com/"

def get_check_timeout():
    return 3

def get_timeout():
    return 30

def read_proxy_page():
    file = get_proxy_page_file_name()
    page = pypickle.read(file)
    return page

def save_proxy_page(page):
    file = get_proxy_page_file_name()
    pypickle.save(page,file)

def read_proxy_list():
    proxy_list_name = get_proxy_list_file_name()
    return pypickle.read(proxy_list_name)

def save_proxy_list(proxy_list):
    proxy_list_name = get_proxy_list_file_name()
    pypickle.save(proxy_list, proxy_list_name)

def get_proxy_item():
    proxy_list = read_proxy_list()
    rd = random.randint(0,len(proxy_list)-1)
    item = proxy_list[rd]
    return item

def send_request(item, url=get_check_url(), timeout=get_timeout()):
    try:
        page = pyproxy.get_url_with_proxy(item, url, timeout=timeout)
    except Exception, e:
        print "get bread: %s" % e
        pylog.error(e)
        return False
    return page

def get_url_with(url,item):
    page = send_request(item, url)
    if not page:
        delete_proxy(item)
        return False
    return page

def delete_proxy(item):
    if item:
        proxy_list = read_proxy_list()
        proxy_list.remove(item)
        print "proxy list count : %s" % len(proxy_list)
        save_proxy_list(proxy_list)
        change_delete_count()

def change_delete_count():
    global delete_proxy_count
    max_delete_count = get_max_delete_count()
    if delete_proxy_count == max_delete_count:
        print "get new proxys"
        check_proxy_list_count()
        delete_proxy_count = 0
    else:
        print "delete proxy"
        delete_proxy_count = delete_proxy_count + 1

def check_404(page):
    ret_list = re.findall("404 Not Found" , page, re.S)
    if ret_list:
        return ret_list
    return False

def check_fault(page):
    try:
        page = page.decode("gb2312","ignore")
    except Exception ,e:
        print e
    ret_list = re.findall(u"错误：您所请求的网址（URL）无法获取" , page, re.S)
    if ret_list:
        print ret_list
        return ret_list
    return False

def get_url(url, name="baidu",check_url=get_check_url(), use_proxy=True):
    if use_proxy == True:
        check_proxy_list_count(check_url)
        max_count = get_retry_times()
        count = 0
        while count < max_count:
            item  = get_proxy_item()
            page = get_url_with(url,item)
            if page:
                ret = check_404(page)
                if ret:
                    print "find 404"
                    delete_proxy(item)
                    return False

                ret = check_fault(page)
                if ret:
                    print "find fault"
                    delete_proxy(item)
                    return False
                write_page(page,name)
                return page
            else:
                count = count + 1
        return False
    else:
        try:
            ret = urllib.urlopen(url)
        except Exception,e:
            print 'get web html %s' % e
            return False
        return ret.read()

def get_proxy_list():
    page = read_proxy_page()
    print "page : %s" % page
    page_size = get_page_size()
    if page:
        page = "%s" % (string.atoi(page) + 1)
        save_proxy_page(page)
    else:
        page = '1'
        save_proxy_page(page)
    return pyproxy.get_enable_proxy_list(page=page, page_size=page_size)

def extend_proxy_list(lt):
    proxy_list = read_proxy_list()
    if proxy_list:
        proxy_list.extend(lt)
    else:
        proxy_list = lt
    return proxy_list

def filter_proxy_list(filter_url):
    lt = get_proxy_list()
    real_lt= []
    timeout = get_check_timeout()
    for item in lt:
        print item
        try:
            content = send_request(item, filter_url, timeout=timeout)
        except Exception, e:
            pylog.error(e)
            continue
        if content:
            real_lt.append(item)
            print "succ : %s" % item
    return real_lt

def update_proxy_list(check_url):
    lt = filter_proxy_list(check_url)
    if not lt:
        pylog.error("do not get any proxy")
        print "do not get any proxy"
        return False

    proxy_list = extend_proxy_list(lt)
    pylog.info("now proxy count is %s" % len(proxy_list))
    save_proxy_list(proxy_list)
    return proxy_list

def check_proxy_list_count(check_url=get_check_url()):
    count = get_proxy_list_min_count()
    proxy_list = read_proxy_list()

    if proxy_list:
        pylog.info("now proxy count is %s" % len(proxy_list))
    else:
        pylog.info("now proxy count is 0")

    proxy_list_count = 0
    if proxy_list:
        proxy_list_count = len(proxy_list)

    if proxy_list_count < count:
        while proxy_list_count < count:
            update_proxy_list(check_url)
            proxy_list = read_proxy_list()
            if proxy_list:
                proxy_list_count = len(proxy_list)

def write_page(page,name,pre="test"):
    f = open("%s%s.html" % (pre,name), 'w')
    f.write(page)
    f.close()

if __name__ == "__main__":
    url = "http://www.baidu.com"
    get_url(url)


