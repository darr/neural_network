#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pyhttp.py
# Create date : 2015-02-25 15:26
# Modified date : 2018-10-29 21:36
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import sys

import tornado.httpclient

if sys.version > '3':
    from pybase import pylog
    from pybase import pyjson
else:
    import pylog
    import pyjson

HTTP_HEAD = "HTTP/1.1 200 OK\r\nAccess-Control-Allow-Origin:*\r\nConnection:Keep-Alive\r\nContent-Length:"

def http_post(url, data):
    method = "POST"
    client = tornado.httpclient.HTTPClient()
    try:
        request = tornado.httpclient.HTTPRequest(url=url, method=method, body=data)
        response = client.fetch(request)
        return (True, response.body)
    except tornado.httpclient.HTTPError as err:
        #pylog.error(err)
        pylog.error("err:%s url:%s, data:%s" % (err, url, data))
        return (False, err.message)

def http_get(url):
    method = "GET"
    client = tornado.httpclient.HTTPClient()
    try:
        request = tornado.httpclient.HTTPRequest(url=url, method=method)
        response = client.fetch(request)
        return (True, response.body)
    except tornado.httpclient.HTTPError as err:
        pylog.error("err:%s url:%s" % (err, url))
        return (False, err.message)

def get_json_url_value(url):
    isSuc, ret = http_get(url)
    if isSuc:
        dic = pyjson.json_to_dict(ret)
        return dic["data"]

    return False

def send(url, data):
    data = pyjson.dict_to_json(data)
    return http_post(url, data)
