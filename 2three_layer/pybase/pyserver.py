#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pyserver.py
# Create date : 2016-09-22 10:42
# Modified date : 2018-10-30 13:46
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
import tornado.httpserver
import tornado.ioloop

if sys.version > '3':
    import asyncio
    from pybase import pycount
    from pybase import pyobject
    from pybase import pyhttp
    from pybase import pyjson
    from pybase import pylog
    from pybase import pythread
else:

    import pycount
    import pyobject
    import pyhttp
    import pyjson
    import pylog
    import pythread

interface_mapping = {}

def get_request_type_get():
    return pyjson.get_request_get()

def get_request_type_post():
    return pyjson.get_request_post()

def get_http_head():
    return pyhttp.HTTP_HEAD

def get_return_message(body):
    message = "%s%d\r\n\r\n%s" % (get_http_head(), len(body), body)
    return message

def no_support_url_method(request):
    content = request.uri
    return pyjson.json_method_not_support(content, request.method)

def deal_with_func(request,
                    method_get,
                    method_post=no_support_url_method,
                    other_method=no_support_url_method):
    if request.method == get_request_type_get():
        return method_get(request)
    elif request.method == get_request_type_post():
        return method_post(request)
    else:
        return other_method(request)

class DLServer(pyobject.BaseObject):
    def __init__(self, proc_name, listen_port):
        super(DLServer, self).__init__()
        self.proc_name = proc_name
        self.listen_port = listen_port
        self.interface_mapping = {}
        self.func_list = []
        self.state = {}
        self.delay = 0

    def get_proc_name(self):
        return self.proc_name

    def get_listen_port(self):
        return self.listen_port

    def get_interface_mapping_dic(self):
        return self.interface_mapping

    def start(self,delay):
        pylog.info("thread id:%s" % pythread.get_thread_id())
        self.map_url_handler()
        self.start_server(delay)
        return True

    def work(self):
        t = threading.Thread(target=self.start,args=(self.delay,))
        t.start()

    def start_server(self,delay):
        if sys.version > '3':
            asyncio.set_event_loop(asyncio.new_event_loop())
        http_server = tornado.httpserver.HTTPServer(self.deal_request, no_keep_alive=False)
        pylog.info("listen_port:%s" % self.get_listen_port())
        http_server.listen(self.get_listen_port())
        if delay:
            tornado.ioloop.IOLoop.instance().call_later(delay,tornado.ioloop.IOLoop.instance().stop)
        tornado.ioloop.IOLoop.instance().start()

    def map_url_handler(self):
        self.func_list.append(self.stop_server)
        self.func_list.append(self.show_all_interface)
        self.func_list.append(self.show_interface_count)

        for func in self.func_list:
            self.make_handler(func)

    def make_handler(self,func):
        interface_name = self.get_interface_name(func)
        pylog.info("interface_name:%s" % interface_name)
        interface_mapping_dic = self.get_interface_mapping_dic()
        interface_mapping_dic[interface_name] = func

    def get_interface_name(self, func):
        proc_name = self.get_proc_name()
        return "/%s/%s" % (proc_name, func.__name__)

    def add_interface_count(self, func):
        interface_name = self.get_interface_name(func)
        pycount.add_count_with_key(interface_name)
        pycount.add_count_with_key("all_request_count")
        pycount.show_interface_load_count_dic()

    def add_interface_url_error_count(self):
        pycount.add_count_with_key("error_url_request_count")

    def check_interface_key(self,request):
        interface_mapping_dic = self.get_interface_mapping_dic()
        if sys.version > '3':
            if request.path.lower() in interface_mapping_dic:
                return True
            else:
                return False
        return interface_mapping_dic.has_key(request.path.lower())

    def deal_request(self,request):
        pylog.debug(request)
        path_lower =request.path.lower()
        pylog.debug("path_lower:%s" % path_lower)
        body = None
        interface_mapping_dic = self.get_interface_mapping_dic()
        if not self.check_interface_key(request):
            body = pyjson.json_method_not_support(request.uri)
            pylog.warning(request)
            pylog.warning(body)
            self.add_interface_url_error_count()
        else:
            url = request.path.lower()
            handler = interface_mapping_dic[url]
            self.add_interface_count(handler)
            body = handler(request)
        message = get_return_message(body)
        #pylog.info("request_return:%s" % message)
        #pylog.info(body)
        if sys.version > '3':
            request.write(bytes(message,encoding='utf-8'))
        else:
            request.write(message)
        request.finish()

    def stop_server(self,request):
        '''GET  停止WEB服务器'''
        return deal_with_func(request, self.stop_server_get,)

    def stop_server_get(self,request):
        ret = "server will stop in a second"
        tornado.ioloop.IOLoop.instance().call_later(1,tornado.ioloop.IOLoop.instance().stop)
        return pyjson.json_ok(ret, request.method)

    def show_all_interface(self,request):
        '''GET 显示所有接口以及接口描述和参数信息'''
        return deal_with_func(request, self.show_all_interface_get)

    def show_all_interface_get(self,request):
        interface_dic = {}
        interface_mapping_dic = self.get_interface_mapping_dic()
        for key in interface_mapping_dic:
            doc = interface_mapping_dic[key].__doc__
            if not doc:
                doc = pyjson.get_str_no_description()

            interface_dic[key] = doc
        content = interface_dic
        return pyjson.json_ok(content, request.method)

    def show_interface_count(self,request):
        '''GET 显示所有接口的访问计数'''
        return deal_with_func(request, self.show_interface_count_get)

    def show_interface_count_get(self,request):
        interface_count_dic = pycount.get_interface_load_count_dic()
        content = interface_count_dic
        return pyjson.json_ok(content, request.method)
