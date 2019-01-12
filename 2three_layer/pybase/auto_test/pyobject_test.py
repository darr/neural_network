#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pyobject_test.py
# Create date : 2018-10-10 14:07
# Modified date : 2018-10-15 18:53
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from pybase.pyobject import BaseObject
from pybase import pybaselog

class Human(BaseObject):
    ''' a class Human'''
    def __init__(self):
        super(Human, self).__init__()
        self.sex = 'male'
        self.age = '20'
        self.name = 'person_name'

    def listen(self):
        con = "%s %s" % (self.__class__.__name__, sys._getframe().f_code.co_name)
        return con

    def look(self):
        con = "%s %s" % (self.__class__.__name__, sys._getframe().f_code.co_name)
        return con

    def eat(self):
        con = "%s %s" % (self.__class__.__name__, sys._getframe().f_code.co_name)
        return con

class Chinese(Human):
    def __init__(self):
        super(Human, self).__init__()
        self.lang = "Chinese"
        self.country = "China"

    def speak(self):
        con = "%s %s :%s" % (self.__class__.__name__, sys._getframe().f_code.co_name, self.lang)
        return con

class American(Human):
    def __init__(self):
        super(Human, self).__init__()
        self.lang = "Eglish"
        self.country = "America"

    def speak(self):
        con = "%s %s :%s" % (self.__class__.__name__, sys._getframe().f_code.co_name,self.lang)
        return con

class German(Human):
    def __init__(self):
        super(Human, self).__init__()
        self.lang = "German"
        self.country = "Germany"

    def speak(self):
        con = "%s %s :%s" % (self.__class__.__name__, sys._getframe().f_code.co_name,self.lang)
        return con

class Worker(Human):
    def __init__(self):
        super(Worker, self).__init__()
        self.work = ''

class Manager(Worker):
    def __init__(self):
        super(Manager, self).__init__()
        self.country = 'beijing'
        self.sex = 'male'
        self.age = '20'
        self.mm = ''

class XiaoMing(Manager,Chinese):
    def __init__(self):
        super(Manager, self).__init__()
        super(Chinese, self).__init__()
        self.sex = 'male'
        self.age = '20'

class Lily(Worker,American):
    def __init__(self):
        super(Worker, self).__init__()
        super(American, self).__init__()
        self.sex = 'famale'
        self.age = '19'

class Davi(Worker,German):
    def __init__(self):
        super(Worker, self).__init__()
        super(German, self).__init__()
        self.sex = 'male'
        self.age = '23'

def run():
    obj = object()
    print(obj)
    print(repr(obj))
    obTest = BaseObject()
    obTest.check_attrs()
    print(obTest)
    print(repr(obTest))
    perTest = Human()
    perTest.check_attrs()
    print(perTest)
    print(repr(perTest))
    manTest = Manager()
    manTest.check_attrs()
    print(manTest)
    print(repr(manTest))
    workTest = Worker()
    workTest.check_attrs()
    print(workTest)
    print(repr(workTest))
    obj = XiaoMing()
    print(obj)
    print(repr(obj))
    obj = Lily()
    print(obj)
    print(repr(obj))
    obj = Davi()
    print(obj)
    print(repr(obj))

def run_test():
    obj = BaseObject()
    pybaselog.test_compare_value(obj.__class__.__module__, "pybase.pyobject")
    pybaselog.test_compare_value(obj.__class__.__name__, "BaseObject")

    obj = Human()
    pybaselog.test_compare_value(obj._name, "protect_name")
    pybaselog.test_compare_value(obj.__class__.__module__, "auto_test.pyobject_test")
    pybaselog.test_compare_value(obj.__class__.__name__, "Human")
    pybaselog.test_compare_value(obj.listen(), "Human listen")


    obj = Worker()
    pybaselog.test_compare_value(obj.__class__.__module__, "auto_test.pyobject_test")
    pybaselog.test_compare_value(obj.__class__.__name__, "Worker")
    pybaselog.test_compare_value(obj.listen(), "Worker listen")

    obj = Manager()
    pybaselog.test_compare_value(obj.__class__.__module__, "auto_test.pyobject_test")
    pybaselog.test_compare_value(obj.__class__.__name__, "Manager")

    obj = XiaoMing()
    pybaselog.test_compare_value(obj.__class__.__module__, "auto_test.pyobject_test")
    pybaselog.test_compare_value(obj.__class__.__name__, "XiaoMing")

    obj = Lily()
    pybaselog.test_compare_value(obj.__class__.__module__, "auto_test.pyobject_test")
    pybaselog.test_compare_value(obj.__class__.__name__, "Lily")

    obj = Davi()
    pybaselog.test_compare_value(obj.__class__.__module__, "auto_test.pyobject_test")
    pybaselog.test_compare_value(obj.__class__.__name__, "Davi")

def test():
    #run()
    run_test()
