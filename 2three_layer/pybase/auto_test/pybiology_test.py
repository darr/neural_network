#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pybiology_test.py
# Create date : 2018-10-10 16:47
# Modified date : 2018-10-15 17:59
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from pybase.pyobject import BaseObject
from pybase import pylog

class Biology(BaseObject):
    ''' a class Biology'''
    def __init__(self):
        super(BaseObject, self).__init__()
        self.age = '20'
        self.name = 'biology_name'

    def survival(self):
        con = "%s %s" % (self.__class__.__name__,sys._getframe().f_code.co_name)
        pylog.info(con)
        return con

    def reproduce(self):
        con = "%s %s" % (self.__class__.__name__,sys._getframe().f_code.co_name)
        pylog.info(con)
        return con

class Animal(Biology):
    ''' a class Animal'''
    def __init__(self):
        super(Biology, self).__init__()
        self.sex = "male"

    def move(self):
        con = "Animal move"
        pylog.info(con)
        return con

class Plant(Biology):
    ''' a class Plant'''
    def __init__(self):
        super(Biology, self).__init__()

    def collect_sunshine(self):
        con = "%s %s" % (self.__class__.__name__,sys._getframe().f_code.co_name)
        pylog.info(con)
        return con

class Cat(Animal):
    ''' a class Cat'''
    def __init__(self):
        super(Animal, self).__init__()

    def move(self):
        self.run()

    def run(self):
        con = "%s %s" % (self.__class__.__name__,sys._getframe().f_code.co_name)
        pylog.info(con)
        return con

    def walk(self):
        con = "%s %s" % (self.__class__.__name__,sys._getframe().f_code.co_name)
        pylog.info(con)
        return con

    def speak(self):
        con = "%s %s" % (self.__class__.__name__,sys._getframe().f_code.co_name)
        pylog.info(con)
        return con

    def eat(self,food):
        con = "%s %s:%s" % (self.__class__.__name__,sys._getframe().f_code.co_name,food)
        pylog.info(con)
        return con

class Dog(Animal):
    ''' a class Dog'''
    def __init__(self):
        super(Animal, self).__init__()

    def move(self):
        self.walk()

    def run(self):
        con = "%s %s" % (self.__class__.__name__,sys._getframe().f_code.co_name)
        pylog.info(con)
        return con

    def walk(self):
        con = "%s %s" % (self.__class__.__name__,sys._getframe().f_code.co_name)
        pylog.info(con)
        return con

    def speak(self):
        con = "%s %s" % (self.__class__.__name__,sys._getframe().f_code.co_name)
        pylog.info(con)
        return con

 
class Fish(Animal):
    ''' a class Fish'''
    def __init__(self):
        super(Animal, self).__init__()

    def swim(self):
        con = "%s %s" % (self.__class__.__name__,sys._getframe().f_code.co_name)
        pylog.info(con)
        return con

def run():
    obj = Biology()
    obj.survival()
    obj.reproduce()

    obj = Animal()
    obj.survival()
    obj.reproduce()

    obj = Plant()
    obj.survival()
    obj.reproduce()

    obj = Cat()
    obj.survival()
    obj.reproduce()
    obj.move()
    obj.run()
    obj.eat("Fish")

    obj = Dog()
    obj.survival()
    obj.reproduce()
    obj.move()
    obj.run()

    obj = Fish()
    obj.survival()
    obj.reproduce()

def test():
    #run()
    pass
