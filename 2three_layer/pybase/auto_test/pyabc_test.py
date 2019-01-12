#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pyabc_test.py
# Create date : 2018-10-15 17:18
# Modified date : 2018-10-15 18:08
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from abc import abstractmethod

from pybase.pyobject import BaseObject
from pybase import pylog


class Biology(BaseObject):
    def __init__(self):
        super(BaseObject, self).__init__()
        self.age = '20'
        self.name = 'biology_name'

    def move(self):
        ret = "%s move style is %s" % (self.__class__.__name__,self.move_func())
        return ret

    @abstractmethod
    def move_func(self):
       '''Implement in subclass'''

class Animal(Biology):
    def __init__(self):
        super(Biology, self).__init__()
        self.sex = "male"

    def move_func(self):
        con = "Animal move"
        return con

class Plant(Biology):
    def __init__(self):
        super(Biology, self).__init__()

    def move_func(self):
        con = "Plant just stay"
        return con

class Cat(Animal):
    def __init__(self):
        super(Animal, self).__init__()

    def move_func(self):
        return self.walk()

    def run(self):
        con = "Cat run"
        return con

    def walk(self):
        con = "Cat walk"
        return con

    def eat(self,food):
        con = "Cat eat"
        return con

class Dog(Animal):
    ''' a class Dog'''
    def __init__(self):
        super(Animal, self).__init__()

    def move_func(self):
        return self.run()

    def run(self):
        con = "Dog run"
        return con

    def walk(self):
        con = "%s %s" % (self.__class__.__name__,sys._getframe().f_code.co_name)
        pylog.info(con)
        return con

class Fish(Animal):
    ''' a class Fish'''
    def __init__(self):
        super(Animal, self).__init__()

    def swim(self):
        con = "Fish swim"
        return con

    def move_func(self):
        return self.swim()

def test():
    obj = Animal()
    pylog.test_compare_value("Animal move style is Animal move",obj.move())
    obj = Plant()
    pylog.test_compare_value("Plant move style is Plant just stay",obj.move())
    obj = Cat()
    pylog.test_compare_value("Cat move style is Cat walk",obj.move())
    obj = Dog()
    pylog.test_compare_value("Dog move style is Dog run",obj.move())
    obj = Fish()
    pylog.test_compare_value("Fish move style is Fish swim",obj.move())
