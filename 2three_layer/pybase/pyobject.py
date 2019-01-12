#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pyobject.py
# Create date : 2013-08-20 12:52
# Modified date : 2018-10-20 13:32
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################

#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

if sys.version > '3':
    from abc import ABC
else:
    from abc import ABCMeta

if sys.version > '3':
    class AbcBase(ABC):
        def __init__(self):
            super(AbcBase, self).__init__()
else:
    class AbcBase():
        __metaclass__ = ABCMeta
        def __init__(self):
            pass


class BaseObject(AbcBase, object):
    def __init__(self):
        super(BaseObject, self).__init__()
        self.name = 'baseobject_name'
        self._name = "protect_name"
        self.__name = "private_name"

    def __del__(self):
        return

    def __str__(self):
        return 'class: %s\n    attrs: %s' % (self.__class__.__name__, self.__gather_attrs())

    def __repr__(self):
        dir_str = dir(self)
        return 'class: %s\n    dir: %s' % (self.__class__.__name__, dir_str)

    def __get_dir_attrs(self):
        dir_list = dir(self)
        dir_str = ''
        for attr in dir_list:
            dir_str = dir_str +  '%s : %s' % (attr, getattr(self, attr))
        return dir_str

    def __gather_attrs(self):
        attrs = []
        for key in sorted(self.__dict__):
            attrs.append('%s=%s' % (key, getattr(self, key)))
        return ', '.join(attrs)

    def check_attrs(self):
        '''check is all attrs of the class have a value:
            all have value return True,else return False '''
        for key in sorted(self.__dict__):
            if not getattr(self, key):
                #content = "%s attrs: %s is %s\n" % (self.__class__.__name__, key, getattr(self, key))
                return False
        return True
