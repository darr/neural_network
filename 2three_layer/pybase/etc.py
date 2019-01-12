#!/usr/bin/python
# -*- coding: utf-8 -*-
########################################
# File name : etc.py
# Create date : 2013-06-20 22:29
# Modified date : 2018-10-24 19:25
# Author : DARREN
# Email : lzygzh@126.com
########################################

#about_print_log
LOG_PRINT_LEVEL = 'info'
LOG_PRINT_ALL_INFO_LEVEL = 'warning'

#about log file
LOG_FILE_NAME ='logfile'
LOG_FILE_PATH ='./log/'

#LOG_LEVEL = 'debug'
LOG_WRITE_LEVEL = 'debug'
LOG_WRITE_DB_LEVEL = 'warning'
LOG_LEVEL_DIC = {}
IS_INIT = False
TMP_FILE_PATH = "./tmp_file/"
ENV_FILE_PATH = "./env_file/"

#about log database
PYLOG_DATABASE_NAME = 'pylog'
PYLOG_DATABASE_PATH = './db/'
PYLOG_DATABASE_HOST = 'localhost'
PYLOG_TABLE_NAME = 'pylog'
PYLOG_DATABASE_CHARSET = 'utf8'
#TABLE_CREATE_STR = ''

DATABASE_USE = 'you'
DATABASE_HOST = 'localhost'
DATABASE_PASSWORD = '888888'
DATABASE_CHARSET = 'utf8'
DATABASE_NAME = 'dbname'
DATABASE_TABLE_NAME = 'table_test'
TABLE_TEST  =''' CREATE TABLE IF NOT EXISTS `table_test` (
                `type_id` int(11) NOT NULL AUTO_INCREMENT,
                `pid` int(11) DEFAULT '0' COMMENT '父分类',
                `ut` int(11) DEFAULT '0' COMMENT '更新时间',
                `ct` int(11) DEFAULT '0' COMMENT '创建时间',
                `name` varchar(64) DEFAULT NULL COMMENT '分类名称',
                `enname` varchar(255) DEFAULT NULL COMMENT '分类英文名称',
                `is_hide` tinyint(1) DEFAULT '0' COMMENT '是否隐藏，0可见,1 不可见',
                PRIMARY KEY (`type_id`),
                KEY `pid` (`pid`)
                   ) ENGINE=MyISAM  DEFAULT CHARSET=utf8 AUTO_INCREMENT=1;
                        '''
