# -*- coding: utf-8 -*-
"""
Created on Sun Apr 02 13:42:50 2017
@author: Saber
"""

import ConfigParser
import os

#获取config配置文件

def getConfig(section,key):
    config=ConfigParser.ConfigParser()
    path="./testdb.conf"
    # path=os.path.split(os.path.realpath(__file__))[0]+'\\testdb.conf'
    config.read(path)
    # print config,path
    return config.get(section,key)