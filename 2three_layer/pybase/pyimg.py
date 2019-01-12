#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pyimg.py
# Create date : 2015-02-25 14:02
# Modified date : 2018-10-16 12:51
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################


import sys
if sys.version > '3':
    from io import BytesIO
    import urllib.request as urllib2
    from . import pylog
else:
    import cStringIO
    import urllib2
    import pylog

from PIL import Image

def get_img_info(url):
    file = None
    try:
        file = urllib2.urlopen(url)
    except urllib2.URLError as e:
        pylog.warning("Reason:%s url:%s" % (e.reason,url))
        return False,0,0,0
    tmpIm = None
    if sys.version > '3':
        tmpIm = BytesIO(file.read())
    else:
        tmpIm = cStringIO.StringIO(file.read())
    im = Image.open(tmpIm)
    imgType = im.format
    (imgWidth,imgHeight) = im.size
    return True,imgType,imgWidth,imgHeight
