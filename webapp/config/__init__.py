# -*- coding: utf-8 -*-
"""
使用 PyMySQL 驱动连接 MySQL（纯 Python 实现，Windows 下无需编译 mysqlclient）
"""
try:
    import pymysql

    pymysql.install_as_MySQLdb()
except ImportError:  # 允许在未安装 pymysql 时执行 manage.py check 等基础命令
    pass
