@echo off
rmdir %~dp0_local_env /s /q
mkdir %~dp0_local_env
mkdir %~dp0_local_env\_tmp 
mkdir %~dp0_local_env\userprofile
mkdir %~dp0_local_env\userroaming
mkdir %~dp0_local_env\localappdata