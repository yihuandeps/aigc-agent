@echo off
rem agent CLI launcher - no venv activation needed
rem Add this folder to PATH to run "agent" from anywhere
set PYTHONIOENCODING=utf-8
"%~dp0.venv\Scripts\python.exe" -c "from aigc_agent.interfaces.cli.main import main; main()" %*
