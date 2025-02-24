@echo off
:loop
python ajout_bdd.py
echo Scraping terminé. Relance dans 2 secondes...
timeout /t 2
goto loop
