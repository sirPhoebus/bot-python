# bot-python
python -m venv venv
venv\scripts\activate

Run pip install for the missing libraries <br/>
Training 
    Run train.py to visualize the best Stop Loss ratio / profit ratio for the training period<br/>


how to keep a python running when I close my ubuntu putty session ?

You can use the command "nohup" to keep a python script running after you close your Ubuntu PuTTY session. For example, if your script is called "script.py," you would run the command "nohup python script.py &" in your PuTTY session. The "&" at the end of the command tells the terminal to run the command in the background, and "nohup" prevents the script from being terminated when you close the session.

Launch the bot with : 
nohup python3 bot.py > mylog.doc

Check if python is running in the bg:
pgrep -fl python