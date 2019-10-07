kill -9 $(sudo lsof -n -i :6050| grep LISTEN | awk '{print $2}') &&
virtualenv -p "/usr/bin/python3.6" dataone_env 
&& source "dataone_env/bin/activate" 
&& pip3 install -r requirements.txt 
&& iptables -A INPUT -p tcp --dport 6001 -j ACCEPT 
&& nohup python run.py > run.log &