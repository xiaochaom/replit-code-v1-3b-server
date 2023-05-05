bind = "127.0.0.1:8801"
workers = 1
daemon = True
loglevel = "info"
errorlog = "/var/log/gunicorn-error.log"
accesslog = "/var/log/gunicorn-access.log"
