bind = "0.0.0.0:8800"
workers = 1
daemon = True
loglevel = "info"
errorlog = "/var/log/gunicorn-error.log"
accesslog = "/var/log/gunicorn-access.log"
