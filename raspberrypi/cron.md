# crontab usage
# list job
`crontab -l`
# add cronjob
`crontab -e -u <USER>`
`* * * * * <COMMAND>` (remember to save log if needed)
# cronjob save log (not recommend)
`sudo vim /etc/rsyslog.d/50-default.conf`
# cronjob execution record
`grep CRON /var/log/syslog`
# chmod of file if run with root user
`chmod +x <FILE>`