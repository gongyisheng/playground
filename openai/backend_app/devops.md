# HTTPS related
https://diamondfsd.com/lets-encrytp-hand-https/

# HTTPS auto renew
0 0 1 * * certbot renew --pre-hook "service nginx stop" --post-hook "service nginx start"