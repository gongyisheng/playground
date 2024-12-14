# Domain Information Groper - dig
## query A record - ipv4
```
dig example.com A

; <<>> DiG 9.10.6 <<>> example.com A
;; global options: +cmd
;; Got answer:
;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: 26782
;; flags: qr rd ra; QUERY: 1, ANSWER: 1, AUTHORITY: 0, ADDITIONAL: 0

;; QUESTION SECTION:
;example.com.                   IN      A

;; ANSWER SECTION:
example.com.            280     IN      A       93.184.215.14

;; Query time: 80 msec
;; SERVER: 100.96.3.193#53(100.96.3.193)
;; WHEN: Sun Dec 01 03:47:30 PST 2024
;; MSG SIZE  rcvd: 45
```
## query AAAA record - ipv6
```
dig example.com AAAA

; <<>> DiG 9.10.6 <<>> example.com AAAA
;; global options: +cmd
;; Got answer:
;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: 25270
;; flags: qr rd ra; QUERY: 1, ANSWER: 1, AUTHORITY: 0, ADDITIONAL: 0

;; QUESTION SECTION:
;example.com.                   IN      AAAA

;; ANSWER SECTION:
example.com.            300     IN      AAAA    2606:2800:21f:cb07:6820:80da:af6b:8b2c

;; Query time: 88 msec
;; SERVER: 100.96.3.193#53(100.96.3.193)
;; WHEN: Sun Dec 01 03:47:53 PST 2024
;; MSG SIZE  rcvd: 57
```
## query MX record - mail exchange
```
dig example.com MX

; <<>> DiG 9.10.6 <<>> example.com MX
;; global options: +cmd
;; Got answer:
;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: 64471
;; flags: qr rd ra ad; QUERY: 1, ANSWER: 1, AUTHORITY: 0, ADDITIONAL: 1

;; OPT PSEUDOSECTION:
; EDNS: version: 0, flags:; udp: 1232
;; QUESTION SECTION:
;example.com.                   IN      MX

;; ANSWER SECTION:
example.com.            1800    IN      MX      0 .

;; Query time: 80 msec
;; SERVER: 100.96.3.193#53(100.96.3.193)
;; WHEN: Sun Dec 01 03:48:21 PST 2024
;; MSG SIZE  rcvd: 55
```
## query CNAME record - canonical name
```
dig www.example.com CNAME

; <<>> DiG 9.10.6 <<>> www.example.com CNAME
;; global options: +cmd
;; Got answer:
;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: 45951
;; flags: qr rd ra ad; QUERY: 1, ANSWER: 0, AUTHORITY: 1, ADDITIONAL: 1

;; OPT PSEUDOSECTION:
; EDNS: version: 0, flags:; udp: 1232
;; QUESTION SECTION:
;www.example.com.               IN      CNAME

;; AUTHORITY SECTION:
example.com.            1629    IN      SOA     ns.icann.org. noc.dns.icann.org. 2024081460 7200 3600 1209600 3600

;; Query time: 79 msec
;; SERVER: 100.96.3.193#53(100.96.3.193)
;; WHEN: Sun Dec 01 03:52:43 PST 2024
;; MSG SIZE  rcvd: 109
```
## query TXT record
```
dig example.com TXT

; <<>> DiG 9.10.6 <<>> example.com TXT
;; global options: +cmd
;; Got answer:
;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: 61858
;; flags: qr rd ra ad; QUERY: 1, ANSWER: 2, AUTHORITY: 0, ADDITIONAL: 1

;; OPT PSEUDOSECTION:
; EDNS: version: 0, flags:; udp: 1232
;; QUESTION SECTION:
;example.com.                   IN      TXT

;; ANSWER SECTION:
example.com.            1800    IN      TXT     "wgyf8z8cgvm2qmxpnbnldrcltvk4xqfn"
example.com.            1800    IN      TXT     "v=spf1 -all"

;; Query time: 95 msec
;; SERVER: 100.96.3.193#53(100.96.3.193)
;; WHEN: Sun Dec 01 03:51:19 PST 2024
;; MSG SIZE  rcvd: 109
```
## other
```
# query using a specific dns server
dig @8.8.8.8 example.com A

# short answer
dig +short example.com

# reverse dns lookup
dig -x 93.184.215.14

# show stats
dig -x 93.184.215.14 +stats
```