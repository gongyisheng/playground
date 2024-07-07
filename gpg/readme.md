# generate gpg key
```
gpg --full-generate-key
```
ref: https://docs.github.com/en/authentication/managing-commit-signature-verification/generating-a-new-gpg-key  

# list gpg key
```
gpg --list-secret-keys --keyid-format LONG
```

# show gpg public key
```
gpg --armor --export <key>
```

# backup gpg key
```
gpg --export --export-options backup --output public.gpg
gpg --export-secret-keys --export-options backup --output private.gpg
gpg --export-ownertrust > trust.gpg
```
ref: https://www.howtogeek.com/816878/how-to-back-up-and-restore-gpg-keys-on-linux/  

# import gpg key
```
gpg --import public.gpg
gpg --import private.gpg
gpg --import-ownertrust trust.gpg
```
ref: https://www.howtogeek.com/816878/how-to-back-up-and-restore-gpg-keys-on-linux/  
