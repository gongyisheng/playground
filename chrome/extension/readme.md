# Chrome Extension

## architecture
- background.js
- manifest.json
- popup.html
- popup.js
- icon.png

## terminology
- manifest: metadata of resource / permission / identification
- service worker: a worker running in background, listening to events, no access to DOM
- content script: run js in content of a web page
- toolbar action: execute code or show popup when user click extension toolbar icon
- side panel: custom ui in side 
- declarative net request: intercept / modify network requests 

## note
- accessing response body is not allowed