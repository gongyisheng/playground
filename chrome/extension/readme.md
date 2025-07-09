# Chrome Extension

## structure
my-extension/
├── manifest.json
├── background.js
├── scripts/
│   ├── content.js
│   └── react.production.min.js
├── popup/
│   ├── popup.html
│   ├── popup.js
│   └── popup.css
└── images/
    ├── icon-16.png
    ├── icon-32.png
    ├── icon-48.png
    └── icon-128.png

## terminology
- manifest: metadata of resource / permission / identification
- service worker: a worker running in background, listening to events, no access to DOM
- content script: run js in content of a web page
- toolbar action: execute code or show popup when user click extension toolbar icon
- side panel: custom ui in side 
- declarative net request: intercept / modify network requests 

## note
- use typescript: npm package chrome-types can be used for auto-completion
- accessing response body is not allowed