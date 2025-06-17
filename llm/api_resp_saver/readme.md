# API Response Saver Chrome Extension

A Chrome extension that captures and saves API responses while you browse. This is useful for debugging, testing, and analyzing web applications.

## Features

- Captures all XHR and Fetch API responses
- Saves response URL, method, status code, and body
- View responses in a popup interface
- Download all captured responses as JSON
- Clear captured responses
- Pretty-prints JSON responses
- Timestamps for all captured responses

## Installation

1. Clone or download this repository
2. Open Chrome and go to `chrome://extensions/`
3. Enable "Developer mode" in the top right
4. Click "Load unpacked" and select the extension directory

## Usage

1. Click the extension icon in your Chrome toolbar to open the popup
2. Browse websites normally - the extension will automatically capture API responses
3. Click on any response in the list to view its details
4. Use the "Download All" button to save all responses as a JSON file
5. Use the "Clear All" button to remove all captured responses

## Development

The extension consists of the following files:

- `manifest.json`: Extension configuration
- `background.js`: Background script that captures API responses
- `popup.html`: Popup interface
- `popup.js`: Popup interface logic
- `icons/`: Extension icons

## Permissions

The extension requires the following permissions:

- `webRequest`: To capture API responses
- `storage`: To store captured responses
- `downloads`: To save responses as files
- `tabs`: To interact with browser tabs

## Notes

- The extension only captures XHR and Fetch API responses
- Responses are stored in Chrome's local storage
- Large responses may be truncated
- The extension respects Chrome's privacy settings

## Contributing

Feel free to submit issues and enhancement requests!
