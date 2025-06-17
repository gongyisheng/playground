// Store for captured responses
let capturedResponses = [];

// Listen for web requests
chrome.webRequest.onCompleted.addListener(
  async function(details) {
    // Only capture API responses
    if (details.type === "xmlhttprequest" || details.type === "fetch") {
      try {
        // Create response object with available details
        const responseData = {
          url: details.url,
          method: details.method,
          timestamp: new Date().toISOString(),
          statusCode: details.statusCode,
          type: details.type,
          tabId: details.tabId,
          requestHeaders: details.requestHeaders,
          responseHeaders: details.responseHeaders
        };

        // Add to captured responses
        capturedResponses.push(responseData);

        // Store in chrome.storage
        chrome.storage.local.set({ 
          capturedResponses: capturedResponses 
        });

        // Notify popup
        chrome.runtime.sendMessage({
          type: 'newResponse',
          data: responseData
        });

        console.log('Captured response:', responseData);
      } catch (error) {
        console.error('Error capturing response:', error);
      }
    }
  },
  { urls: ["<all_urls>"] }
);

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'getResponses') {
    sendResponse({ responses: capturedResponses });
  } else if (request.type === 'clearResponses') {
    capturedResponses = [];
    chrome.storage.local.set({ capturedResponses: [] });
    sendResponse({ success: true });
  } else if (request.type === 'downloadResponses') {
    const blob = new Blob([JSON.stringify(capturedResponses, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    
    chrome.downloads.download({
      url: url,
      filename: `api-responses-${timestamp}.json`,
      saveAs: true
    });
    
    sendResponse({ success: true });
  }
  return true;
}); 